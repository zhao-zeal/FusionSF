from __future__ import annotations

import torch
from torch import nn


class FusionSFSolar(nn.Module):
    """Horizon-native FusionSF adapter; never repeats/resamples history to the forecast length."""

    def __init__(self, nwp_dim: int, hidden_dim: int = 64, max_pred_len: int = 288):
        super().__init__()
        self.nwp_dim = nwp_dim
        self.ts_encoder = nn.GRU(1, hidden_dim, batch_first=True)
        self.horizon_queries = nn.Embedding(max_pred_len, hidden_dim)
        self.guide_encoder = nn.Sequential(nn.Linear(nwp_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) if nwp_dim else None
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.prediction_head = nn.Linear(hidden_dim, 1)

    def encode(self, history_power: torch.Tensor, future_nwp: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if history_power.ndim != 3 or history_power.shape[-1] != 1:
            raise ValueError("history_power must be [B, seq_len, 1]")
        if future_nwp is None:
            pred_len = 1
        else:
            if future_nwp.ndim != 3 or future_nwp.shape[-1] != self.nwp_dim:
                raise ValueError("future_nwp must be [B, pred_len, C]")
            pred_len = future_nwp.shape[1]
        ts_tokens, ts_state = self.ts_encoder(history_power)
        query = ts_state[-1][:, None, :] + self.horizon_queries(torch.arange(pred_len, device=history_power.device))[None]
        attended, _ = self.cross_attention(query, ts_tokens, ts_tokens)
        if future_nwp is not None:
            guide = self.guide_encoder(future_nwp)
            fusion = self.fusion_norm(attended + query + guide)
        else:
            guide = torch.zeros_like(attended)
            fusion = self.fusion_norm(attended + query)
        return {"ts_embedding": ts_state[-1], "guide_embedding": guide, "fusion_embedding": fusion}

    def forward(self, history_power: torch.Tensor, future_nwp: torch.Tensor | None = None, pred_len: int | None = None) -> torch.Tensor:
        if future_nwp is None:
            if pred_len is None:
                raise ValueError("pred_len is required in Power mode")
            embeddings = self._encode_power(history_power, pred_len)
        else:
            embeddings = self.encode(history_power, future_nwp)
        return self.prediction_head(embeddings["fusion_embedding"])

    def _encode_power(self, history_power: torch.Tensor, pred_len: int) -> dict[str, torch.Tensor]:
        ts_tokens, ts_state = self.ts_encoder(history_power)
        query = ts_state[-1][:, None, :] + self.horizon_queries(torch.arange(pred_len, device=history_power.device))[None]
        attended, _ = self.cross_attention(query, ts_tokens, ts_tokens)
        fusion = self.fusion_norm(attended + query)
        return {"ts_embedding": ts_state[-1], "guide_embedding": torch.zeros_like(fusion), "fusion_embedding": fusion}
