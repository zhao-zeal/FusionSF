from __future__ import annotations

import torch
from torch import nn

from src.models.fusionSF_3modal import Attention, CrossTransformer, Transformer
from src.models.modules.attention_modules import FeedForward, PreNorm


class PrototypeHorizonGRU(nn.Module):
    """Stage-2 pipeline prototype. This is not a FusionSF architecture or result."""

    model_family = "prototype_horizon_gru"
    architecture_version = "prototype_horizon_gru_v1"

    def __init__(self, weather_dim: int, hidden_dim: int = 64, max_pred_len: int = 288):
        super().__init__()
        self.weather_dim = weather_dim
        self.ts_encoder = nn.GRU(1, hidden_dim, batch_first=True)
        self.horizon_queries = nn.Embedding(max_pred_len, hidden_dim)
        self.guide_encoder = nn.Sequential(nn.Linear(weather_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) if weather_dim else None
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.prediction_head = nn.Linear(hidden_dim, 1)

    def forward(self, history_power: torch.Tensor, future_weather: torch.Tensor | None = None, pred_len: int | None = None, **_) -> torch.Tensor:
        pred_len = future_weather.shape[1] if future_weather is not None else pred_len
        if pred_len is None:
            raise ValueError("pred_len is required in Power mode")
        tokens, state = self.ts_encoder(history_power)
        query = state[-1][:, None] + self.horizon_queries(torch.arange(pred_len, device=history_power.device))[None]
        attended, _ = self.cross_attention(query, tokens, tokens)
        if future_weather is not None:
            attended = attended + self.guide_encoder(future_weather)
        return self.prediction_head(self.fusion_norm(attended + query))


class DynamicFusionSFTransformer(nn.Module):
    """FusionSF Transformer block without a fixed learned sequence table."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            sequence = attention(sequence) + sequence
            sequence = feed_forward(sequence) + sequence
        return self.norm(sequence)


class FusionSFSolar(nn.Module):
    """Official horizon-native solar migration of FusionSF."""

    model_family = "FusionSF"
    architecture_version = "fusionsf_solar_v1"

    def __init__(
        self,
        weather_dim: int,
        time_dim: int = 8,
        seq_len: int = 336,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        max_pred_len: int = 288,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.weather_dim = weather_dim
        self.seq_len = seq_len
        self.dim = dim
        self.ts_embedding = nn.Linear(1 + time_dim, dim)
        self.ts_encoder = Transformer(dim, seq_len, depth, heads, dim_head, dim * 4, dropout)
        self.horizon_embedding = nn.Embedding(max_pred_len, dim)
        self.future_time_embedding = nn.Linear(time_dim, dim)
        self.horizon_cross_attention = CrossTransformer(
            dim, depth, heads, dim_head, dim * 4, (seq_len, 1), dropout,
            use_rotary=False, use_glu=True,
        )
        self.guide_embedding = nn.Linear(weather_dim, dim) if weather_dim else None
        self.guide_encoder = DynamicFusionSFTransformer(dim, depth, heads, dim_head, dim * 4, dropout) if weather_dim else None
        self.fusion_encoder = DynamicFusionSFTransformer(dim, depth, heads, dim_head, dim * 4, dropout)
        self.prediction_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def encode(
        self,
        history_power: torch.Tensor,
        history_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        future_weather: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if history_power.shape[:2] != history_time_features.shape[:2] or history_power.shape[-1] != 1:
            raise ValueError("history tensors must align as [B, seq_len, *]")
        if history_power.shape[1] != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {history_power.shape[1]}")
        batch, pred_len = future_time_features.shape[:2]
        history_tokens = self.ts_encoder(self.ts_embedding(torch.cat([history_power, history_time_features], dim=-1)))
        offsets = torch.arange(pred_len, device=history_power.device)
        horizon_queries = self.horizon_embedding(offsets)[None].expand(batch, -1, -1)
        horizon_queries = horizon_queries + self.future_time_embedding(future_time_features)
        horizon_representation, _ = self.horizon_cross_attention(
            history_tokens, horizon_queries, None, None
        )
        if future_weather is None:
            guide_representation = torch.zeros_like(horizon_representation)
            fusion_input = horizon_representation
        else:
            if self.guide_encoder is None or future_weather.shape[:2] != (batch, pred_len):
                raise ValueError("future weather must align as [B, pred_len, C]")
            guide_representation = self.guide_encoder(self.guide_embedding(future_weather))
            fusion_input = horizon_representation + guide_representation
        fusion_embedding = self.fusion_encoder(fusion_input)
        return {
            "ts_embedding": history_tokens,
            "horizon_query": horizon_queries,
            "horizon_representation": horizon_representation,
            "guide_embedding": guide_representation,
            "fusion_embedding": fusion_embedding,
        }

    def forward(
        self,
        history_power: torch.Tensor,
        history_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        future_weather: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ):
        embeddings = self.encode(history_power, history_time_features, future_time_features, future_weather)
        prediction = self.prediction_head(embeddings["fusion_embedding"])
        return (prediction, embeddings) if return_embeddings else prediction
