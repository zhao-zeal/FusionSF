"""Lightweight FusionSF-to-Chronos-2 correlation adapter.

This is the Stage 4B CoRA-inspired module.  It is a local dual-branch adapter,
not a vendored or complete implementation of an external CoRA model.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class FusionChronosCorrelationAdapter(nn.Module):
    """Inject unpooled FusionSF tokens into frozen Chronos-2 hidden states."""

    def __init__(
        self,
        fusion_dim: int = 64,
        chronos_dim: int = 768,
        heads: int = 12,
        global_hidden: int = 192,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion_projection = nn.Linear(fusion_dim, chronos_dim)
        self.query_norm = nn.LayerNorm(chronos_dim)
        self.fusion_norm = nn.LayerNorm(chronos_dim)
        self.dynamic_attention = nn.MultiheadAttention(
            chronos_dim, heads, dropout=dropout, batch_first=True
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(chronos_dim, global_hidden),
            nn.GELU(),
            nn.Linear(global_hidden, chronos_dim),
        )
        # Exact identity at initialization: the first forward equals Chronos-2.
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(
        self, chronos_tokens: torch.Tensor, fusion_tokens: torch.Tensor
    ) -> torch.Tensor:
        if chronos_tokens.ndim != 3 or fusion_tokens.ndim != 3:
            raise ValueError("chronos_tokens and fusion_tokens must be [batch, tokens, dim]")
        if chronos_tokens.shape[0] != fusion_tokens.shape[0]:
            raise ValueError("Chronos and FusionSF token batches must align")
        projected = self.fusion_norm(self.fusion_projection(fusion_tokens))
        dynamic, _ = self.dynamic_attention(
            self.query_norm(chronos_tokens), projected, projected, need_weights=False
        )
        global_condition = self.global_mlp(projected.mean(dim=1)).unsqueeze(1)
        return chronos_tokens + self.alpha * dynamic + self.beta * global_condition


# Compatibility name retained for the archived Stage 4B terminology.
CoRACorrelationAdapter = FusionChronosCorrelationAdapter


def adapter_forward(
    chronos: nn.Module,
    adapter: nn.Module,
    context: torch.Tensor,
    fusion_tokens: torch.Tensor,
    prediction_length: int = 24,
) -> torch.Tensor:
    """Run the native Chronos-2 quantile head after the correlation adapter.

    Returns ``[batch, quantiles, prediction_length]`` in the original scale.
    """
    patch_size = chronos.chronos_config.output_patch_size
    num_output_patches = (prediction_length + patch_size - 1) // patch_size
    batch_size = context.shape[0]
    group_ids = torch.arange(batch_size, device=context.device)
    encoded, loc_scale, _, _ = chronos.encode(
        context=context,
        group_ids=group_ids,
        num_output_patches=num_output_patches,
    )
    hidden = adapter(encoded.last_hidden_state, fusion_tokens)
    forecast = hidden[:, -num_output_patches:]
    quantile_predictions = chronos.output_patch_embedding(forecast)
    quantile_predictions = rearrange(
        quantile_predictions,
        "b n (q p) -> b q (n p)",
        q=chronos.num_quantiles,
        p=patch_size,
    )
    flattened = rearrange(quantile_predictions, "b q h -> b (q h)")
    flattened = chronos.instance_norm.inverse(flattened, loc_scale)
    quantile_predictions = rearrange(
        flattened, "b (q h) -> b q h", q=chronos.num_quantiles
    )
    return quantile_predictions[..., :prediction_length]


def freeze_backbones(chronos: nn.Module, fusionsf: nn.Module) -> None:
    """Freeze both pretrained backbones; only the adapter remains trainable."""
    chronos.eval().requires_grad_(False)
    fusionsf.eval().requires_grad_(False)


def trainable_parameter_names(module: nn.Module) -> list[str]:
    return [name for name, value in module.named_parameters() if value.requires_grad]
