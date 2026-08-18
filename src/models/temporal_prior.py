"""Small, auditable temporal-prior injectors for FusionSF ablations."""

import torch
from torch import nn


class TemporalPriorInjector(nn.Module):
    """Add a global temporal prior to every FusionSF power token.

    ``mlp`` is a parameter-matched control that only sees the flattened historical
    power window. ``chronos`` consumes an offline, frozen Chronos-2 representation.
    """

    VALID_MODES = {"none", "mlp", "chronos"}

    def __init__(
        self,
        mode: str,
        fusion_dim: int,
        history_length: int,
        chronos_dim: int = 768,
        mlp_hidden_dim: int = 553,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")
        self.mode = mode
        self.fusion_dim = int(fusion_dim)
        self.history_length = int(history_length)
        self.chronos_dim = int(chronos_dim)
        if mode == "none":
            self.projection = None
        elif mode == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(self.history_length, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, self.fusion_dim),
            )
        else:
            self.projection = nn.Linear(self.chronos_dim, self.fusion_dim)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        power_tokens: torch.Tensor,
        history_power: torch.Tensor,
        chronos_representation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if power_tokens.ndim != 3:
            raise ValueError("power_tokens must have shape [B, T, D]")
        if history_power.ndim != 3 or history_power.shape[:2] != power_tokens.shape[:2]:
            raise ValueError("history_power must align with power_tokens as [B, T, C]")
        prior = self.project_prior(power_tokens, history_power, chronos_representation)
        if prior is None:
            return power_tokens
        return power_tokens + prior.unsqueeze(1)

    def project_prior(
        self,
        power_tokens: torch.Tensor,
        history_power: torch.Tensor,
        chronos_representation: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Project one global prior without deciding where it is injected."""
        if power_tokens.ndim != 3:
            raise ValueError("power_tokens must have shape [B, T, D]")
        if history_power.ndim != 3 or history_power.shape[:2] != power_tokens.shape[:2]:
            raise ValueError("history_power must align with power_tokens as [B, T, C]")
        if self.mode == "none":
            return None
        if self.mode == "mlp":
            if history_power.shape[1] != self.history_length:
                raise ValueError("history_power length does not match the configured history_length")
            prior = self.projection(history_power[..., 0])
        else:
            if chronos_representation is None:
                raise ValueError("chronos mode requires chronos_representation")
            if chronos_representation.shape != (power_tokens.shape[0], self.chronos_dim):
                raise ValueError(
                    "chronos_representation must have shape "
                    f"[B, {self.chronos_dim}], got {tuple(chronos_representation.shape)}"
                )
            prior = self.projection(chronos_representation.to(power_tokens.dtype))
        return prior
