"""Causal pairwise net-advantage head for sparse topology interventions."""

from __future__ import annotations

import torch
from torch import nn


class PairwiseNetAdvantageHead(nn.Module):
    """Predict whether one alternative beats the frozen policy's native branch."""

    def __init__(self, input_dim: int = 768, projection_dim: int = 96) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(input_dim)
        self.project = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(projection_dim * 6 + 2),
            nn.Linear(projection_dim * 6 + 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.better_logit = nn.Linear(128, 1)
        self.positive_gain = nn.Sequential(nn.Linear(128, 1), nn.Softplus())

    def forward(
        self,
        instruction: torch.Tensor,
        current_history: torch.Tensor,
        temporal_history: torch.Tensor,
        native: torch.Tensor,
        alternative: torch.Tensor,
        immediate_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = (
            instruction,
            current_history,
            temporal_history,
            native,
            alternative,
            alternative - native,
        )
        encoded = [self.project(self.normalizer(value)) for value in raw]
        fused = self.fusion(torch.cat([*encoded, immediate_costs], dim=-1))
        return self.better_logit(fused).squeeze(-1), self.positive_gain(fused).squeeze(-1)

