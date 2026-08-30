"""Causal pre-action risk head for the frozen ETP policy."""

from __future__ import annotations

import torch
from torch import nn


class ETPFailureRiskHead(nn.Module):
    """Predict eventual frozen-policy failure from one online branch state."""

    def __init__(self, input_dim: int = 768, projection_dim: int = 64) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(input_dim)
        self.project = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim * 6 + 2),
            nn.Linear(projection_dim * 6 + 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        instruction: torch.Tensor,
        current_history: torch.Tensor,
        temporal_history: torch.Tensor,
        native: torch.Tensor,
        alternative: torch.Tensor,
        immediate_costs: torch.Tensor,
    ) -> torch.Tensor:
        raw = (
            instruction,
            current_history,
            temporal_history,
            native,
            alternative,
            alternative - native,
        )
        encoded = [self.project(self.normalizer(value)) for value in raw]
        return self.classifier(
            torch.cat([*encoded, immediate_costs], dim=-1)
        ).squeeze(-1)
