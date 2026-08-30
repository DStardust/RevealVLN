"""Low-capacity invariant failure-risk model for frozen ETP embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


FEATURE_NAMES = (
    "instruction_current_cosine",
    "instruction_temporal_cosine",
    "instruction_native_cosine",
    "instruction_alternative_cosine",
    "current_temporal_cosine",
    "current_native_cosine",
    "current_alternative_cosine",
    "temporal_native_cosine",
    "temporal_alternative_cosine",
    "native_alternative_cosine",
    "instruction_alternative_minus_native",
    "current_alternative_minus_native",
    "temporal_alternative_minus_native",
    "native_distance_scaled",
    "alternative_distance_scaled",
    "alternative_minus_native_distance_scaled",
)


def causal_scalar_features(
    instruction: torch.Tensor,
    current: torch.Tensor,
    temporal: torch.Tensor,
    native: torch.Tensor,
    alternative: torch.Tensor,
    immediate_costs: torch.Tensor,
) -> torch.Tensor:
    def cosine(left, right):
        return F.cosine_similarity(left, right, dim=-1, eps=1e-6)

    instruction_current = cosine(instruction, current)
    instruction_temporal = cosine(instruction, temporal)
    instruction_native = cosine(instruction, native)
    instruction_alternative = cosine(instruction, alternative)
    current_temporal = cosine(current, temporal)
    current_native = cosine(current, native)
    current_alternative = cosine(current, alternative)
    temporal_native = cosine(temporal, native)
    temporal_alternative = cosine(temporal, alternative)
    native_alternative = cosine(native, alternative)
    return torch.stack((
        instruction_current,
        instruction_temporal,
        instruction_native,
        instruction_alternative,
        current_temporal,
        current_native,
        current_alternative,
        temporal_native,
        temporal_alternative,
        native_alternative,
        instruction_alternative - instruction_native,
        current_alternative - current_native,
        temporal_alternative - temporal_native,
        immediate_costs[..., 0],
        immediate_costs[..., 1],
        immediate_costs[..., 1] - immediate_costs[..., 0],
    ), dim=-1)


class ScalarETPFailureRiskHead(nn.Module):
    """A standardized logistic model that cannot memorize raw scene features."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        if tuple(mean.shape) != (len(FEATURE_NAMES),) or mean.shape != scale.shape:
            raise ValueError("scalar failure-risk normalization shape mismatch")
        self.register_buffer("mean", mean.detach().float())
        self.register_buffer("scale", scale.detach().float().clamp_min(1e-4))
        self.classifier = nn.Linear(len(FEATURE_NAMES), 1)

    def forward(
        self,
        instruction: torch.Tensor,
        current: torch.Tensor,
        temporal: torch.Tensor,
        native: torch.Tensor,
        alternative: torch.Tensor,
        immediate_costs: torch.Tensor,
    ) -> torch.Tensor:
        features = causal_scalar_features(
            instruction, current, temporal, native, alternative,
            immediate_costs,
        )
        return self.classifier((features - self.mean) / self.scale).squeeze(-1)
