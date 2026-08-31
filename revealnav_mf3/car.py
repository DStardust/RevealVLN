"""Criterion-aligned risk-constrained switch policy primitives.

MF3ZM-CAR v1 keeps the frozen RCSP v1.1 representation and architecture.  The
revision changes only the training objective: the forward decision is the same
hard ``logit > 0`` switch used at deployment, while the straight-through
surrogate supplies a gradient for the pre-registered risk and scene
constraints.  This module deliberately contains no data loading or public
split access.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from revealnav_mf3.rcsp_v1_1 import (
    ENGINEERED_FEATURE_DIM,
    POLICY_FEATURE_NAMES,
    EngineeredRCSPControl,
    RelativeSemanticSwitchPolicy,
)


CAR_REVISION = "mf3zm_car_v1"
CAR_CHECKPOINT_SCHEMA = "revealnav-mf3zm-car-checkpoint/1"
CAR_POLICY_FEATURE_NAMES = POLICY_FEATURE_NAMES
CAR_ENGINEERED_FEATURE_DIM = ENGINEERED_FEATURE_DIM


class CARZeroSelectionError(RuntimeError):
    """A hard-forward domain has no selected intervention."""


def event_domain_weights(datasets: np.ndarray) -> np.ndarray:
    """Give every domain mass 1/D and every event equal mass within it."""

    values = np.asarray([str(value) for value in datasets])
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("CAR domains must be a non-empty vector")
    domains = sorted(set(values))
    result = np.zeros(len(values), dtype=np.float64)
    for domain in domains:
        mask = values == domain
        result[mask] = 1.0 / (len(domains) * int(mask.sum()))
    if (
        not np.isfinite(result).all()
        or not np.all(result > 0)
        or not math.isclose(float(result.sum()), 1.0, rel_tol=1e-12)
    ):
        raise ValueError("invalid CAR event/domain weights")
    return result


def straight_through_gate(logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(surrogate, hard_mask, probability)``.

    ``surrogate`` is exactly the hard mask in the forward pass and has the
    sigmoid derivative in the backward pass.  The strict comparison is fixed
    at zero; no calibration threshold is learned here.
    """

    if (
        logits.ndim != 1
        or len(logits) == 0
        or not torch.is_floating_point(logits)
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("CAR logits must be a finite non-empty vector")
    probability = torch.sigmoid(logits)
    hard = (logits > 0.0).to(logits.dtype)
    surrogate = hard + probability - probability.detach()
    return surrogate, hard, probability


def _validate_loss_inputs(
    logits: Tensor, target: Tensor, sample_weight: Tensor,
) -> None:
    if (
        logits.ndim != 1
        or target.shape != logits.shape
        or sample_weight.shape != logits.shape
        or len(logits) == 0
        or any(
            not torch.is_floating_point(value)
            or not torch.isfinite(value).all()
            for value in (logits, target, sample_weight)
        )
        or not torch.all(sample_weight > 0)
    ):
        raise ValueError("invalid CAR loss input")


def utility_weighted_preference_loss(
    logits: Tensor,
    delta_utility: Tensor,
    sample_weight: Tensor,
) -> Tensor:
    """Magnitude-weighted native-versus-runner preference loss."""

    _validate_loss_inputs(logits, delta_utility, sample_weight)
    magnitude = sample_weight * torch.abs(delta_utility)
    denominator = torch.sum(magnitude)
    if not bool(denominator > 0):
        raise ValueError("CAR preference cohort has zero utility magnitude")
    label = (delta_utility > 0.0).to(logits.dtype)
    row_loss = F.binary_cross_entropy_with_logits(
        logits, label, reduction="none"
    )
    return torch.sum(magnitude * row_loss) / denominator


def catastrophic_rate_constraint(
    logits: Tensor,
    catastrophic: Tensor,
    sample_weight: Tensor,
    ungated_rate: float,
    *,
    hard_forward: bool = True,
) -> tuple[Tensor, bool]:
    """Return selected catastrophic-rate excess and a zero-selection flag.

    With ``hard_forward=True`` the returned scalar evaluates exactly to the
    event-level hard rate minus ``ungated_rate``.  A domain with no hard
    selection returns a zero-gradient scalar and ``True``; callers must mark
    that candidate infeasible rather than divide by epsilon or treat it as
    safe.  ``hard_forward=False`` is the explicitly named soft-risk control.
    """

    _validate_loss_inputs(logits, catastrophic, sample_weight)
    if (
        not math.isfinite(float(ungated_rate))
        or not 0.0 <= float(ungated_rate) <= 1.0
        or not torch.all((catastrophic == 0.0) | (catastrophic == 1.0))
    ):
        raise ValueError("invalid CAR catastrophic constraint input")
    surrogate, hard, probability = straight_through_gate(logits)
    selected = surrogate if hard_forward else probability
    hard_mass = torch.sum(sample_weight * hard)
    if hard_forward and not bool(hard_mass > 0):
        # Keep the training graph valid, but expose the condition to the
        # caller.  The final hard policy is never allowed to use this state.
        return torch.sum(logits) * 0.0, True
    mass = torch.sum(sample_weight * selected)
    if not bool(torch.isfinite(mass)) or not bool(mass > 0):
        raise CARZeroSelectionError("CAR catastrophic denominator is zero")
    rate = torch.sum(sample_weight * selected * catastrophic) / mass
    result = rate - float(ungated_rate)
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("CAR catastrophic constraint is non-finite")
    return result, False


def selected_utility_constraint(
    logits: Tensor,
    delta_utility: Tensor,
    sample_weight: Tensor,
    subset_mask: Tensor,
    *,
    hard_forward: bool = True,
) -> tuple[Tensor, bool]:
    """Return the constraint ``-mean_selected_utility <= 0``.

    The denominator is the fixed event population mass, not the selected
    count.  This matches the pre-registered ITT/leave-one-scene statistic and
    makes zero selection an explicit zero-utility (therefore failing) state.
    """

    _validate_loss_inputs(logits, delta_utility, sample_weight)
    if (
        subset_mask.ndim != 1
        or subset_mask.shape != logits.shape
        or subset_mask.dtype != torch.bool
        or not bool(subset_mask.any())
    ):
        raise ValueError("invalid CAR scene subset")
    surrogate, hard, probability = straight_through_gate(logits)
    selected = surrogate if hard_forward else probability
    selected_sum = torch.sum(
        sample_weight[subset_mask]
        * selected[subset_mask]
        * delta_utility[subset_mask]
    )
    result = -selected_sum
    zero_selected = not bool(torch.any(hard[subset_mask]))
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("CAR utility constraint is non-finite")
    return result, zero_selected


def projected_dual_update(
    dual: Tensor,
    constraint: Tensor,
    learning_rate: float,
    *,
    maximum: float = 100.0,
) -> Tensor:
    """Projected non-negative dual ascent with a fixed numerical cap."""

    if (
        dual.ndim != 0
        or constraint.ndim != 0
        or not torch.isfinite(dual)
        or not torch.isfinite(constraint)
        or bool(dual < 0)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
        or not math.isfinite(float(maximum))
        or maximum <= 0
    ):
        raise ValueError("invalid CAR dual update")
    return torch.clamp(
        dual + float(learning_rate) * constraint.detach(),
        min=0.0,
        max=float(maximum),
    )


class CriterionAlignedSemanticPolicy(RelativeSemanticSwitchPolicy):
    """The frozen RCSP v1.1 semantic architecture, under the CAR objective."""

    def __init__(self, policy_dim: int) -> None:
        # Deliberately delegate the architecture unchanged.  A separate class
        # makes the revision explicit without changing its state-dict schema.
        super().__init__(policy_dim, embedding_dim=768, rank=4)


class PolicyOnlyCARControl(nn.Module):
    """Small policy-scalar-only attribution control (no semantic embeddings)."""

    def __init__(self, input_dim: int = len(POLICY_FEATURE_NAMES)) -> None:
        super().__init__()
        if input_dim != len(POLICY_FEATURE_NAMES):
            raise ValueError("CAR policy-only dimension drift")
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24), nn.GELU(), nn.Linear(24, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))

    def set_standardization(
        self, mean: Tensor | np.ndarray, scale: Tensor | np.ndarray,
    ) -> None:
        mean = torch.as_tensor(mean, dtype=self.feature_mean.dtype)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if (
            mean.shape != (self.input_dim,)
            or scale.shape != (self.input_dim,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or not torch.all(scale > 0)
        ):
            raise ValueError("invalid CAR policy-only standardization")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def forward(self, features: Tensor) -> Tensor:
        if (
            features.ndim != 2
            or features.shape[1] != self.input_dim
            or not torch.is_floating_point(features)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("invalid CAR policy-only tensor")
        value = self.network(
            (features - self.feature_mean) / self.feature_scale
        ).squeeze(1)
        if not torch.isfinite(value).all():
            raise RuntimeError("CAR policy-only output is non-finite")
        return value


def build_model(representation: str) -> nn.Module:
    """Build one of the pre-registered main/control representations."""

    if representation == "semantic":
        return CriterionAlignedSemanticPolicy(len(POLICY_FEATURE_NAMES))
    if representation == "engineered_28d":
        return EngineeredRCSPControl(CAR_ENGINEERED_FEATURE_DIM)
    if representation == "policy_only":
        return PolicyOnlyCARControl(len(POLICY_FEATURE_NAMES))
    raise ValueError(f"unknown CAR representation: {representation}")


__all__ = [
    "CAR_CHECKPOINT_SCHEMA",
    "CAR_ENGINEERED_FEATURE_DIM",
    "CAR_POLICY_FEATURE_NAMES",
    "CAR_REVISION",
    "CARZeroSelectionError",
    "CriterionAlignedSemanticPolicy",
    "PolicyOnlyCARControl",
    "build_model",
    "catastrophic_rate_constraint",
    "event_domain_weights",
    "projected_dual_update",
    "selected_utility_constraint",
    "straight_through_gate",
    "utility_weighted_preference_loss",
]
