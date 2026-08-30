"""Risk-constrained counterfactual switch policy for MF3ZL-RCSP v1."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


POLICY_FEATURE_NAMES = (
    "step",
    "mf3v_score",
    "native_margin",
    "minimum_advantage",
    "median_advantage",
    "robust_advantage",
    "ensemble_mad",
    "cold_start_floor_ratio",
    "cold_start_relative_mad",
    "candidate_count",
)
CHECKPOINT_SCHEMA = "revealnav-mf3zl-rcsp-checkpoint/1"
ENGINEERED_FEATURE_DIM = 28


def policy_features(decision: dict) -> np.ndarray:
    """Extract the ten frozen proposal-side causal scalars."""

    result = np.asarray([
        float(decision["step"]),
        float(decision["policy_risk_adjusted_score"]),
        float(decision["native_margin"]),
        float(decision["minimum_top2_advantage"]),
        float(decision["median_top2_advantage"]),
        float(decision["robust_top2_advantage"]),
        float(decision["ensemble_mad"]),
        float(decision["cold_start_floor_ratio"]),
        float(decision["cold_start_relative_mad"]),
        float(len(decision["current_local_action_ids"])),
    ], dtype=np.float64)
    if result.shape != (len(POLICY_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("RCSP policy feature drift")
    return result


class RelativeSemanticSwitchPolicy(nn.Module):
    """Rank-4 native-versus-runner semantic preference policy."""

    def __init__(
        self,
        policy_dim: int,
        embedding_dim: int = 768,
        rank: int = 4,
    ) -> None:
        super().__init__()
        if policy_dim != len(POLICY_FEATURE_NAMES):
            raise ValueError("RCSP policy feature dimension drift")
        if embedding_dim != 768 or rank != 4:
            raise ValueError("MF3ZL-RCSP v1 architecture is frozen")
        self.policy_dim = policy_dim
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.context_alpha_logit = nn.Parameter(torch.zeros(()))
        self.context_projection = nn.Linear(embedding_dim, rank, bias=False)
        self.difference_projection = nn.Linear(embedding_dim, rank, bias=False)
        self.policy_head = nn.Linear(policy_dim, 1)
        self.register_buffer("policy_mean", torch.zeros(policy_dim))
        self.register_buffer("policy_scale", torch.ones(policy_dim))
        nn.init.xavier_uniform_(self.context_projection.weight, gain=0.1)
        nn.init.xavier_uniform_(self.difference_projection.weight, gain=0.1)
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def set_policy_standardization(
        self, mean: Tensor | np.ndarray, scale: Tensor | np.ndarray,
    ) -> None:
        mean = torch.as_tensor(mean, dtype=self.policy_mean.dtype)
        scale = torch.as_tensor(scale, dtype=self.policy_scale.dtype)
        if (
            mean.shape != (self.policy_dim,)
            or scale.shape != (self.policy_dim,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or not torch.all(scale > 0)
        ):
            raise ValueError("invalid RCSP policy standardization")
        self.policy_mean.copy_(mean)
        self.policy_scale.copy_(scale)

    def forward(
        self,
        policy_features: Tensor,
        instruction: Tensor,
        history: Tensor,
        native: Tensor,
        runner: Tensor,
    ) -> Tensor:
        batch = policy_features.shape[0] if policy_features.ndim == 2 else -1
        expected_embedding = (batch, self.embedding_dim)
        if policy_features.shape != (batch, self.policy_dim) or batch < 1:
            raise ValueError("RCSP policy tensor shape drift")
        if any(value.shape != expected_embedding for value in (
            instruction, history, native, runner,
        )):
            raise ValueError("RCSP embedding tensor shape drift")
        values = (policy_features, instruction, history, native, runner)
        if any(
            not torch.is_floating_point(value) or not torch.isfinite(value).all()
            for value in values
        ):
            raise ValueError("RCSP inputs must be finite floating-point tensors")
        difference = runner - native
        norms = tuple(torch.linalg.vector_norm(value, dim=1) for value in (
            instruction, history, difference,
        ))
        if any(torch.any(value <= 1e-8) for value in norms):
            raise ValueError("RCSP semantic input has zero norm")
        instruction_unit = instruction / norms[0][:, None]
        history_unit = history / norms[1][:, None]
        difference_unit = difference / norms[2][:, None]
        alpha = torch.sigmoid(self.context_alpha_logit)
        context = alpha * instruction_unit + (1.0 - alpha) * history_unit
        compatibility = torch.sum(
            self.context_projection(context)
            * self.difference_projection(difference_unit),
            dim=1,
        )
        standardized = (
            policy_features - self.policy_mean
        ) / self.policy_scale
        logits = self.policy_head(standardized).squeeze(1) + compatibility
        if not torch.isfinite(logits).all():
            raise RuntimeError("RCSP produced a non-finite switch logit")
        return logits


class EngineeredRCSPControl(nn.Module):
    """Pre-registered 28D representation control with the same policy loss."""

    def __init__(self, input_dim: int = ENGINEERED_FEATURE_DIM) -> None:
        super().__init__()
        if input_dim != ENGINEERED_FEATURE_DIM:
            raise ValueError("RCSP-28D control feature dimension drift")
        self.input_dim = input_dim
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
            raise ValueError("invalid RCSP-28D standardization")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def forward(self, features: Tensor) -> Tensor:
        if (
            features.ndim != 2
            or features.shape[1] != self.input_dim
            or not torch.is_floating_point(features)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("invalid RCSP-28D feature tensor")
        result = self.network(
            (features - self.feature_mean) / self.feature_scale
        ).squeeze(1)
        if not torch.isfinite(result).all():
            raise RuntimeError("RCSP-28D produced non-finite logits")
        return result


def utility_weighted_preference_loss(
    logits: Tensor,
    delta_utility: Tensor,
    sample_weight: Tensor,
) -> Tensor:
    """Magnitude-weighted native/runner preference logistic loss."""

    if (
        logits.ndim != 1
        or delta_utility.shape != logits.shape
        or sample_weight.shape != logits.shape
        or len(logits) == 0
        or any(not torch.is_floating_point(value) for value in (
            logits, delta_utility, sample_weight,
        ))
        or any(not torch.isfinite(value).all() for value in (
            logits, delta_utility, sample_weight,
        ))
        or not torch.all(sample_weight > 0)
    ):
        raise ValueError("invalid RCSP preference-loss input")
    magnitude_weight = sample_weight * torch.abs(delta_utility)
    denominator = torch.sum(magnitude_weight)
    if not bool(denominator > 0):
        raise ValueError("RCSP preference cohort has zero utility magnitude")
    target = (delta_utility > 0).to(logits.dtype)
    row_loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    return torch.sum(magnitude_weight * row_loss) / denominator


def catastrophic_constraint(
    logits: Tensor,
    catastrophic: Tensor,
    sample_weight: Tensor,
    ungated_rate: float,
) -> Tensor:
    """Soft policy constraint with the same sign as selected risk excess."""

    if (
        logits.ndim != 1
        or catastrophic.shape != logits.shape
        or sample_weight.shape != logits.shape
        or len(logits) == 0
        or not math.isfinite(float(ungated_rate))
        or not 0.0 <= float(ungated_rate) <= 1.0
        or not torch.isfinite(logits).all()
        or not torch.isfinite(catastrophic).all()
        or not torch.isfinite(sample_weight).all()
        or not torch.all(sample_weight > 0)
        or not torch.all((catastrophic == 0) | (catastrophic == 1))
    ):
        raise ValueError("invalid RCSP catastrophic constraint input")
    probability = torch.sigmoid(logits)
    excess = catastrophic.to(logits.dtype) - float(ungated_rate)
    return torch.sum(sample_weight * probability * excess) / torch.sum(
        sample_weight
    )


def projected_dual_update(
    dual: Tensor, constraint: Tensor, learning_rate: float,
) -> Tensor:
    if (
        dual.ndim != 0
        or constraint.ndim != 0
        or not torch.isfinite(dual)
        or not torch.isfinite(constraint)
        or dual < 0
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("invalid RCSP dual update")
    return torch.clamp(
        dual + float(learning_rate) * constraint.detach(), min=0.0
    )


class RelativeSemanticSwitchGate:
    """Strict fixed-boundary inference wrapper for an accepted RCSP model."""

    def __init__(self, checkpoint: dict) -> None:
        if not (
            isinstance(checkpoint, dict)
            and checkpoint.get("schema_version") == CHECKPOINT_SCHEMA
            and checkpoint.get("policy_feature_names")
            == list(POLICY_FEATURE_NAMES)
            and checkpoint.get("embedding_dim") == 768
            and checkpoint.get("rank") == 4
            and checkpoint.get("decision_rule") == "switch_logit > 0"
        ):
            raise RuntimeError("RCSP checkpoint contract drift")
        states = checkpoint.get("state_dicts")
        if not isinstance(states, list) or not states:
            raise RuntimeError("RCSP checkpoint has no ensemble")
        self.models = []
        for state in states:
            model = RelativeSemanticSwitchPolicy(len(POLICY_FEATURE_NAMES))
            model.load_state_dict(state, strict=True)
            model.eval()
            self.models.append(model)

    def evaluate(
        self,
        policy: np.ndarray,
        instruction: np.ndarray,
        history: np.ndarray,
        native: np.ndarray,
        runner: np.ndarray,
    ) -> dict:
        arrays = tuple(np.asarray(value, dtype=np.float32) for value in (
            policy, instruction, history, native, runner,
        ))
        if arrays[0].shape != (len(POLICY_FEATURE_NAMES),) or any(
            value.shape != (768,) for value in arrays[1:]
        ) or any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("RCSP inference input drift")
        tensors = [torch.from_numpy(value[None]) for value in arrays]
        with torch.no_grad():
            members = torch.stack([
                model(*tensors)[0] for model in self.models
            ])
        logit = float(torch.median(members).item())
        if not math.isfinite(logit):
            raise RuntimeError("RCSP ensemble logit is non-finite")
        return {
            "switch_logit": logit,
            "authorized": logit > 0.0,
            "decision_rule": "switch_logit > 0",
        }
