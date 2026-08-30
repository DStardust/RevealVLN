"""Policy-anchored distributional critic for a frozen runner-up switch.

The module deliberately has no proposal logic.  It receives the existing
causal action-aligned feature vector and estimates the distribution of the
exact one-switch utility difference relative to the frozen native action.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from revealnav_mf3.action_aligned import FEATURE_NAMES


QUANTILES = (0.20, 0.50, 0.80)
CHECKPOINT_SCHEMA = "revealnav-mf3zk-dsr-ensemble/1"
NATIVE_MARGIN_INDEX = FEATURE_NAMES.index("native_margin")


class DistributionalSwitchError(RuntimeError):
    """Raised when a DSR model or checkpoint violates its frozen contract."""


class DistributionalSwitchCritic(nn.Module):
    """Small ordered-quantile critic anchored to frozen-policy uncertainty."""

    def __init__(self, input_dim: int = 28, hidden_dim: int = 24) -> None:
        super().__init__()
        if input_dim != len(FEATURE_NAMES):
            raise ValueError(
                f"DSR input_dim must equal the frozen {len(FEATURE_NAMES)} features"
            )
        if hidden_dim != 24:
            raise ValueError("DSR v1 hidden_dim is frozen at 24")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.median_residual = nn.Linear(self.hidden_dim, 1)
        self.lower_span = nn.Linear(self.hidden_dim, 1)
        self.upper_span = nn.Linear(self.hidden_dim, 1)
        # softplus(inverse_softplus(1)) == 1 at initialization.
        self.anchor_log_scale = nn.Parameter(torch.tensor(
            math.log(math.expm1(1.0)), dtype=torch.float32,
        ))
        self.register_buffer(
            "feature_mean", torch.zeros(self.input_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "feature_scale", torch.ones(self.input_dim, dtype=torch.float32)
        )
        # A small, symmetric 0.05 initial interval matches the utility scale
        # without biasing the median residual.  These constants are frozen in
        # the pre-training protocol rather than chosen from outcomes.
        initial_span_bias = math.log(math.expm1(0.05))
        for head in (self.median_residual, self.lower_span, self.upper_span):
            nn.init.zeros_(head.weight)
        nn.init.zeros_(self.median_residual.bias)
        nn.init.constant_(self.lower_span.bias, initial_span_bias)
        nn.init.constant_(self.upper_span.bias, initial_span_bias)

    def set_standardization(
        self, mean: torch.Tensor | np.ndarray, scale: torch.Tensor | np.ndarray,
    ) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=self.feature_mean.dtype)
        scale_tensor = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if (
            mean_tensor.shape != (self.input_dim,)
            or scale_tensor.shape != (self.input_dim,)
            or not torch.isfinite(mean_tensor).all()
            or not torch.isfinite(scale_tensor).all()
            or not torch.all(scale_tensor > 0)
        ):
            raise ValueError("invalid DSR feature standardization")
        self.feature_mean.copy_(mean_tensor)
        self.feature_scale.copy_(scale_tensor)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"DSR features must have shape (N, {self.input_dim})"
            )
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("DSR features must be finite floating-point values")
        normalized = (features - self.feature_mean) / self.feature_scale
        hidden = F.gelu(self.hidden(normalized))
        native_margin = torch.clamp(features[:, NATIVE_MARGIN_INDEX], min=0.0)
        anchor = -torch.log1p(native_margin)
        median = (
            F.softplus(self.anchor_log_scale) * anchor
            + self.median_residual(hidden).squeeze(-1)
        )
        lower = median - F.softplus(self.lower_span(hidden).squeeze(-1))
        upper = median + F.softplus(self.upper_span(hidden).squeeze(-1))
        if not all(torch.isfinite(value).all() for value in (lower, median, upper)):
            raise DistributionalSwitchError("DSR produced a non-finite quantile")
        return {
            "lower_q20": lower,
            "median_q50": median,
            "upper_q80": upper,
        }


def quantile_switch_loss(
    prediction: Mapping[str, torch.Tensor],
    target_delta_utility: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean pinball loss for the three frozen DSR quantiles."""

    required = ("lower_q20", "median_q50", "upper_q80")
    if tuple(prediction.keys()) != required:
        raise ValueError("DSR prediction schema drift")
    target = target_delta_utility
    weight = sample_weight
    if (
        target.ndim != 1
        or weight.ndim != 1
        or len(target) == 0
        or len(target) != len(weight)
        or not torch.is_floating_point(target)
        or not torch.is_floating_point(weight)
        or not torch.isfinite(target).all()
        or not torch.isfinite(weight).all()
        or not torch.all(weight > 0)
    ):
        raise ValueError("invalid DSR target or sample weight")
    losses = []
    for name, quantile in zip(required, QUANTILES, strict=True):
        value = prediction[name]
        if value.shape != target.shape or not torch.isfinite(value).all():
            raise ValueError("invalid DSR prediction tensor")
        error = target - value
        losses.append(torch.maximum(quantile * error, (quantile - 1.0) * error))
    stacked = torch.stack(losses, dim=1).mean(dim=1)
    return torch.sum(stacked * weight) / torch.sum(weight)


def ensemble_checkpoint(
    models: list[DistributionalSwitchCritic], *, metadata: dict | None = None,
) -> dict:
    """Build the safe, tensor-only payload consumed by the deployment gate."""

    if not models:
        raise ValueError("DSR ensemble must contain at least one model")
    if any(
        model.input_dim != len(FEATURE_NAMES) or model.hidden_dim != 24
        for model in models
    ):
        raise ValueError("DSR ensemble architecture drift")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": 24,
        "quantiles": list(QUANTILES),
        "decision_rule": "lower_q20_utility > 0",
        "state_dicts": [
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for model in models
        ],
        "metadata": {} if metadata is None else dict(metadata),
    }


class DistributionalSwitchGate:
    """Immutable q20>0 authorization gate for a fitted DSR ensemble."""

    def __init__(self, model_path: Path) -> None:
        model_path = Path(model_path)
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise DistributionalSwitchError("DSR checkpoint schema drift")
        if (
            tuple(payload.get("feature_names", ())) != FEATURE_NAMES
            or payload.get("input_dim") != len(FEATURE_NAMES)
            or payload.get("hidden_dim") != 24
            or tuple(float(value) for value in payload.get("quantiles", ()))
            != QUANTILES
            or payload.get("decision_rule") != "lower_q20_utility > 0"
        ):
            raise DistributionalSwitchError("DSR checkpoint contract drift")
        states = payload.get("state_dicts")
        if not isinstance(states, list) or not states:
            raise DistributionalSwitchError("DSR checkpoint has no ensemble")
        self.models = []
        for state in states:
            if not isinstance(state, dict):
                raise DistributionalSwitchError("DSR checkpoint state drift")
            model = DistributionalSwitchCritic()
            model.load_state_dict(state, strict=True)
            model.eval()
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise DistributionalSwitchError("DSR checkpoint contains non-finite state")
            self.models.append(model)

    def evaluate(self, features: np.ndarray) -> dict:
        value = np.asarray(features, dtype=np.float32)
        if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
            raise ValueError("DSR inference feature drift")
        tensor = torch.from_numpy(value[None, :])
        with torch.no_grad():
            members = [model(tensor) for model in self.models]
        lower = float(torch.median(torch.stack([
            member["lower_q20"][0] for member in members
        ])).item())
        median = float(torch.median(torch.stack([
            member["median_q50"][0] for member in members
        ])).item())
        upper = float(torch.median(torch.stack([
            member["upper_q80"][0] for member in members
        ])).item())
        if not all(math.isfinite(item) for item in (lower, median, upper)):
            raise DistributionalSwitchError("DSR ensemble produced non-finite output")
        if lower > median or median > upper:
            raise DistributionalSwitchError("DSR ensemble quantile ordering drift")
        return {
            "lower_q20_utility": lower,
            "median_q50_utility": median,
            "upper_q80_utility": upper,
            "authorized": lower > 0.0,
            "decision_rule": "lower_q20_utility > 0",
        }
