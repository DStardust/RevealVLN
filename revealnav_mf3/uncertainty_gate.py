"""Small, explicit safety gate for the native-margin uncertainty action."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# Unlike the MF3ZE learned-proposal gate, this representation contains only
# quantities available to the uncertainty controller itself.  Keeping the
# schema separate prevents silently filling learned-proposal fields with
# fabricated zeros.
FEATURE_NAMES = (
    "step_over_10",
    "log1p_step",
    "is_step_zero",
    "native_margin",
    "log1p_native_margin",
    "candidate_count_over_10",
    "cos_instruction_checkpoint",
    "cos_instruction_native",
    "cos_instruction_alternative",
    "cos_instruction_delta",
    "cos_checkpoint_native",
    "cos_checkpoint_alternative",
    "cos_native_alternative",
    "native_rms",
    "alternative_rms",
    "delta_rms",
    "delta_mean_abs",
)

# Compact relations shared by the learned UAD proposal and the native-margin
# proposal.  The transfer critic intentionally excludes proposal-source and
# scene identifiers so its online contract is identical for both sources.
TRANSFER_FEATURE_NAMES = (
    "step_over_10",
    "native_margin",
    "cos_instruction_checkpoint",
    "cos_instruction_native",
    "cos_instruction_alternative",
    "cos_instruction_delta",
    "cos_checkpoint_native",
    "cos_native_alternative",
)
_TRANSFER_INDICES = tuple(FEATURE_NAMES.index(name) for name in TRANSFER_FEATURE_NAMES)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-8))


def uncertainty_action_features(
    decision: dict,
    instruction: np.ndarray,
    checkpoint: np.ndarray,
    native: np.ndarray,
    alternative: np.ndarray,
) -> np.ndarray:
    values = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (instruction, checkpoint, native, alternative)
    )
    if any(value.shape != (768,) or not np.isfinite(value).all() for value in values):
        raise ValueError("MF3ZI uncertainty embedding drift")
    instruction, checkpoint, native, alternative = values
    step = float(decision["step"])
    margin = float(decision["native_margin"])
    if not math.isfinite(step) or step < 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError("MF3ZI uncertainty decision drift")
    delta = alternative - native
    result = np.asarray([
        step / 10.0,
        math.log1p(step),
        float(step == 0),
        margin,
        math.log1p(margin),
        len(decision["current_local_action_ids"]) / 10.0,
        _cosine(instruction, checkpoint),
        _cosine(instruction, native),
        _cosine(instruction, alternative),
        _cosine(instruction, delta),
        _cosine(checkpoint, native),
        _cosine(checkpoint, alternative),
        _cosine(native, alternative),
        float(np.linalg.norm(native) / math.sqrt(native.size)),
        float(np.linalg.norm(alternative) / math.sqrt(alternative.size)),
        float(np.linalg.norm(delta) / math.sqrt(delta.size)),
        float(np.mean(np.abs(delta))),
    ], dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("MF3ZI uncertainty feature drift")
    return result


def transfer_action_features(
    decision: dict,
    instruction: np.ndarray,
    checkpoint: np.ndarray,
    native: np.ndarray,
    alternative: np.ndarray,
) -> np.ndarray:
    """Return the fixed proposal-source-invariant counterfactual features."""
    full = uncertainty_action_features(
        decision, instruction, checkpoint, native, alternative
    )
    result = full[np.asarray(_TRANSFER_INDICES)].copy()
    if result.shape != (len(TRANSFER_FEATURE_NAMES),):
        raise RuntimeError("MF3ZJ transfer feature drift")
    return result


class UncertaintyReturnGate:
    """Bootstrap linear return/harm ensemble for one-shot uncertainty actions."""

    def __init__(self, model_path: Path, rule: dict) -> None:
        with np.load(model_path, allow_pickle=False) as payload:
            names = tuple(str(value) for value in payload["feature_names"].tolist())
            self.means = payload["means"].copy()
            self.scales = payload["scales"].copy()
            self.return_coefficients = payload["return_coefficients"].copy()
            self.harm_coefficients = payload["harm_coefficients"].copy()
        dimensions = len(FEATURE_NAMES)
        members = len(self.means)
        if names != FEATURE_NAMES or not (
            self.means.shape == self.scales.shape == (members, dimensions)
            and self.return_coefficients.shape == self.harm_coefficients.shape
            == (members, dimensions + 1)
            and members >= 1
            and np.isfinite(self.means).all()
            and np.isfinite(self.scales).all()
            and np.isfinite(self.return_coefficients).all()
            and np.isfinite(self.harm_coefficients).all()
            and (self.scales > 0).all()
        ):
            raise RuntimeError("MF3ZI uncertainty model tensor drift")
        self.return_threshold = float(rule["return_threshold"])
        self.harm_threshold = float(rule["harm_probability_threshold"])

    def evaluate(self, features: np.ndarray) -> dict:
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError("MF3ZI uncertainty inference feature drift")
        normalized = (features[None, :] - self.means) / self.scales
        design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
        returns = np.sum(design * self.return_coefficients, axis=1)
        logits = np.clip(np.sum(design * self.harm_coefficients, axis=1), -30.0, 30.0)
        harms = 1.0 / (1.0 + np.exp(-logits))
        median = float(np.median(returns))
        robust = median - 0.5 * float(np.median(np.abs(returns - median)))
        upper_harm = float(np.quantile(harms, 0.75))
        return {
            "robust_expected_utility": robust,
            "upper_harm_probability": upper_harm,
            "return_safe": robust >= self.return_threshold,
            "harm_safe": upper_harm <= self.harm_threshold,
            "authorized": robust >= self.return_threshold and upper_harm <= self.harm_threshold,
        }


class CounterfactualTransferGate:
    """Bounded-return ensemble transferred to a one-shot fallback proposal."""

    def __init__(self, model_path: Path, rule: dict) -> None:
        with np.load(model_path, allow_pickle=False) as payload:
            names = tuple(str(value) for value in payload["feature_names"].tolist())
            self.means = payload["means"].copy()
            self.scales = payload["scales"].copy()
            self.return_coefficients = payload["return_coefficients"].copy()
            self.harm_coefficients = payload["harm_coefficients"].copy()
        dimensions = len(TRANSFER_FEATURE_NAMES)
        members = len(self.means)
        if names != TRANSFER_FEATURE_NAMES or not (
            self.means.shape == self.scales.shape == (members, dimensions)
            and self.return_coefficients.shape == self.harm_coefficients.shape
            == (members, dimensions + 1)
            and members >= 1
            and np.isfinite(self.means).all()
            and np.isfinite(self.scales).all()
            and np.isfinite(self.return_coefficients).all()
            and np.isfinite(self.harm_coefficients).all()
            and (self.scales > 0).all()
        ):
            raise RuntimeError("MF3ZJ transfer model tensor drift")
        self.return_threshold = float(rule["return_threshold"])
        self.harm_threshold = float(rule["harm_probability_threshold"])
        if self.return_threshold < 0:
            raise RuntimeError("MF3ZJ transfer rule permits negative return")

    def evaluate(self, features: np.ndarray) -> dict:
        if (
            features.shape != (len(TRANSFER_FEATURE_NAMES),)
            or not np.isfinite(features).all()
        ):
            raise ValueError("MF3ZJ transfer inference feature drift")
        normalized = (features[None, :] - self.means) / self.scales
        design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
        returns = np.sum(design * self.return_coefficients, axis=1)
        logits = np.clip(
            np.sum(design * self.harm_coefficients, axis=1), -30.0, 30.0
        )
        harms = 1.0 / (1.0 + np.exp(-logits))
        median = float(np.median(returns))
        robust = median - 0.5 * float(np.median(np.abs(returns - median)))
        upper_harm = float(np.quantile(harms, 0.75))
        return {
            "robust_expected_utility": robust,
            "upper_harm_probability": upper_harm,
            "return_safe": robust >= self.return_threshold,
            "harm_safe": upper_harm <= self.harm_threshold,
            "authorized": (
                robust >= self.return_threshold
                and upper_harm <= self.harm_threshold
            ),
        }
