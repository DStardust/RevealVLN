"""Shared causal feature and inference code for the MF3ZE safety gate."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "step_over_10", "log1p_step", "is_step_zero", "is_step_lt_two",
    "mf3v_score", "native_margin", "log1p_native_margin",
    "minimum_advantage", "median_advantage", "robust_advantage",
    "ensemble_mad", "floor_ratio", "relative_mad",
    "candidate_count_over_10", "cos_instruction_checkpoint",
    "cos_instruction_native", "cos_instruction_alternative",
    "cos_instruction_delta", "cos_checkpoint_native",
    "cos_checkpoint_alternative", "cos_checkpoint_delta",
    "cos_native_alternative", "instruction_rms", "checkpoint_rms",
    "native_rms", "alternative_rms", "delta_rms", "delta_mean_abs",
)


def hierarchical_proposal_tier(
    score: float, expansion_threshold: float, core_threshold: float,
    upper_threshold: float, *, core_evaluated: bool,
    expansion_evaluated: bool, intervened: bool,
) -> str | None:
    """Select one disjoint proposal tier without letting a veto consume the other."""
    values = (score, expansion_threshold, core_threshold, upper_threshold)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("MF3ZG proposal threshold is non-finite")
    if not expansion_threshold < core_threshold < upper_threshold:
        raise ValueError("MF3ZG proposal thresholds are not ordered")
    if intervened:
        return None
    if not core_evaluated and core_threshold < score <= upper_threshold:
        return "core"
    if (
        not expansion_evaluated
        and expansion_threshold < score <= core_threshold
    ):
        return "expansion"
    return None


def residual_with_uncertainty_source(
    learned_authorized: bool, native_margin: float, uncertainty_threshold: float,
) -> str | None:
    """Give the learned residual priority, then fall back to uncertainty."""
    if not all(math.isfinite(value) for value in (
        native_margin, uncertainty_threshold,
    )):
        raise ValueError("MF3ZH uncertainty margin is non-finite")
    if learned_authorized:
        return "learned_residual"
    if native_margin <= uncertainty_threshold:
        return "uncertainty_floor"
    return None


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-8))


def action_aligned_features(
    decision: dict, instruction: np.ndarray, checkpoint: np.ndarray,
    native: np.ndarray, alternative: np.ndarray,
) -> np.ndarray:
    values = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (instruction, checkpoint, native, alternative)
    )
    if any(value.shape != (768,) or not np.isfinite(value).all() for value in values):
        raise ValueError("MF3ZE online embedding drift")
    instruction, checkpoint, native, alternative = values
    delta = alternative - native
    step = float(decision["step"])
    result = np.asarray([
        step / 10.0, math.log1p(step), float(step == 0), float(step < 2),
        float(decision["policy_risk_adjusted_score"]),
        float(decision["native_margin"]),
        math.log1p(max(0.0, float(decision["native_margin"]))),
        float(decision["minimum_top2_advantage"]),
        float(decision["median_top2_advantage"]),
        float(decision["robust_top2_advantage"]),
        float(decision["ensemble_mad"]),
        float(decision["cold_start_floor_ratio"]),
        float(decision["cold_start_relative_mad"]),
        len(decision["current_local_action_ids"]) / 10.0,
        _cosine(instruction, checkpoint), _cosine(instruction, native),
        _cosine(instruction, alternative), _cosine(instruction, delta),
        _cosine(checkpoint, native), _cosine(checkpoint, alternative),
        _cosine(checkpoint, delta), _cosine(native, alternative),
        float(np.linalg.norm(instruction) / math.sqrt(instruction.size)),
        float(np.linalg.norm(checkpoint) / math.sqrt(checkpoint.size)),
        float(np.linalg.norm(native) / math.sqrt(native.size)),
        float(np.linalg.norm(alternative) / math.sqrt(alternative.size)),
        float(np.linalg.norm(delta) / math.sqrt(delta.size)),
        float(np.mean(np.abs(delta))),
    ], dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("MF3ZE causal feature drift")
    return result


class ActionAlignedReturnGate:
    """Immutable bagged linear return/harm gate used after an MF3V proposal."""

    def __init__(self, model_path: Path, rule: dict) -> None:
        with np.load(model_path, allow_pickle=False) as payload:
            names = tuple(str(value) for value in payload["feature_names"].tolist())
            self.means = payload["means"].copy()
            self.scales = payload["scales"].copy()
            self.return_coefficients = payload["return_coefficients"].copy()
            self.harm_coefficients = payload["harm_coefficients"].copy()
        if names != FEATURE_NAMES:
            raise RuntimeError("MF3ZE model feature schema drift")
        members = len(self.means)
        dimensions = len(FEATURE_NAMES)
        if not (
            self.means.shape == self.scales.shape == (members, dimensions)
            and self.return_coefficients.shape == self.harm_coefficients.shape
            == (members, dimensions + 1)
            and members >= 1
            and all(np.isfinite(value).all() for value in (
                self.means, self.scales, self.return_coefficients,
                self.harm_coefficients,
            ))
            and (self.scales > 0).all()
        ):
            raise RuntimeError("MF3ZE model tensor drift")
        self.return_threshold = float(rule["return_threshold"])
        self.harm_threshold = float(rule["harm_probability_threshold"])

    def evaluate(self, features: np.ndarray) -> dict:
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError("MF3ZE inference feature drift")
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
            "authorized": (
                robust >= self.return_threshold and upper_harm <= self.harm_threshold
            ),
        }
