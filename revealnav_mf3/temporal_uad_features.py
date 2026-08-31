"""Fixed, outcome-free temporal summaries for MF3ZN identifiability audits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .temporal_uad_schema import TemporalSequence, causal_array_bytes


# This order is a sealed interface between collection and the fixed probes.
# Candidate IDs themselves are stored in frozen policy-rank order by the
# causal schema, so no target or oracle rank is needed here.
POLICY_FEATURE_NAMES = (
    "native_score",
    "native_margin",
    "mf3v_score",
    "uncertainty_mad",
    "instruction_history_alignment",
)
NATIVE_SCORE_INDEX = POLICY_FEATURE_NAMES.index("native_score")
NATIVE_MARGIN_INDEX = POLICY_FEATURE_NAMES.index("native_margin")
MF3V_SCORE_INDEX = POLICY_FEATURE_NAMES.index("mf3v_score")
UNCERTAINTY_MAD_INDEX = POLICY_FEATURE_NAMES.index("uncertainty_mad")
ALIGNMENT_INDEX = POLICY_FEATURE_NAMES.index(
    "instruction_history_alignment"
)

TEMPORAL_SUMMARY_NAMES = (
    "score_slope",
    "margin_slope",
    "candidate_birth_count",
    "candidate_expiry_count",
    "native_persistence",
    "runner_persistence",
    "rank_switch_count",
    "checkpoint_embedding_drift",
    "instruction_history_alignment_drift",
    "candidate_set_jaccard",
)

SNAPSHOT_SUMMARY_NAMES = (
    "current_mf3v_score",
    "current_native_margin",
    "current_uncertainty_mad",
    "current_instruction_history_alignment",
    "current_candidate_count",
    "current_native_rank",
    "current_checkpoint_rms",
)

STRUCTURAL_FEATURE_NAMES = (
    "candidate_count",
    "candidate_birth_count",
    "candidate_expiry_count",
    "native_persistence_indicator",
    "runner_persistence_indicator",
    "rank_change_indicator",
    "candidate_set_jaccard",
    "step_delta",
)


def _readonly_vector(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("temporal summary must be a finite vector")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    immutable.setflags(write=False)
    return immutable


def _readonly_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError("causal sequence features must be a finite matrix")
    immutable = np.frombuffer(
        array.tobytes(order="C"), dtype=array.dtype,
    ).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _validate_policy_contract(sequence: TemporalSequence) -> None:
    if not isinstance(sequence, TemporalSequence):
        raise TypeError(
            "temporal feature builders accept only TemporalSequence inputs"
        )
    expected = (len(POLICY_FEATURE_NAMES),)
    if any(step.policy_features.shape != expected for step in sequence.steps):
        raise ValueError(
            "policy_features do not match the fixed MF3ZN feature contract"
        )
    for step in sequence.steps:
        margin = float(step.policy_features[NATIVE_MARGIN_INDEX])
        mad = float(step.policy_features[UNCERTAINTY_MAD_INDEX])
        alignment = float(step.policy_features[ALIGNMENT_INDEX])
        if margin < 0.0 or mad < 0.0:
            raise ValueError("native margin and uncertainty MAD must be non-negative")
        if not -1.0 <= alignment <= 1.0:
            raise ValueError(
                "instruction/history alignment must be a cosine in [-1, 1]"
            )


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered, y - y.mean()) / denominator)


def _runner_id(native: str, ranked: tuple[str, ...]) -> str | None:
    return next((value for value in ranked if value != native), None)


def _persistence(values: tuple[str | None, ...]) -> float:
    if len(values) == 1:
        return float(values[0] is not None)
    return float(sum(
        left is not None and right is not None and left == right
        for left, right in zip(values, values[1:])
    ) / (len(values) - 1))


def _candidate_jaccard(
    left: tuple[str, ...], right: tuple[str, ...],
) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _rank_changed(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Report a relative-order change among candidates visible in both."""

    shared = set(left) & set(right)
    if len(shared) < 2:
        return False
    return (
        tuple(value for value in left if value in shared)
        != tuple(value for value in right if value in shared)
    )


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return 1.0 - min(1.0, max(-1.0, similarity))


def causal_temporal_summary(sequence: TemporalSequence) -> np.ndarray:
    """Build the single pre-registered temporal probe vector.

    The computation uses only fields in ``CausalTemporalStep``.  It has no
    label, outcome, action-value, scene-ID, or treatment-result argument, so
    those quantities cannot affect the returned tensor.
    """

    _validate_policy_contract(sequence)
    steps = sequence.steps
    times = np.asarray([step.step for step in steps], dtype=np.float64)
    scores = np.asarray([
        step.policy_features[MF3V_SCORE_INDEX] for step in steps
    ], dtype=np.float64)
    margins = np.asarray([
        step.policy_features[NATIVE_MARGIN_INDEX] for step in steps
    ], dtype=np.float64)
    alignments = np.asarray([
        step.policy_features[ALIGNMENT_INDEX] for step in steps
    ], dtype=np.float64)

    births = 0
    expiries = 0
    previous: set[str] = set()
    for step in steps:
        current = set(step.candidate_action_ids)
        births += len(current - previous)
        expiries += len(previous - current)
        previous = current

    natives = tuple(step.native_action_id for step in steps)
    runners = tuple(
        _runner_id(step.native_action_id, step.candidate_action_ids)
        for step in steps
    )
    rank_switches = sum(
        _rank_changed(left.candidate_action_ids, right.candidate_action_ids)
        for left, right in zip(steps, steps[1:])
    )
    embedding_drifts = [
        _cosine_distance(left.checkpoint_embedding, right.checkpoint_embedding)
        for left, right in zip(steps, steps[1:])
    ]
    alignment_drifts = np.abs(np.diff(alignments))
    jaccards = [
        _candidate_jaccard(left.candidate_action_ids, right.candidate_action_ids)
        for left, right in zip(steps, steps[1:])
    ]

    result = _readonly_vector([
        _slope(times, scores),
        _slope(times, margins),
        float(births),
        float(expiries),
        _persistence(natives),
        _persistence(runners),
        float(rank_switches),
        float(np.mean(embedding_drifts)) if embedding_drifts else 0.0,
        float(np.mean(alignment_drifts)) if len(alignment_drifts) else 0.0,
        float(np.mean(jaccards)) if jaccards else 1.0,
    ])
    if result.shape != (len(TEMPORAL_SUMMARY_NAMES),):
        raise RuntimeError("temporal summary width drift")
    return result


def causal_snapshot_summary(sequence: TemporalSequence) -> np.ndarray:
    """Fixed current-only control vector for the observability audit."""

    _validate_policy_contract(sequence)
    current = sequence.steps[-1]
    native_rank = current.candidate_action_ids.index(current.native_action_id)
    checkpoint_rms = float(
        np.linalg.norm(current.checkpoint_embedding)
        / np.sqrt(current.checkpoint_embedding.size)
    )
    result = _readonly_vector([
        float(current.policy_features[MF3V_SCORE_INDEX]),
        float(current.policy_features[NATIVE_MARGIN_INDEX]),
        float(current.policy_features[UNCERTAINTY_MAD_INDEX]),
        float(current.policy_features[ALIGNMENT_INDEX]),
        float(len(current.candidate_action_ids)),
        float(native_rank),
        checkpoint_rms,
    ])
    if result.shape != (len(SNAPSHOT_SUMMARY_NAMES),):
        raise RuntimeError("snapshot summary width drift")
    return result


def causal_sequence_feature_width(sequence: TemporalSequence) -> int:
    """Return ``P + C + I + 2*A + S`` for the frozen sequence contract."""

    _validate_policy_contract(sequence)
    policy_width = len(POLICY_FEATURE_NAMES)
    checkpoint_width = sequence.steps[0].checkpoint_embedding.size
    instruction_width = sequence.steps[0].instruction_embedding.size
    action_width = sequence.steps[0].action_embeddings.shape[1]
    return (
        policy_width
        + checkpoint_width
        + instruction_width
        + 2 * action_width
        + len(STRUCTURAL_FEATURE_NAMES)
    )


def causal_sequence_features(sequence: TemporalSequence) -> np.ndarray:
    """Build the fixed per-prefix matrix consumed by the causal GRU.

    Each row concatenates, in order:

    ``policy | checkpoint | instruction | native action |
    mean executable non-native | structural dynamics``.

    Structural dynamics compare the current prefix only with its immediately
    preceding prefix.  The first row uses an empty prior candidate set, zero
    persistence/rank-change/step-delta, and Jaccard one.  Candidate identity
    order is the sealed frozen-policy rank; embeddings remain ID-aligned.
    Consequently no value at row ``j`` depends on any row after ``j``.
    """

    _validate_policy_contract(sequence)
    rows: list[np.ndarray] = []
    previous_candidates: tuple[str, ...] | None = None
    previous_native: str | None = None
    previous_runner: str | None = None
    previous_step: int | None = None
    action_width = sequence.steps[0].action_embeddings.shape[1]

    for step in sequence.steps:
        ranked = step.candidate_action_ids
        native_index = ranked.index(step.native_action_id)
        native_embedding = step.action_embeddings[native_index]
        non_native_mask = np.ones(len(ranked), dtype=np.bool_)
        non_native_mask[native_index] = False
        non_native_mean = (
            step.action_embeddings[non_native_mask].mean(axis=0)
            if bool(non_native_mask.any())
            else np.zeros(action_width, dtype=step.action_embeddings.dtype)
        )
        current_set = set(ranked)
        if previous_candidates is None:
            births = len(current_set)
            expiries = 0
            native_persistent = 0.0
            runner_persistent = 0.0
            rank_changed = 0.0
            jaccard = 1.0
            step_delta = 0.0
        else:
            previous_set = set(previous_candidates)
            births = len(current_set - previous_set)
            expiries = len(previous_set - current_set)
            native_persistent = float(step.native_action_id == previous_native)
            runner = _runner_id(step.native_action_id, ranked)
            runner_persistent = float(
                runner is not None
                and previous_runner is not None
                and runner == previous_runner
            )
            rank_changed = float(_rank_changed(previous_candidates, ranked))
            jaccard = _candidate_jaccard(previous_candidates, ranked)
            step_delta = float(step.step - previous_step)  # type: ignore[operator]

        runner = _runner_id(step.native_action_id, ranked)
        structural = np.asarray([
            float(len(ranked)),
            float(births),
            float(expiries),
            native_persistent,
            runner_persistent,
            rank_changed,
            jaccard,
            step_delta,
        ], dtype=np.float64)
        rows.append(np.concatenate((
            step.policy_features.astype(np.float64, copy=False),
            step.checkpoint_embedding.astype(np.float64, copy=False),
            step.instruction_embedding.astype(np.float64, copy=False),
            native_embedding.astype(np.float64, copy=False),
            non_native_mean.astype(np.float64, copy=False),
            structural,
        )))
        previous_candidates = ranked
        previous_native = step.native_action_id
        previous_runner = runner
        previous_step = step.step

    result = _readonly_matrix(np.stack(rows))
    expected = causal_sequence_feature_width(sequence)
    if result.shape != (len(sequence.steps), expected):
        raise RuntimeError("causal sequence feature width drift")
    return result


def causal_current_only_features(sequence: TemporalSequence) -> np.ndarray:
    """Return the complete decision row with transition history neutralized."""

    current = np.asarray(causal_sequence_features(sequence)[-1], dtype=np.float64).copy()
    width = len(STRUCTURAL_FEATURE_NAMES)
    candidate_count = current[-width]
    current[-width:] = 0.0
    current[-width] = candidate_count
    current[-width + 1] = candidate_count
    current[-2] = 1.0
    return _readonly_vector(current)


def causal_temporal_summary_from_mapping(
    value: Mapping[str, Any],
) -> np.ndarray:
    """Parse through the strict causal schema before building a tensor."""

    return causal_temporal_summary(TemporalSequence.from_mapping(value))


def temporal_summary_bytes(sequence: TemporalSequence) -> bytes:
    """Canonical byte representation used by causality regression tests."""

    return causal_array_bytes(causal_temporal_summary(sequence))
