"""Pre-collection identifiability audits for MF3ZN-TUAD v1.

All probes are fixed low-capacity ridge models evaluated with one shared
five-fold raw-scene partition.  The module exposes no model or regularization
search.  A failed or incomplete sub-audit denies treatment collection.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Mapping, Sequence

import numpy as np

from .temporal_uad_labels import derive_uad
from .tuad_selection import REQUIRED_DOMAINS, assign_tuad_scene_folds


RIDGE_L2 = 1.0
HUBER_DELTA = 1.0
LABEL_VALIDITY_PILOT_EVENTS = 300
UAD_KAPPA_MINIMUM = 0.65
EVIDENCE_KAPPA_MINIMUM = 0.70
LABEL_VALIDITY_PILOT_SALT = "mf3zn-tuad-v1-label-validity-pilot/1"


def canonical_audit_event_id(
    dataset: object,
    scene_id: object,
    episode_id: object,
    decision_step: int,
) -> str:
    """Commit the outcome-free identity of one existing CAR decision event."""

    if isinstance(decision_step, bool) or not isinstance(
        decision_step, (int, np.integer)
    ) or int(decision_step) < 0:
        raise ValueError("decision_step must be a non-negative integer")
    payload = {
        "dataset": str(dataset),
        "scene_id": str(scene_id),
        "episode_id": str(episode_id),
        "decision_step": int(decision_step),
    }
    if any(not payload[key] for key in ("dataset", "scene_id", "episode_id")):
        raise ValueError("canonical audit identity contains an empty field")
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def _matrix(value: np.ndarray, rows: int | None = None, name: str = "matrix") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or (rows is not None and len(matrix) != rows):
        raise ValueError(f"{name} must be a nonempty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _tensor3(value: np.ndarray, name: str) -> np.ndarray:
    tensor = np.asarray(value, dtype=np.float64)
    if tensor.ndim != 3 or min(tensor.shape) < 1:
        raise ValueError(f"{name} must be a nonempty rank-3 tensor")
    if not np.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _boolean_matrix(
    value: np.ndarray,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.dtype != np.bool_ or matrix.shape != shape:
        raise ValueError(f"{name} must be a Boolean matrix with shape {shape}")
    return matrix


def _vector(
    value: Sequence[object], rows: int | None = None, *, dtype=None, name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=dtype)
    if vector.ndim != 1 or (rows is not None and len(vector) != rows) or len(vector) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    return vector


def _ridge_fit(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(matrix)), matrix))
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_L2
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _ridge_predict(matrix: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(matrix)), matrix)) @ coefficient


def _scene_oof_prediction(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    for fold in range(5):
        fit = folds != fold
        held = folds == fold
        if not fit.any() or not held.any():
            raise ValueError("incomplete five-fold scene OOF partition")
        prediction[held] = _ridge_predict(matrix[held], _ridge_fit(matrix[fit], target[fit]))
    if not np.isfinite(prediction).all():
        raise RuntimeError("ridge OOF prediction is incomplete")
    return prediction


def _scene_oof_prediction_masked(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """OOF ridge prediction on an explicit censor/risk population only."""

    if (
        target.ndim != 1
        or folds.shape != target.shape
        or mask.dtype != np.bool_
        or mask.shape != target.shape
        or len(matrix) != len(target)
        or not mask.any()
    ):
        raise ValueError("invalid masked scene-OOF population")
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    for fold in range(5):
        fit = (folds != fold) & mask
        held = (folds == fold) & mask
        if not held.any():
            continue
        if not fit.any():
            raise ValueError("censored target has no OOF fit population")
        prediction[held] = _ridge_predict(
            matrix[held], _ridge_fit(matrix[fit], target[fit])
        )
    if not np.isfinite(prediction[mask]).all():
        raise RuntimeError("masked ridge OOF prediction is incomplete")
    return prediction


def _huber(error: np.ndarray) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(
        absolute <= HUBER_DELTA,
        0.5 * error * error,
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _binary_nll(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))


def _scene_bootstrap_row_improvement(
    improvement: np.ndarray,
    scenes: np.ndarray,
    mask: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    population = np.unique(scenes[mask])
    if len(population) < 2 or replicates < 1:
        raise ValueError("scene bootstrap needs at least two scenes")
    indices = {scene: np.flatnonzero(mask & (scenes == scene)) for scene in population}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = rng.choice(population, size=len(population), replace=True)
        rows = np.concatenate([indices[scene] for scene in sampled])
        values[replicate] = float(improvement[rows].mean())
    return {
        "observed": float(improvement[mask].mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def oracle_relevance_audit(
    current_features: np.ndarray,
    oracle_features: np.ndarray,
    delta_utility: Sequence[float],
    scenes: Sequence[object],
    datasets: Sequence[object],
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260831,
) -> dict:
    """Audit A: does true UAD/reveal/expiry add held-scene information?"""

    current = _matrix(current_features, name="current_features")
    rows = len(current)
    oracle = _matrix(oracle_features, rows, "oracle_features")
    target = _vector(delta_utility, rows, dtype=np.float64, name="delta_utility")
    scene = _vector(scenes, rows, dtype=str, name="scenes")
    domain = _vector(datasets, rows, dtype=str, name="datasets")
    if not np.isfinite(target).all() or set(domain.tolist()) != set(REQUIRED_DOMAINS):
        raise ValueError("invalid oracle-relevance population")
    folds, mapping = assign_tuad_scene_folds(scene)
    control = _scene_oof_prediction(current, target, folds)
    augmented = _scene_oof_prediction(
        np.concatenate((current, oracle), axis=1), target, folds
    )
    improvement = _huber(control - target) - _huber(augmented - target)
    domains = {}
    failures = []
    for index, domain_value in enumerate(REQUIRED_DOMAINS):
        interval = _scene_bootstrap_row_improvement(
            improvement,
            scene,
            domain == domain_value,
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        passed = interval["observed"] > 0.0 and interval["lower_95"] > 0.0
        domains[domain_value] = {"delta_huber": interval, "pass": passed}
        if not passed:
            failures.append(f"{domain_value}:oracle_delta_huber_not_positive")
    return {
        "audit": "oracle_relevance",
        "probe": "fixed_ridge_l2_1",
        "folds": 5,
        "scene_fold_mapping": mapping,
        "domains": domains,
        "status": "ORACLE_RELEVANCE_PASS" if not failures else "TEMPORAL_ORACLE_RELEVANCE_FAIL",
        "failures": failures,
    }


def _factor_probabilities(
    matrix: np.ndarray,
    targets: tuple[np.ndarray, np.ndarray, np.ndarray],
    folds: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        _sigmoid(_scene_oof_prediction_masked(
            matrix, 2.0 * value.astype(np.float64) - 1.0, folds, mask,
        ))
        for value in targets
    )


def _validate_prefix_supervision(
    target_in_set: np.ndarray,
    candidate_separated: np.ndarray,
    evidence_closed: np.ndarray,
    factor_mask: np.ndarray,
    reveal_event: np.ndarray,
    reveal_at_risk: np.ndarray,
    expiry_event: np.ndarray,
    expiry_at_risk: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    values = tuple(
        _boolean_matrix(value, shape, name)
        for value, name in (
            (target_in_set, "target_in_set"),
            (candidate_separated, "candidate_separated"),
            (evidence_closed, "evidence_closed"),
            (factor_mask, "factor_mask"),
            (reveal_event, "reveal_event"),
            (reveal_at_risk, "reveal_at_risk"),
            (expiry_event, "expiry_event"),
            (expiry_at_risk, "expiry_at_risk"),
        )
    )
    in_set, separated, evidence, factors, reveal, reveal_risk, expiry, expiry_risk = values
    lengths = factors.sum(axis=1)
    expected_factor_mask = (
        np.arange(shape[1], dtype=np.int64)[None, :] < lengths[:, None]
    )
    if np.any(lengths == 0) or not np.array_equal(factors, expected_factor_mask):
        raise ValueError("factor_mask must be a nonempty prefix mask")
    if any(np.any(value & ~factors) for value in (
        in_set, separated, evidence, reveal, reveal_risk, expiry, expiry_risk,
    )):
        raise ValueError("oracle supervision is nonzero outside factor_mask")
    if np.any(separated & ~in_set) or np.any(evidence & ~in_set):
        raise ValueError("separation/evidence cannot hold before target-in-set")
    for event, risk, name in (
        (reveal, reveal_risk, "reveal"),
        (expiry, expiry_risk, "expiry"),
    ):
        if np.any(event & ~risk) or np.any(event.sum(axis=1) > 1):
            raise ValueError(f"{name} event must occur at most once while at risk")
        for row in range(shape[0]):
            risk_indices = np.flatnonzero(risk[row])
            if len(risk_indices) and (
                risk_indices[-1] - risk_indices[0] + 1 != len(risk_indices)
            ):
                raise ValueError(f"{name} at-risk mask must be one contiguous interval")
            event_indices = np.flatnonzero(event[row])
            if len(event_indices) and (
                len(risk_indices) == 0 or event_indices[0] != risk_indices[-1]
            ):
                raise ValueError(f"{name} risk interval must end at its event")
    return values


def _final_uad_from_prefix_factors(
    in_set: np.ndarray,
    separated: np.ndarray,
    evidence: np.ndarray,
    factor_mask: np.ndarray,
    *,
    probability: bool,
) -> np.ndarray:
    result: list[str] = []
    for row in range(len(factor_mask)):
        valid = factor_mask[row]
        states = derive_uad(
            tuple(bool(value) for value in (
                in_set[row, valid] >= 0.5 if probability else in_set[row, valid]
            )),
            tuple(bool(value) for value in (
                separated[row, valid] >= 0.5
                if probability else separated[row, valid]
            )),
            tuple(bool(value) for value in (
                evidence[row, valid] >= 0.5 if probability else evidence[row, valid]
            )),
        )
        result.append(states[-1].value)
    return np.asarray(result, dtype="<U1")


def decision_time_uad_truth(
    target_in_set: np.ndarray,
    candidate_separated: np.ndarray,
    evidence_closed: np.ndarray,
    factor_mask: np.ndarray,
) -> np.ndarray:
    """Derive final U/A/D truth from strict per-prefix Boolean factors."""

    mask = np.asarray(factor_mask)
    if mask.ndim != 2 or min(mask.shape) < 1 or mask.dtype != np.bool_:
        raise ValueError("factor_mask must be a nonempty Boolean matrix")
    shape = mask.shape
    in_set = _boolean_matrix(target_in_set, shape, "target_in_set")
    separated = _boolean_matrix(
        candidate_separated, shape, "candidate_separated"
    )
    evidence = _boolean_matrix(evidence_closed, shape, "evidence_closed")
    lengths = mask.sum(axis=1)
    expected = np.arange(shape[1])[None, :] < lengths[:, None]
    if np.any(lengths == 0) or not np.array_equal(mask, expected):
        raise ValueError("factor_mask must be a nonempty prefix mask")
    if any(np.any(value & ~mask) for value in (in_set, separated, evidence)):
        raise ValueError("factor supervision is nonzero outside factor_mask")
    if np.any(separated & ~in_set) or np.any(evidence & ~in_set):
        raise ValueError("separation/evidence cannot hold before target-in-set")
    return _final_uad_from_prefix_factors(
        in_set, separated, evidence, mask, probability=False,
    )


def _macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for label in ("U", "A", "D"):
        true_positive = int(((truth == label) & (prediction == label)).sum())
        false_positive = int(((truth != label) & (prediction == label)).sum())
        false_negative = int(((truth == label) & (prediction != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2.0 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def _scene_bootstrap_metric(
    scenes: np.ndarray,
    mask: np.ndarray,
    metric: Callable[[np.ndarray], float],
    *,
    replicates: int,
    seed: int,
) -> dict:
    population = np.unique(scenes[mask])
    if len(population) < 2 or replicates < 1:
        raise ValueError("metric bootstrap needs at least two scenes")
    scene_rows = {scene: np.flatnonzero(mask & (scenes == scene)) for scene in population}
    rng = np.random.default_rng(int(seed))
    sampled_values = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = rng.choice(population, size=len(population), replace=True)
        sampled_values[replicate] = metric(
            np.concatenate([scene_rows[scene] for scene in sampled])
        )
    observed_rows = np.flatnonzero(mask)
    return {
        "observed": float(metric(observed_rows)),
        "lower_95": float(np.quantile(sampled_values, 0.025)),
        "upper_95": float(np.quantile(sampled_values, 0.975)),
    }


def causal_observability_audit(
    snapshot_features: np.ndarray,
    temporal_summary: np.ndarray,
    target_in_set: np.ndarray,
    candidate_separated: np.ndarray,
    evidence_closed: np.ndarray,
    factor_mask: np.ndarray,
    reveal_event: np.ndarray,
    reveal_at_risk: np.ndarray,
    expiry_event: np.ndarray,
    expiry_at_risk: np.ndarray,
    scenes: Sequence[object],
    datasets: Sequence[object],
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260841,
) -> dict:
    """Audit B with prefix factors and explicit hazard censor populations.

    The fixed probes are fitted on causal summaries at every valid prefix.
    U/A/D truth and predictions are both derived with the frozen K=3/reset
    rule, then evaluated at the event decision prefix.  Reveal/expiry NLL is
    fitted and scored only on the corresponding explicit at-risk population.
    """

    snapshot = _tensor3(snapshot_features, "snapshot_features")
    rows, prefixes = snapshot.shape[:2]
    temporal = _tensor3(temporal_summary, "temporal_summary")
    if temporal.shape[:2] != (rows, prefixes):
        raise ValueError("snapshot and temporal prefix tensors are not aligned")
    labels = _validate_prefix_supervision(
        target_in_set,
        candidate_separated,
        evidence_closed,
        factor_mask,
        reveal_event,
        reveal_at_risk,
        expiry_event,
        expiry_at_risk,
        (rows, prefixes),
    )
    in_set, separated, evidence, factors, reveal, reveal_risk, expiry, expiry_risk = labels
    if np.any(snapshot[~factors] != 0.0) or np.any(temporal[~factors] != 0.0):
        raise ValueError("causal probe padding must be exactly zero")
    truth_uad = decision_time_uad_truth(
        in_set, separated, evidence, factors,
    )
    scene = _vector(scenes, rows, dtype=str, name="scenes")
    domain = _vector(datasets, rows, dtype=str, name="datasets")
    if set(domain.tolist()) != set(REQUIRED_DOMAINS):
        raise ValueError("observability audit requires RxR and R2R")
    event_folds, mapping = assign_tuad_scene_folds(scene)
    flat_folds = np.repeat(event_folds, prefixes)
    flat_scene = np.repeat(scene, prefixes)
    flat_domain = np.repeat(domain, prefixes)
    flat_snapshot = snapshot.reshape(rows * prefixes, snapshot.shape[2])
    flat_temporal = temporal.reshape(rows * prefixes, temporal.shape[2])
    temporal_probe = np.concatenate((flat_snapshot, flat_temporal), axis=1)
    flat_factors = factors.reshape(-1)
    flat_factor_targets = tuple(value.reshape(-1) for value in (
        in_set, separated, evidence,
    ))
    snapshot_factor_flat = _factor_probabilities(
        flat_snapshot, flat_factor_targets, flat_folds, flat_factors,
    )
    temporal_factor_flat = _factor_probabilities(
        temporal_probe, flat_factor_targets, flat_folds, flat_factors,
    )
    snapshot_factor = tuple(value.reshape(rows, prefixes) for value in snapshot_factor_flat)
    temporal_factor = tuple(value.reshape(rows, prefixes) for value in temporal_factor_flat)
    snapshot_uad = _final_uad_from_prefix_factors(
        *snapshot_factor, factors, probability=True,
    )
    temporal_uad = _final_uad_from_prefix_factors(
        *temporal_factor, factors, probability=True,
    )

    flat_reveal = reveal.reshape(-1)
    flat_reveal_risk = reveal_risk.reshape(-1)
    flat_expiry = expiry.reshape(-1)
    flat_expiry_risk = expiry_risk.reshape(-1)
    snapshot_reveal = _sigmoid(_scene_oof_prediction_masked(
        flat_snapshot,
        2.0 * flat_reveal.astype(np.float64) - 1.0,
        flat_folds,
        flat_reveal_risk,
    ))
    temporal_reveal = _sigmoid(_scene_oof_prediction_masked(
        temporal_probe,
        2.0 * flat_reveal.astype(np.float64) - 1.0,
        flat_folds,
        flat_reveal_risk,
    ))
    snapshot_expiry = _sigmoid(_scene_oof_prediction_masked(
        flat_snapshot,
        2.0 * flat_expiry.astype(np.float64) - 1.0,
        flat_folds,
        flat_expiry_risk,
    ))
    temporal_expiry = _sigmoid(_scene_oof_prediction_masked(
        temporal_probe,
        2.0 * flat_expiry.astype(np.float64) - 1.0,
        flat_folds,
        flat_expiry_risk,
    ))
    reveal_improvement = np.zeros(rows * prefixes, dtype=np.float64)
    reveal_improvement[flat_reveal_risk] = (
        _binary_nll(snapshot_reveal[flat_reveal_risk], flat_reveal[flat_reveal_risk])
        - _binary_nll(temporal_reveal[flat_reveal_risk], flat_reveal[flat_reveal_risk])
    )
    expiry_improvement = np.zeros(rows * prefixes, dtype=np.float64)
    expiry_improvement[flat_expiry_risk] = (
        _binary_nll(snapshot_expiry[flat_expiry_risk], flat_expiry[flat_expiry_risk])
        - _binary_nll(temporal_expiry[flat_expiry_risk], flat_expiry[flat_expiry_risk])
    )

    domains = {}
    failures = []
    for index, domain_value in enumerate(REQUIRED_DOMAINS):
        mask = domain == domain_value
        uad = _scene_bootstrap_metric(
            scene,
            mask,
            lambda selected: _macro_f1(truth_uad[selected], temporal_uad[selected])
            - _macro_f1(truth_uad[selected], snapshot_uad[selected]),
            replicates=bootstrap_replicates,
            seed=seed + index * 3,
        )
        reveal = _scene_bootstrap_row_improvement(
            reveal_improvement,
            flat_scene,
            (flat_domain == domain_value) & flat_reveal_risk,
            replicates=bootstrap_replicates, seed=seed + index * 3 + 1,
        )
        expiry = _scene_bootstrap_row_improvement(
            expiry_improvement,
            flat_scene,
            (flat_domain == domain_value) & flat_expiry_risk,
            replicates=bootstrap_replicates, seed=seed + index * 3 + 2,
        )
        passed = all(
            value["observed"] > 0.0 and value["lower_95"] > 0.0
            for value in (uad, reveal, expiry)
        )
        domains[domain_value] = {
            "delta_uad_macro_f1": uad,
            "delta_reveal_nll": reveal,
            "delta_expiry_nll": expiry,
            "decision_events": int(mask.sum()),
            "reveal_at_risk_prefixes": int(
                ((flat_domain == domain_value) & flat_reveal_risk).sum()
            ),
            "expiry_at_risk_prefixes": int(
                ((flat_domain == domain_value) & flat_expiry_risk).sum()
            ),
            "pass": passed,
        }
        if not passed:
            failures.append(f"{domain_value}:causal_history_not_observable")
    return {
        "audit": "causal_observability",
        "snapshot_probe": "fixed_ridge_l2_1_current_only",
        "temporal_probe": "fixed_ridge_l2_1_current_plus_sealed_summary",
        "folds": 5,
        "scene_fold_mapping": mapping,
        "domains": domains,
        "status": (
            "CAUSAL_OBSERVABILITY_PASS"
            if not failures else "TEMPORAL_CAUSAL_OBSERVABILITY_FAIL"
        ),
        "failures": failures,
    }


def cohen_kappa(left: Sequence[object], right: Sequence[object]) -> float:
    left = _vector(left, name="left_rater")
    right = _vector(right, len(left), name="right_rater")
    labels = sorted(set(left.tolist()) | set(right.tolist()), key=str)
    observed = float((left == right).mean())
    expected = sum(
        float((left == label).mean()) * float((right == label).mean())
        for label in labels
    )
    return 1.0 if expected == 1.0 and observed == 1.0 else (
        (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    )


def deterministic_review_pilot_indices(
    event_ids: Sequence[object],
    scenes: Sequence[object],
    *,
    pilot_events: int = LABEL_VALIDITY_PILOT_EVENTS,
    required_scenes: int = 39,
) -> np.ndarray:
    """Select a fixed capacity-constrained scene-balanced review pilot.

    Rows are allocated round-robin across all raw scenes, skipping only a scene
    whose available event population is exhausted.  Thus rare scenes are fully
    represented and every unsaturated scene differs in allocation by at most
    one.  Scene and within-scene ordering are SHA-256 keyed by a source-sealed
    salt, so review outcomes cannot influence which rows enter Audit C.
    """

    identity = _vector(event_ids, dtype=str, name="event_ids")
    scene = _vector(scenes, len(identity), dtype=str, name="scenes")
    if len(set(identity.tolist())) != len(identity):
        raise ValueError("review-pilot event identities must be unique")
    population = sorted(set(scene.tolist()))
    if len(population) != int(required_scenes):
        raise ValueError(
            f"review pilot requires exactly {required_scenes} raw scenes"
        )
    if (
        isinstance(pilot_events, bool)
        or not isinstance(pilot_events, (int, np.integer))
        or int(pilot_events) < len(population)
        or int(pilot_events) > len(identity)
    ):
        raise ValueError("invalid review-pilot event count")
    pilot_events = int(pilot_events)

    def digest(*parts: str) -> str:
        return hashlib.sha256(
            "\0".join((LABEL_VALIDITY_PILOT_SALT, *parts)).encode("utf-8")
        ).hexdigest()

    ordered_scenes = sorted(population, key=lambda value: (
        digest("scene-order", value), value,
    ))
    candidates_by_scene: dict[str, list[int]] = {}
    for scene_value in ordered_scenes:
        candidates = np.flatnonzero(scene == scene_value).tolist()
        candidates.sort(key=lambda index: (
            digest("event", scene_value, identity[index]), identity[index],
        ))
        candidates_by_scene[scene_value] = candidates
    selected: list[int] = []
    offsets = {scene_value: 0 for scene_value in ordered_scenes}
    while len(selected) < pilot_events:
        progressed = False
        for scene_value in ordered_scenes:
            offset = offsets[scene_value]
            candidates = candidates_by_scene[scene_value]
            if offset >= len(candidates):
                continue
            selected.append(candidates[offset])
            offsets[scene_value] += 1
            progressed = True
            if len(selected) == pilot_events:
                break
        if not progressed:
            raise ValueError("review population cannot fill the fixed pilot")
    result = np.asarray(selected, dtype=np.int64)
    if len(result) != pilot_events or len(set(result.tolist())) != pilot_events:
        raise RuntimeError("deterministic review-pilot allocation drift")
    counts = {value: int((scene[result] == value).sum()) for value in population}
    maximum = max(counts.values())
    for value in population:
        capacity = int((scene == value).sum())
        if counts[value] < capacity and maximum - counts[value] > 1:
            raise RuntimeError(
                "deterministic review pilot is not capacity-balanced"
            )
    return result


def label_validity_audit(
    uad_rater_a: Sequence[object],
    uad_rater_b: Sequence[object],
    evidence_rater_a: Sequence[object],
    evidence_rater_b: Sequence[object],
    scenes: Sequence[object],
    *,
    minimum_events: int = LABEL_VALIDITY_PILOT_EVENTS,
    require_scene_balance: bool = True,
    expected_scene_count: int | None = None,
    scene_capacities: Mapping[str, int] | None = None,
) -> dict:
    """Audit C: fixed inter-rater validity gate on a scene-balanced pilot."""

    uad_a = _vector(uad_rater_a, name="uad_rater_a")
    rows = len(uad_a)
    uad_b = _vector(uad_rater_b, rows, name="uad_rater_b")
    evidence_a = _vector(evidence_rater_a, rows, name="evidence_rater_a")
    evidence_b = _vector(evidence_rater_b, rows, name="evidence_rater_b")
    scene = _vector(scenes, rows, dtype=str, name="scenes")
    if not set(uad_a.tolist() + uad_b.tolist()) <= {"U", "A", "D"}:
        raise ValueError("UAD reviews contain an unknown state")
    if any(
        not isinstance(value, (bool, np.bool_))
        for value in (*evidence_a.tolist(), *evidence_b.tolist())
    ):
        raise ValueError("evidence-closure reviews must be boolean")
    if rows < int(minimum_events):
        raise ValueError(f"label-validity pilot has {rows}, needs {minimum_events}")
    selected_counts = {
        str(value): int((scene == value).sum()) for value in np.unique(scene)
    }
    counts = np.asarray(list(selected_counts.values()), dtype=np.int64)
    if scene_capacities is None:
        balanced = bool(
            len(counts) > 0 and int(counts.max() - counts.min()) <= 1
        )
        balance_rule = "equal_allocation"
    else:
        capacities = {str(key): int(value) for key, value in scene_capacities.items()}
        if (
            not capacities
            or any(value < 1 for value in capacities.values())
            or set(selected_counts) != set(capacities)
            or any(selected_counts[key] > capacities[key] for key in capacities)
        ):
            raise ValueError("invalid scene capacities for label-validity pilot")
        maximum = max(selected_counts.values())
        balanced = all(
            selected_counts[key] == capacities[key]
            or maximum - selected_counts[key] <= 1
            for key in capacities
        )
        balance_rule = "capacity_constrained_round_robin"
    if require_scene_balance and not balanced:
        raise ValueError("label-validity pilot is not scene balanced")
    if expected_scene_count is not None and len(counts) != int(expected_scene_count):
        raise ValueError(
            f"label-validity pilot must cover {expected_scene_count} raw scenes"
        )
    uad = float(cohen_kappa(uad_a, uad_b))
    evidence = float(cohen_kappa(evidence_a, evidence_b))
    passed = uad >= UAD_KAPPA_MINIMUM and evidence >= EVIDENCE_KAPPA_MINIMUM
    return {
        "audit": "label_validity",
        "events": rows,
        "scenes": int(len(counts)),
        "scene_balanced": balanced,
        "scene_balance_rule": balance_rule,
        "uad_kappa": uad,
        "evidence_closure_kappa": evidence,
        "thresholds": {
            "uad_kappa": UAD_KAPPA_MINIMUM,
            "evidence_closure_kappa": EVIDENCE_KAPPA_MINIMUM,
        },
        "status": "LABEL_VALIDITY_PASS" if passed else "TEMPORAL_LABEL_VALIDITY_FAIL",
    }


def _subaudit_passes(value: dict, audit: str, status: str) -> bool:
    if not isinstance(value, dict) or value.get("audit") != audit:
        return False
    if value.get("status") != status:
        return False
    if value.get("failures", []) != []:
        return False
    if audit in {"oracle_relevance", "causal_observability"}:
        domains = value.get("domains")
        return (
            isinstance(domains, dict)
            and set(domains) == set(REQUIRED_DOMAINS)
            and all(
                isinstance(domains[domain], dict)
                and domains[domain].get("pass") is True
                for domain in REQUIRED_DOMAINS
            )
        )
    return (
        value.get("scene_balanced") is True
        and isinstance(value.get("uad_kappa"), (int, float))
        and isinstance(value.get("evidence_closure_kappa"), (int, float))
        and float(value["uad_kappa"]) >= UAD_KAPPA_MINIMUM
        and float(value["evidence_closure_kappa"])
        >= EVIDENCE_KAPPA_MINIMUM
    )


def identifiability_gate(
    oracle_relevance: dict,
    causal_observability: dict,
    label_validity: dict,
) -> dict:
    """Authorize collection only if all three pre-collection audits pass."""

    expected = (
        (oracle_relevance, "oracle_relevance", "ORACLE_RELEVANCE_PASS"),
        (
            causal_observability,
            "causal_observability",
            "CAUSAL_OBSERVABILITY_PASS",
        ),
        (label_validity, "label_validity", "LABEL_VALIDITY_PASS"),
    )
    failures = [
        value.get("status", "MISSING")
        if isinstance(value, dict) else "MISSING"
        for value, audit, required in expected
        if not _subaudit_passes(value, audit, required)
    ]
    return {
        "schema_version": "revealnav-mf3zn-identifiability-result/1",
        "method_id": "mf3zn_tuad_v1",
        "status": (
            "MF3ZN_IDENTIFIABILITY_PASS"
            if not failures else "MF3ZN_IDENTIFIABILITY_FAIL"
        ),
        "collection_authorized": not failures,
        "failures": failures,
        "audits": {
            "oracle_relevance": oracle_relevance,
            "causal_observability": causal_observability,
            "label_validity": label_validity,
        },
        "public_authorization": False,
    }


__all__ = [
    "EVIDENCE_KAPPA_MINIMUM",
    "LABEL_VALIDITY_PILOT_EVENTS",
    "RIDGE_L2",
    "UAD_KAPPA_MINIMUM",
    "causal_observability_audit",
    "canonical_audit_event_id",
    "cohen_kappa",
    "decision_time_uad_truth",
    "deterministic_review_pilot_indices",
    "identifiability_gate",
    "label_validity_audit",
    "oracle_relevance_audit",
]
