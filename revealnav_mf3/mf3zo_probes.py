"""Fixed, low-capacity MF3ZO probes with whole-scene OOF evaluation.

There is intentionally no hyperparameter-selection API in this module.
Every probe uses ridge L2=1, a fixed five-fold raw-scene partition, fold-fit
normalization, and a fixed zero boundary for the diagnostic action-value rule.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Mapping, Sequence

import numpy as np

from .mf3zo_temporal_schema import (
    CausalTemporalRecord,
    TemporalOracleLabel,
    UADState,
    derive_uad,
)


RIDGE_L2 = 1.0
HUBER_DELTA = 1.0
OUTER_FOLDS = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260831
CATASTROPHIC_THRESHOLD = -0.10
REQUIRED_DOMAINS = ("R2R", "RxR")
SCENE_FOLD_SALT = "mf3zo-temporal-oracle-gap-v1-scene-folds/1"


def _matrix(value: np.ndarray, *, rows: int | None = None, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 1 or (
        rows is not None and len(result) != rows
    ) or not np.isfinite(result).all():
        raise ValueError(f"invalid finite matrix: {name}")
    return result


def _vector(
    value: Sequence[object] | np.ndarray,
    *,
    rows: int | None = None,
    dtype=None,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1 or len(result) == 0 or (
        rows is not None and len(result) != rows
    ):
        raise ValueError(f"invalid vector: {name}")
    return result


def assign_scene_folds(scenes: Sequence[object]) -> tuple[np.ndarray, dict[str, int]]:
    scene = _vector(scenes, dtype=str, name="scenes")
    population = sorted(set(scene.tolist()))
    if len(population) < OUTER_FOLDS:
        raise ValueError("MF3ZO requires at least five raw MP3D scenes")
    ordered = sorted(population, key=lambda value: (
        hashlib.sha256(
            f"{SCENE_FOLD_SALT}\0{value}".encode("utf-8")
        ).hexdigest(),
        value,
    ))
    mapping = {value: index % OUTER_FOLDS for index, value in enumerate(ordered)}
    folds = np.asarray([mapping[value] for value in scene], dtype=np.int64)
    if set(folds.tolist()) != set(range(OUTER_FOLDS)):
        raise ValueError("MF3ZO scene fold partition is incomplete")
    return folds, mapping


def fit_standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = _matrix(matrix, name="fit_matrix")
    mean = value.mean(axis=0)
    scale = value.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def _ridge_fit(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(matrix)), matrix))
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_L2
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def ridge_scene_oof(
    matrix: np.ndarray,
    target: Sequence[float] | np.ndarray,
    folds: Sequence[int] | np.ndarray,
    *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Fixed ridge OOF with a scaler fitted only on each fold's fit rows."""

    value = _matrix(matrix, name="matrix")
    rows = len(value)
    outcome = _vector(target, rows=rows, dtype=np.float64, name="target")
    fold = _vector(folds, rows=rows, dtype=np.int64, name="folds")
    if not np.isfinite(outcome).all() or set(fold.tolist()) != set(range(OUTER_FOLDS)):
        raise ValueError("invalid OOF target/folds")
    if eligible is None:
        use = np.ones(rows, dtype=np.bool_)
    else:
        use = np.asarray(eligible)
        if use.dtype != np.bool_ or use.shape != (rows,):
            raise ValueError("eligible mask must be Boolean and row-aligned")
    prediction = np.full(rows, np.nan, dtype=np.float64)
    for fold_value in range(OUTER_FOLDS):
        fit = (fold != fold_value) & use
        held = (fold == fold_value) & use
        if held.any() and not fit.any():
            raise ValueError("OOF fold has no eligible fit population")
        if not held.any():
            continue
        mean, scale = fit_standardizer(value[fit])
        coefficient = _ridge_fit((value[fit] - mean) / scale, outcome[fit])
        prediction[held] = np.column_stack((
            np.ones(int(held.sum())), (value[held] - mean) / scale,
        )) @ coefficient
    if not np.isfinite(prediction[use]).all():
        raise RuntimeError("MF3ZO OOF prediction is incomplete")
    return prediction


def huber(error: np.ndarray) -> np.ndarray:
    value = np.asarray(error, dtype=np.float64)
    absolute = np.abs(value)
    return np.where(
        absolute <= HUBER_DELTA,
        0.5 * value * value,
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _binary_nll(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    value = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return -(target * np.log(value) + (1.0 - target) * np.log(1.0 - value))


def scene_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    scenes: Sequence[object],
    mask: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    value = _vector(values, dtype=np.float64, name="values")
    scene = _vector(scenes, rows=len(value), dtype=str, name="scenes")
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != value.shape:
        raise ValueError("bootstrap mask must be Boolean and row-aligned")
    population = np.unique(scene[selected])
    if len(population) < 2 or replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("MF3ZO bootstrap requires 10000 draws and >=2 scenes")
    rows = {item: np.flatnonzero(selected & (scene == item)) for item in population}
    rng = np.random.default_rng(int(seed))
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.choice(population, size=len(population), replace=True)
        indices = np.concatenate([rows[item] for item in draw])
        samples[index] = float(value[indices].mean())
    return {
        "observed": float(value[selected].mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_value) * np.linalg.norm(right_value))
    if denominator <= 0.0:
        raise ValueError("zero-norm semantic embedding")
    return float(np.clip(np.dot(left_value, right_value) / denominator, -1.0, 1.0))


def current_snapshot_features(record: CausalTemporalRecord) -> np.ndarray:
    current = record.steps[-1]
    if (
        current.checkpoint_embedding is None
        or current.action_embeddings is None
        or len(current.embedded_action_ids) != 2
        or not current.policy_feature_mask.all()
    ):
        raise ValueError("current snapshot lacks fixed MF3ZO features")
    instruction = current.instruction_embedding
    checkpoint = current.checkpoint_embedding
    native, alternative = current.action_embeddings
    semantic = np.asarray([
        _cosine(instruction, checkpoint),
        _cosine(instruction, native),
        _cosine(instruction, alternative),
        _cosine(checkpoint, native),
        _cosine(checkpoint, alternative),
        _cosine(native, alternative),
    ], dtype=np.float64)
    return np.concatenate((current.policy_features.astype(np.float64), semantic))


def oracle_feature_vector(
    label: TemporalOracleLabel, decision_step: int,
) -> np.ndarray:
    if not label.complete:
        raise ValueError("oracle feature vector requires complete verified labels")
    states = derive_uad(
        label.target_in_set,
        label.candidate_separated,
        label.evidence_closed,
    )
    state = states[-1]
    one_hot = np.asarray([
        float(state is UADState.UNOBSERVED),
        float(state is UADState.AMBIGUOUS),
        float(state is UADState.DECISIVE),
    ])
    lower, upper = label.reveal_interval
    return np.concatenate((one_hot, np.asarray([
        float(lower - decision_step),
        float(upper - decision_step),
        float(label.expiry_step - decision_step),
        float(label.resolvable),
    ])))


def probe_a_oracle_relevance(
    current_features: np.ndarray,
    oracle_features: np.ndarray,
    delta_utility: Sequence[float],
    scenes: Sequence[object],
    datasets: Sequence[object],
    folds: Sequence[int],
) -> dict:
    current = _matrix(current_features, name="current_features")
    rows = len(current)
    oracle = _matrix(oracle_features, rows=rows, name="oracle_features")
    target = _vector(delta_utility, rows=rows, dtype=np.float64, name="delta_utility")
    scene = _vector(scenes, rows=rows, dtype=str, name="scenes")
    domain = _vector(datasets, rows=rows, dtype=str, name="datasets")
    fold = _vector(folds, rows=rows, dtype=np.int64, name="folds")
    if set(domain.tolist()) != set(REQUIRED_DOMAINS):
        raise ValueError("Probe A requires both R2R and RxR")
    current_prediction = ridge_scene_oof(current, target, fold)
    augmented_prediction = ridge_scene_oof(
        np.concatenate((current, oracle), axis=1), target, fold,
    )
    improvement = huber(current_prediction - target) - huber(
        augmented_prediction - target
    )
    results = {}
    failures = []
    for index, name in enumerate(REQUIRED_DOMAINS):
        interval = scene_bootstrap_mean(
            improvement,
            scene,
            domain == name,
            seed=BOOTSTRAP_SEED + index,
        )
        passed = interval["observed"] > 0.0 and interval["lower_95"] > 0.0
        results[name] = {"delta_huber": interval, "pass": passed}
        if not passed:
            failures.append(f"{name}:oracle_delta_huber_not_positive")
    return {
        "schema_version": "revealnav-mf3zo-probe-a/1",
        "probe": "A_oracle_relevance",
        "model": "fixed_ridge_l2_1_fold_fit_standardization",
        "domains": results,
        "failures": failures,
        "status": (
            "ORACLE_RELEVANCE_PASS"
            if not failures else "TEMPORAL_ORACLE_RELEVANCE_FAIL"
        ),
    }


def _macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    values = []
    for label in ("U", "A", "D"):
        tp = int(((truth == label) & (prediction == label)).sum())
        fp = int(((truth != label) & (prediction == label)).sum())
        fn = int(((truth == label) & (prediction != label)).sum())
        denominator = 2 * tp + fp + fn
        values.append(2.0 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def _metric_bootstrap(
    scenes: np.ndarray,
    mask: np.ndarray,
    metric: Callable[[np.ndarray], float],
    *,
    seed: int,
) -> dict:
    population = np.unique(scenes[mask])
    if len(population) < 2:
        raise ValueError("metric bootstrap needs at least two scenes")
    rows = {item: np.flatnonzero(mask & (scenes == item)) for item in population}
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.choice(population, len(population), replace=True)
        values[index] = metric(np.concatenate([rows[item] for item in draw]))
    observed_rows = np.flatnonzero(mask)
    return {
        "observed": float(metric(observed_rows)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _final_uad(
    probabilities: tuple[np.ndarray, np.ndarray, np.ndarray],
    factors: tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted: list[str] = []
    truth: list[str] = []
    for row in range(len(mask)):
        valid = mask[row]
        predicted.append(derive_uad(*(
            tuple(bool(item) for item in value[row, valid] >= 0.5)
            for value in probabilities
        ))[-1].value)
        truth.append(derive_uad(*(
            tuple(bool(item) for item in value[row, valid])
            for value in factors
        ))[-1].value)
    return np.asarray(predicted), np.asarray(truth)


def probe_b_temporal_observability(
    snapshot_features: np.ndarray,
    temporal_features: np.ndarray,
    target_in_set: np.ndarray,
    candidate_separated: np.ndarray,
    evidence_closed: np.ndarray,
    prefix_mask: np.ndarray,
    reveal_event: np.ndarray,
    reveal_at_risk: np.ndarray,
    expiry_event: np.ndarray,
    expiry_at_risk: np.ndarray,
    scenes: Sequence[object],
    datasets: Sequence[object],
    folds: Sequence[int],
) -> dict:
    snapshot = np.asarray(snapshot_features, dtype=np.float64)
    temporal = np.asarray(temporal_features, dtype=np.float64)
    if snapshot.ndim != 3 or temporal.ndim != 3 or snapshot.shape[:2] != temporal.shape[:2]:
        raise ValueError("Probe B requires aligned rank-3 causal tensors")
    rows, prefixes = snapshot.shape[:2]
    if not np.isfinite(snapshot).all() or not np.isfinite(temporal).all():
        raise ValueError("Probe B causal tensors are non-finite")
    mask = np.asarray(prefix_mask)
    labels = tuple(np.asarray(value) for value in (
        target_in_set, candidate_separated, evidence_closed,
        reveal_event, reveal_at_risk, expiry_event, expiry_at_risk,
    ))
    if mask.dtype != np.bool_ or mask.shape != (rows, prefixes) or any(
        value.dtype != np.bool_ or value.shape != mask.shape for value in labels
    ):
        raise ValueError("Probe B supervision masks must be aligned Booleans")
    in_set, separated, evidence, reveal, reveal_risk, expiry, expiry_risk = labels
    if np.any(separated & ~in_set) or np.any(evidence & ~in_set):
        raise ValueError("Probe B oracle factors violate UAD semantics")
    if any(np.any(value & ~mask) for value in labels):
        raise ValueError("Probe B oracle values exist outside causal prefix")
    scene = _vector(scenes, rows=rows, dtype=str, name="scenes")
    domain = _vector(datasets, rows=rows, dtype=str, name="datasets")
    event_fold = _vector(folds, rows=rows, dtype=np.int64, name="folds")
    flat_fold = np.repeat(event_fold, prefixes)
    flat_mask = mask.reshape(-1)
    flat_snapshot = snapshot.reshape(rows * prefixes, -1)
    flat_temporal = np.concatenate((snapshot, temporal), axis=2).reshape(
        rows * prefixes, -1
    )
    factor_targets = tuple(value.reshape(-1) for value in (in_set, separated, evidence))
    snapshot_probabilities = tuple(_sigmoid(ridge_scene_oof(
        flat_snapshot, 2.0 * value.astype(np.float64) - 1.0, flat_fold,
        eligible=flat_mask,
    )).reshape(rows, prefixes) for value in factor_targets)
    temporal_probabilities = tuple(_sigmoid(ridge_scene_oof(
        flat_temporal, 2.0 * value.astype(np.float64) - 1.0, flat_fold,
        eligible=flat_mask,
    )).reshape(rows, prefixes) for value in factor_targets)
    snapshot_uad, truth_uad = _final_uad(
        snapshot_probabilities, (in_set, separated, evidence), mask,
    )
    temporal_uad, _ = _final_uad(
        temporal_probabilities, (in_set, separated, evidence), mask,
    )

    flat_reveal_risk = reveal_risk.reshape(-1)
    flat_expiry_risk = expiry_risk.reshape(-1)
    snapshot_reveal = _sigmoid(ridge_scene_oof(
        flat_snapshot, 2.0 * reveal.reshape(-1).astype(float) - 1.0,
        flat_fold, eligible=flat_reveal_risk,
    ))
    temporal_reveal = _sigmoid(ridge_scene_oof(
        flat_temporal, 2.0 * reveal.reshape(-1).astype(float) - 1.0,
        flat_fold, eligible=flat_reveal_risk,
    ))
    snapshot_expiry = _sigmoid(ridge_scene_oof(
        flat_snapshot, 2.0 * expiry.reshape(-1).astype(float) - 1.0,
        flat_fold, eligible=flat_expiry_risk,
    ))
    temporal_expiry = _sigmoid(ridge_scene_oof(
        flat_temporal, 2.0 * expiry.reshape(-1).astype(float) - 1.0,
        flat_fold, eligible=flat_expiry_risk,
    ))
    reveal_improvement = np.zeros(rows * prefixes)
    reveal_improvement[flat_reveal_risk] = _binary_nll(
        snapshot_reveal[flat_reveal_risk], reveal.reshape(-1)[flat_reveal_risk]
    ) - _binary_nll(
        temporal_reveal[flat_reveal_risk], reveal.reshape(-1)[flat_reveal_risk]
    )
    expiry_improvement = np.zeros(rows * prefixes)
    expiry_improvement[flat_expiry_risk] = _binary_nll(
        snapshot_expiry[flat_expiry_risk], expiry.reshape(-1)[flat_expiry_risk]
    ) - _binary_nll(
        temporal_expiry[flat_expiry_risk], expiry.reshape(-1)[flat_expiry_risk]
    )
    flat_scene = np.repeat(scene, prefixes)
    flat_domain = np.repeat(domain, prefixes)
    results = {}
    failures = []
    for index, name in enumerate(REQUIRED_DOMAINS):
        domain_mask = domain == name
        uad = _metric_bootstrap(
            scene,
            domain_mask,
            lambda selected: _macro_f1(
                truth_uad[selected], temporal_uad[selected]
            ) - _macro_f1(truth_uad[selected], snapshot_uad[selected]),
            seed=BOOTSTRAP_SEED + 10 + 3 * index,
        )
        reveal_interval = scene_bootstrap_mean(
            reveal_improvement,
            flat_scene,
            (flat_domain == name) & flat_reveal_risk,
            seed=BOOTSTRAP_SEED + 11 + 3 * index,
        )
        expiry_interval = scene_bootstrap_mean(
            expiry_improvement,
            flat_scene,
            (flat_domain == name) & flat_expiry_risk,
            seed=BOOTSTRAP_SEED + 12 + 3 * index,
        )
        passed = all(
            value["observed"] > 0.0 and value["lower_95"] > 0.0
            for value in (uad, reveal_interval, expiry_interval)
        )
        results[name] = {
            "delta_uad_macro_f1": uad,
            "delta_reveal_nll": reveal_interval,
            "delta_expiry_nll": expiry_interval,
            "pass": passed,
        }
        if not passed:
            failures.append(f"{name}:primary_temporal_observability_not_positive")
    return {
        "schema_version": "revealnav-mf3zo-probe-b/1",
        "probe": "B_temporal_observability",
        "models": "fixed_ridge_l2_1_fold_fit_standardization",
        "domains": results,
        "failures": failures,
        "status": (
            "TEMPORAL_OBSERVABILITY_PASS"
            if not failures else "TEMPORAL_CAUSAL_OBSERVABILITY_FAIL"
        ),
    }


def _catastrophic_rate(mask: np.ndarray, target: np.ndarray) -> float | None:
    return (
        float((target[mask] <= CATASTROPHIC_THRESHOLD).mean())
        if mask.any() else None
    )


def _policy_evidence(
    selected: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> dict:
    results = {}
    for domain in REQUIRED_DOMAINS:
        domain_mask = datasets == domain
        selected_scenes = np.unique(scenes[domain_mask & selected])
        leave_one = [
            float(target[domain_mask & selected & (scenes != scene)].sum())
            for scene in selected_scenes
        ]
        fold_values = {}
        for fold in range(OUTER_FOLDS):
            stratum = domain_mask & (folds == fold)
            fold_values[str(fold)] = {
                "selected": int(selected[stratum].sum()),
                "utility": float(target[stratum & selected].sum()),
                "catastrophic_rate": _catastrophic_rate(
                    stratum & selected, target,
                ),
            }
        results[domain] = {
            "selected": int((domain_mask & selected).sum()),
            "utility": float(target[domain_mask & selected].sum()),
            "minimum_leave_one_scene_utility": min(leave_one) if leave_one else None,
            "catastrophic_rate": _catastrophic_rate(domain_mask & selected, target),
            "folds": fold_values,
        }
    return results


def matched_budget_baselines(
    selected: np.ndarray,
    proposal_score: np.ndarray,
    native_margin: np.ndarray,
    identities: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> Mapping[str, np.ndarray]:
    rows = len(selected)
    outputs = {
        "high_proposal_score": np.zeros(rows, dtype=np.bool_),
        "low_native_margin": np.zeros(rows, dtype=np.bool_),
        "fixed_random": np.zeros(rows, dtype=np.bool_),
    }
    for fold in range(OUTER_FOLDS):
        for domain in REQUIRED_DOMAINS:
            stratum = np.flatnonzero((folds == fold) & (datasets == domain))
            budget = int(selected[stratum].sum())
            if budget == 0:
                continue
            keys = {
                "high_proposal_score": [
                    (-float(proposal_score[index]), str(identities[index]), index)
                    for index in stratum
                ],
                "low_native_margin": [
                    (float(native_margin[index]), str(identities[index]), index)
                    for index in stratum
                ],
                "fixed_random": [
                    (hashlib.sha256(
                        f"mf3zo-random/1\0{identities[index]}".encode("utf-8")
                    ).hexdigest(), str(identities[index]), index)
                    for index in stratum
                ],
            }
            for name, values in keys.items():
                values.sort()
                outputs[name][[item[-1] for item in values[:budget]]] = True
    return outputs


def probe_c_learned_state_relevance(
    current_features: np.ndarray,
    frozen_temporal_state: np.ndarray,
    delta_utility: Sequence[float],
    scenes: Sequence[object],
    datasets: Sequence[object],
    folds: Sequence[int],
    proposal_score: Sequence[float],
    native_margin: Sequence[float],
    identities: Sequence[object],
) -> dict:
    current = _matrix(current_features, name="current_features")
    rows = len(current)
    state = _matrix(frozen_temporal_state, rows=rows, name="frozen_temporal_state")
    target = _vector(delta_utility, rows=rows, dtype=np.float64, name="target")
    scene = _vector(scenes, rows=rows, dtype=str, name="scenes")
    domain = _vector(datasets, rows=rows, dtype=str, name="datasets")
    fold = _vector(folds, rows=rows, dtype=np.int64, name="folds")
    score = _vector(proposal_score, rows=rows, dtype=np.float64, name="proposal_score")
    margin = _vector(native_margin, rows=rows, dtype=np.float64, name="native_margin")
    identity = _vector(identities, rows=rows, dtype=str, name="identities")
    current_prediction = ridge_scene_oof(current, target, fold)
    temporal_prediction = ridge_scene_oof(
        np.concatenate((current, state), axis=1), target, fold,
    )
    selected = temporal_prediction > 0.0
    improvement = huber(current_prediction - target) - huber(
        temporal_prediction - target
    )
    evidence = _policy_evidence(selected, target, scene, domain, fold)
    baseline_masks = matched_budget_baselines(
        selected, score, margin, identity, domain, fold,
    )
    baseline_evidence = {
        name: _policy_evidence(mask, target, scene, domain, fold)
        for name, mask in baseline_masks.items()
    }
    failures = []
    intervals = {}
    for index, name in enumerate(REQUIRED_DOMAINS):
        interval = scene_bootstrap_mean(
            improvement,
            scene,
            domain == name,
            seed=BOOTSTRAP_SEED + 30 + index,
        )
        intervals[name] = interval
        value = evidence[name]
        deterministic = [
            baseline_evidence[item][name]
            for item in ("high_proposal_score", "low_native_margin")
        ]
        strongest_risk = min(
            item["catastrophic_rate"]
            for item in deterministic
            if item["catastrophic_rate"] is not None
        )
        if interval["observed"] <= 0.0 or interval["lower_95"] <= 0.0:
            failures.append(f"{name}:temporal_delta_huber_not_positive")
        if value["utility"] <= 0.0:
            failures.append(f"{name}:selected_utility_not_positive")
        if value["minimum_leave_one_scene_utility"] is None or value[
            "minimum_leave_one_scene_utility"
        ] <= 0.0:
            failures.append(f"{name}:leave_one_scene_utility_not_positive")
        if any(item["selected"] == 0 or item["utility"] < 0.0 for item in value["folds"].values()):
            failures.append(f"{name}:negative_or_zero_coverage_fold")
        if value["catastrophic_rate"] is None or value[
            "catastrophic_rate"
        ] > strongest_risk:
            failures.append(f"{name}:catastrophic_rate_above_strongest_baseline")
    return {
        "schema_version": "revealnav-mf3zo-probe-c/1",
        "probe": "C_learned_temporal_state_relevance",
        "decision_rule": "prediction > 0",
        "delta_huber": intervals,
        "temporal_policy": evidence,
        "matched_baselines": baseline_evidence,
        "failures": failures,
        "status": (
            "LEARNED_TEMPORAL_STATE_RELEVANCE_PASS"
            if not failures else "LEARNED_TEMPORAL_STATE_RELEVANCE_FAIL"
        ),
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CATASTROPHIC_THRESHOLD",
    "HUBER_DELTA",
    "OUTER_FOLDS",
    "REQUIRED_DOMAINS",
    "RIDGE_L2",
    "SCENE_FOLD_SALT",
    "assign_scene_folds",
    "current_snapshot_features",
    "fit_standardizer",
    "huber",
    "matched_budget_baselines",
    "oracle_feature_vector",
    "probe_a_oracle_relevance",
    "probe_b_temporal_observability",
    "probe_c_learned_state_relevance",
    "ridge_scene_oof",
    "scene_bootstrap_mean",
]
