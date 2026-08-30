"""Nested scene-level selection for the versioned MF3ZK revision.

The original MF3ZK trainer selected a rule after scanning all scene-OOF
predictions.  This module keeps the estimator deliberately small, but moves
every L2/threshold choice inside an inner scene split.  The outer predictions
are used only for the unbiased development report and for a pre-declared
modal/median rule aggregation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from collections.abc import Sequence

import numpy as np


HARM_LABEL_THRESHOLD = -0.05
CATASTROPHIC_THRESHOLD = -0.10


class NestedSelectionError(RuntimeError):
    """Raised when a required scene-level split or inner rule is invalid."""


def _mask(mask: np.ndarray, length: int, name: str) -> np.ndarray:
    """Return a one-dimensional boolean mask with an explicit row contract."""

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 1 or len(value) != length:
        raise ValueError(f"{name} mask has the wrong shape")
    return value


def deterministic_scene_folds(
    scenes: Sequence[object], n_folds: int, *, salt: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign whole scenes to balanced, deterministic folds.

    Sorting hashed scene IDs before round-robin assignment avoids an accidental
    empty inner fold while keeping the assignment independent of row order.
    """

    if n_folds < 2:
        raise ValueError("scene fold count must be at least two")
    unique = sorted({str(scene) for scene in scenes})
    if len(unique) < n_folds:
        raise NestedSelectionError(
            f"need at least {n_folds} scenes, found {len(unique)}"
        )
    ordered = sorted(
        unique,
        key=lambda scene: hashlib.sha256(
            f"{salt}\0{scene}".encode("utf-8")
        ).hexdigest(),
    )
    mapping = {scene: index % n_folds for index, scene in enumerate(ordered)}
    folds = np.asarray([mapping[str(scene)] for scene in scenes], dtype=np.int64)
    if set(mapping.values()) != set(range(n_folds)):
        raise NestedSelectionError("scene fold assignment left an empty fold")
    return folds, mapping


def canonicalize_exact_counterfactual_rows(
    rows: Sequence[dict], hierarchy: dict,
) -> tuple[list[dict], dict]:
    """Collapse byte-identical source overlap and assign the frozen tier.

    The historical core and expansion collectors can contain the same exact
    episode/step counterfactual.  A pooled estimator must not count that label
    twice.  Conflicting records at the same decision identity are rejected;
    retained rows are assigned to the tier implied by the frozen hierarchy,
    not by their source filename.
    """

    expansion = float(hierarchy["expansion_score_threshold"])
    core = float(hierarchy["core_score_threshold"])
    upper = float(hierarchy["score_upper_threshold"])
    if not (
        math.isfinite(expansion)
        and math.isfinite(core)
        and math.isfinite(upper)
        and expansion < core < upper
    ):
        raise NestedSelectionError("invalid frozen proposal hierarchy")
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        decision = row.get("decision", {})
        key = (
            str(row["dataset"]),
            str(row["episode_id"]),
            int(decision.get("step", row.get("decision_step", -1))),
        )
        if key[2] < 0:
            raise NestedSelectionError("counterfactual row has no decision step")
        groups.setdefault(key, []).append(row)

    canonical = []
    source_to_frozen = Counter()
    duplicate_groups = 0
    collapsed = 0
    for key in sorted(groups):
        group = groups[key]
        reference = group[0]
        signature = (
            str(reference["scene_id"]),
            reference["decision"],
            float(reference["target"]),
            str(reference["feature"]["sha256"]),
        )
        for row in group[1:]:
            other = (
                str(row["scene_id"]),
                row["decision"],
                float(row["target"]),
                str(row["feature"]["sha256"]),
            )
            if other != signature:
                raise NestedSelectionError(
                    f"conflicting counterfactual rows at {key}"
                )
        score = float(reference["decision"]["policy_risk_adjusted_score"])
        frozen_tier = (
            "core" if core < score <= upper
            else "expansion" if expansion < score <= core
            else None
        )
        if frozen_tier is None:
            raise NestedSelectionError(
                f"counterfactual row falls outside frozen hierarchy at {key}"
            )
        source_tiers = sorted({str(row["tier"]) for row in group})
        for source_tier in source_tiers:
            source_to_frozen[(source_tier, frozen_tier)] += 1
        retained = dict(reference)
        retained["tier"] = frozen_tier
        retained["source_tiers"] = source_tiers
        retained["source_records"] = [
            {
                "tier": str(row["tier"]),
                "manifest": str(row["source_manifest"]),
                "row_index": int(row["source_row_index"]),
                "feature_sha256": str(row["feature"]["sha256"]),
            }
            for row in sorted(
                group,
                key=lambda value: (
                    str(value["tier"]), int(value["source_row_index"])
                ),
            )
        ]
        canonical.append(retained)
        if len(group) > 1:
            duplicate_groups += 1
            collapsed += len(group) - 1
    return canonical, {
        "source_rows": len(rows),
        "canonical_rows": len(canonical),
        "duplicate_decision_groups": duplicate_groups,
        "duplicate_rows_collapsed": collapsed,
        "source_to_frozen_tier_counts": {
            f"{source}/{frozen}": count
            for (source, frozen), count in sorted(source_to_frozen.items())
        },
        "frozen_tier_counts": dict(sorted(Counter(
            row["tier"] for row in canonical
        ).items())),
        "conflicting_groups": 0,
    }


def _validate_arrays(
    matrix: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
) -> None:
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("feature matrix must be a non-empty 2-D array")
    if not (
        target.ndim == scenes.ndim == datasets.ndim == 1
        and len(target) == len(scenes) == len(datasets) == len(matrix)
    ):
        raise ValueError("row arrays have inconsistent lengths")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("training arrays contain non-finite values")
    if not set(str(value) for value in datasets):
        raise ValueError("training data has no benchmark domain")


def dataset_weights(datasets: np.ndarray) -> np.ndarray:
    """Give each benchmark equal effective weight in a pooled fit."""

    values = np.asarray([str(value) for value in datasets])
    domains = sorted(set(values))
    if len(domains) == 1:
        return np.ones(len(values), dtype=np.float64)
    result = np.zeros(len(values), dtype=np.float64)
    for domain in domains:
        mask = values == domain
        result[mask] = len(values) / (len(domains) * int(mask.sum()))
    return result


def _standardize(
    matrix: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = max(float(weights.sum()), 1e-12)
    mean = (matrix * weights[:, None]).sum(0) / total
    variance = (((matrix - mean) ** 2) * weights[:, None]).sum(0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return (matrix - mean) / scale, mean, scale


def _design(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((np.ones((len(matrix), 1)), matrix), axis=1)


def _ridge_fit(
    matrix: np.ndarray, target: np.ndarray, weights: np.ndarray, l2: float,
) -> np.ndarray:
    design = _design(matrix)
    weighted = weights[:, None] * design
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 1e-8
    return np.linalg.solve(
        design.T @ weighted + penalty,
        design.T @ (weights * target),
    )


def _logistic_fit(
    matrix: np.ndarray, target: np.ndarray, weights: np.ndarray, l2: float,
) -> np.ndarray:
    design = _design(matrix)
    prior = (float((weights * target).sum()) + 0.5) / (
        float(weights.sum()) + 1.0
    )
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prior / (1.0 - prior))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 1e-8
    for _ in range(80):
        logits = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.maximum(probability * (1.0 - probability), 1e-5)
        gradient = design.T @ (weights * (probability - target))
        gradient += penalty @ coefficients
        hessian = design.T @ ((weights * curvature)[:, None] * design)
        hessian += penalty
        update = np.linalg.solve(hessian, gradient)
        coefficients -= update
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return coefficients


def bootstrap_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    l2: float,
    seed: int,
    *,
    bootstraps: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit a scene-cluster bootstrap ensemble for return and harm."""

    _validate_arrays(matrix, target, scenes, datasets)
    if bootstraps < 1 or l2 <= 0 or not math.isfinite(float(l2)):
        raise ValueError("invalid bootstrap fit configuration")
    rng = np.random.default_rng(seed)
    values = np.asarray([str(value) for value in datasets])
    scene_values = np.asarray([str(value) for value in scenes])
    all_scenes = np.unique(scene_values)
    harm = (target <= HARM_LABEL_THRESHOLD).astype(np.float64)
    models = []
    for _ in range(int(bootstraps)):
        # A shared MP3D scene is one statistical cluster even when it appears
        # in both RxR and R2R.  Sample the union once, then apply benchmark
        # balancing through observation weights; sampling it independently per
        # domain would understate cross-benchmark scene correlation.
        sampled = rng.choice(all_scenes, size=len(all_scenes), replace=True)
        selected_indices: list[int] = []
        for scene in sampled:
            selected_indices.extend(
                int(index) for index in np.flatnonzero(scene_values == scene)
            )
        selected = np.asarray(selected_indices, dtype=np.int64)
        if len(selected) == 0:
            raise NestedSelectionError("empty scene bootstrap sample")
        weights = dataset_weights(values[selected])
        standardized, mean, scale = _standardize(matrix[selected], weights)
        models.append((
            mean,
            scale,
            _ridge_fit(standardized, target[selected], weights, l2),
            _logistic_fit(standardized, harm[selected], weights, l2),
        ))
    return models


def predict_ensemble(
    models: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not models or matrix.ndim != 2:
        raise ValueError("invalid ensemble prediction input")
    expected_members = []
    harm_members = []
    for mean, scale, return_coef, harm_coef in models:
        standardized = (matrix - mean) / scale
        design = _design(standardized)
        expected_members.append(design @ return_coef)
        logits = np.clip(design @ harm_coef, -30.0, 30.0)
        harm_members.append(1.0 / (1.0 + np.exp(-logits)))
    expected = np.stack(expected_members, axis=1)
    harm = np.stack(harm_members, axis=1)
    median = np.median(expected, axis=1)
    robust = median - 0.5 * np.median(
        np.abs(expected - median[:, None]), axis=1
    )
    upper_harm = np.quantile(harm, 0.75, axis=1)
    if not np.isfinite(robust).all() or not np.isfinite(upper_harm).all():
        raise NestedSelectionError("non-finite ensemble prediction")
    return robust, upper_harm


def scene_crossfit_predictions(
    matrix: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    fold_ids: np.ndarray,
    n_folds: int,
    l2: float,
    seed: int,
    *,
    bootstraps: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Produce predictions with a fixed whole-scene fold assignment."""

    _validate_arrays(matrix, target, scenes, datasets)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if len(fold_ids) != len(target) or set(fold_ids) != set(range(n_folds)):
        raise NestedSelectionError("scene cross-fit has an empty fold")
    expected = np.zeros(len(target), dtype=np.float64)
    harm = np.zeros(len(target), dtype=np.float64)
    evidence = []
    for fold in range(n_folds):
        evaluate = fold_ids == fold
        fit = ~evaluate
        if not evaluate.any() or not fit.any():
            raise NestedSelectionError(
                f"scene cross-fit fold {fold} has an empty partition"
            )
        fit_scenes = set(str(value) for value in scenes[fit])
        eval_scenes = set(str(value) for value in scenes[evaluate])
        overlap = sorted(fit_scenes & eval_scenes)
        if overlap:
            raise NestedSelectionError(
                f"scene overlap in fold {fold}: {overlap}"
            )
        models = bootstrap_fit(
            matrix[fit], target[fit], scenes[fit], datasets[fit],
            l2, seed + fold * 1009, bootstraps=bootstraps,
        )
        expected[evaluate], harm[evaluate] = predict_ensemble(
            models, matrix[evaluate]
        )
        evidence.append({
            "fold": int(fold),
            "fit_rows": int(fit.sum()),
            "evaluation_rows": int(evaluate.sum()),
            "fit_scenes": sorted(fit_scenes),
            "evaluation_scenes": sorted(eval_scenes),
            "scene_overlap": overlap,
            "fit_domains": sorted(set(str(value) for value in datasets[fit])),
            "evaluation_domains": sorted(
                set(str(value) for value in datasets[evaluate])
            ),
        })
    return expected, harm, evidence


def rule_grid(expected: np.ndarray, harm: np.ndarray):
    """Yield a fixed, prediction-only operating-point grid."""

    if len(expected) == 0 or len(expected) != len(harm):
        raise ValueError("empty or mismatched rule-grid inputs")
    return_thresholds = np.unique(
        np.quantile(expected, np.linspace(0.0, 0.90, 19))
    )
    harm_thresholds = np.unique(
        np.quantile(harm, np.linspace(0.10, 1.0, 19))
    )
    for return_threshold in return_thresholds:
        for harm_threshold in harm_thresholds:
            yield (
                float(return_threshold),
                float(harm_threshold),
                (expected >= return_threshold) & (harm <= harm_threshold),
            )


def outcome_evidence(
    mask: np.ndarray, target: np.ndarray, scenes: np.ndarray,
) -> dict:
    target = np.asarray(target, dtype=np.float64)
    mask = _mask(mask, len(target), "outcome")
    if target.ndim != 1:
        raise ValueError("outcome target must be one-dimensional")
    if len(scenes) != len(target):
        raise ValueError("outcome scenes have the wrong length")
    # Keep scene comparisons type-stable even for diagnostic callers that use
    # integer row IDs as a synthetic scene array.
    scenes = np.asarray([str(value) for value in scenes])
    selected = target[mask]
    selected_scenes = sorted(set(str(value) for value in scenes[mask]))
    leave_one = [
        float(target[mask & (scenes != scene)].sum())
        for scene in selected_scenes
    ]
    total = float(selected.sum())
    return {
        "eligible": int(len(target)),
        "authorized": int(mask.sum()),
        "coverage": float(mask.mean()) if len(mask) else 0.0,
        "positive": int((selected > 1e-8).sum()),
        "negative": int((selected < -1e-8).sum()),
        "ties": int((np.abs(selected) <= 1e-8).sum()),
        "catastrophic": int((selected <= CATASTROPHIC_THRESHOLD).sum()),
        "catastrophic_rate": (
            float((selected <= CATASTROPHIC_THRESHOLD).mean())
            if len(selected) else 0.0
        ),
        "total_utility": total,
        "deployed_mean_utility": total / len(target) if len(target) else 0.0,
        "selected_mean_utility": float(selected.mean()) if len(selected) else 0.0,
        "minimum_leave_one_selected_scene_out_total": (
            min(leave_one) if leave_one else 0.0
        ),
        "selected_scene_count": len(selected_scenes),
    }


def domain_evidence(
    mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
) -> dict:
    target = np.asarray(target, dtype=np.float64)
    mask = _mask(mask, len(target), "domain")
    scenes = np.asarray(scenes)
    datasets = np.asarray(datasets)
    if scenes.ndim != 1 or datasets.ndim != 1 or len(scenes) != len(target) \
            or len(datasets) != len(target):
        raise ValueError("domain evidence arrays have inconsistent lengths")
    result = {}
    for domain in sorted(set(str(value) for value in datasets)):
        domain_mask = np.asarray(
            [str(value) == domain for value in datasets], dtype=bool
        )
        result[domain] = outcome_evidence(
            mask[domain_mask], target[domain_mask], scenes[domain_mask]
        )
    return result


def choose_inner_rule(
    expected: np.ndarray,
    harm: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    *,
    minimum_authorized: int,
    minimum_per_domain: int,
) -> dict:
    """Select one rule using only one inner OOF cohort.

    Catastrophic harm is constrained to be non-increasing (``<=``), not
    strictly decreasing.  Requiring a strict decrease would reject a valid
    no-catastrophe cohort for a purely syntactic reason.
    """

    ungated = outcome_evidence(
        np.ones(len(target), dtype=bool), target, scenes
    )
    ungated_domains = domain_evidence(
        np.ones(len(target), dtype=bool), target, scenes, datasets
    )
    candidates = []
    for return_threshold, harm_threshold, mask in rule_grid(expected, harm):
        evidence = outcome_evidence(mask, target, scenes)
        domains = domain_evidence(mask, target, scenes, datasets)
        feasible = (
            evidence["authorized"] >= minimum_authorized
            and evidence["total_utility"] > 0.0
            and evidence["catastrophic_rate"] <= ungated["catastrophic_rate"]
            and evidence["minimum_leave_one_selected_scene_out_total"] > 0.0
            and all(
                value["authorized"] >= minimum_per_domain
                and value["total_utility"] > 0.0
                and value["catastrophic_rate"]
                <= ungated_domains[domain]["catastrophic_rate"]
                and value["minimum_leave_one_selected_scene_out_total"] > 0.0
                for domain, value in domains.items()
            )
        )
        candidates.append({
            "return_threshold": return_threshold,
            "harm_probability_threshold": harm_threshold,
            "feasible": bool(feasible),
            **evidence,
            "domains": domains,
        })
    if not candidates:
        raise NestedSelectionError("inner rule grid is empty")
    best_any = max(
        candidates,
        key=lambda value: (
            value["feasible"],
            value["total_utility"],
            -value["catastrophic"],
            -value["authorized"],
            -value["return_threshold"],
            -value["harm_probability_threshold"],
        ),
    )
    feasible = [value for value in candidates if value["feasible"]]
    best = max(
        feasible,
        key=lambda value: (
            value["total_utility"],
            -value["catastrophic"],
            -value["authorized"],
            -value["return_threshold"],
            -value["harm_probability_threshold"],
        ),
    ) if feasible else None
    return {
        "selected_rule": best,
        "best_any_rule": best_any,
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "ungated": ungated,
        "ungated_domains": ungated_domains,
    }


def _modal_numeric(values: Sequence[float]) -> float:
    counts = Counter(float(value) for value in values)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def nested_scene_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    outer_folds: np.ndarray,
    *,
    outer_fold_count: int,
    inner_fold_count: int,
    l2_grid: Sequence[float],
    seed: int,
    bootstraps: int,
    minimum_authorized: int,
    minimum_per_domain: int,
    inner_salt: str,
) -> dict:
    """Fit a pooled gate with nested whole-scene selection.

    ``outer_oof`` is never used to select a threshold.  The final operating
    point is the median threshold across outer-fold inner selections and the
    modal L2 (ties choose the smaller value).
    """

    _validate_arrays(matrix, target, scenes, datasets)
    outer_folds = np.asarray(outer_folds, dtype=np.int64)
    if len(outer_folds) != len(target) or set(outer_folds) != set(
        range(outer_fold_count)
    ):
        raise NestedSelectionError("outer scene folds are incomplete")
    scene_fold_values: dict[str, set[int]] = {}
    for scene, fold in zip(scenes, outer_folds):
        scene_fold_values.setdefault(str(scene), set()).add(int(fold))
    if any(len(values) != 1 for values in scene_fold_values.values()):
        raise NestedSelectionError("one MP3D scene was split across outer folds")
    outer_expected = np.zeros(len(target), dtype=np.float64)
    outer_harm = np.zeros(len(target), dtype=np.float64)
    # Each row is scored with the rule selected without its outer scene.  This
    # is the only authorization mask used for the unbiased development report.
    outer_return_safe = np.zeros(len(target), dtype=bool)
    outer_harm_safe = np.zeros(len(target), dtype=bool)
    outer_authorized = np.zeros(len(target), dtype=bool)
    outer_return_threshold = np.full(len(target), np.nan, dtype=np.float64)
    outer_harm_threshold = np.full(len(target), np.nan, dtype=np.float64)
    outer_records = []
    selected_l2 = []
    selected_return = []
    selected_harm = []

    for outer_fold in range(outer_fold_count):
        outer_eval = outer_folds == outer_fold
        outer_fit = ~outer_eval
        fit_scenes = sorted(set(str(value) for value in scenes[outer_fit]))
        eval_scenes = sorted(set(str(value) for value in scenes[outer_eval]))
        if not fit_scenes or not eval_scenes:
            raise NestedSelectionError(
                f"outer fold {outer_fold} has an empty scene partition"
            )
        if set(fit_scenes) & set(eval_scenes):
            raise NestedSelectionError(
                f"outer fold {outer_fold} has scene overlap"
            )
        inner_folds, inner_mapping = deterministic_scene_folds(
            fit_scenes, inner_fold_count,
            salt=f"{inner_salt}:outer:{outer_fold}",
        )
        inner_row_folds = np.asarray(
            [inner_mapping[str(scene)] for scene in scenes[outer_fit]],
            dtype=np.int64,
        )
        l2_trials = []
        for l2 in l2_grid:
            inner_expected, inner_harm, inner_cv = scene_crossfit_predictions(
                matrix[outer_fit], target[outer_fit], scenes[outer_fit],
                datasets[outer_fit], inner_row_folds, inner_fold_count,
                float(l2), seed + outer_fold * 10000 + int(float(l2) * 100),
                bootstraps=bootstraps,
            )
            selection = choose_inner_rule(
                inner_expected, inner_harm, target[outer_fit],
                scenes[outer_fit], datasets[outer_fit],
                minimum_authorized=minimum_authorized,
                minimum_per_domain=minimum_per_domain,
            )
            l2_trials.append({
                "l2": float(l2),
                "inner_cv": inner_cv,
                **selection,
            })
        feasible_trials = [
            trial for trial in l2_trials
            if trial["selected_rule"] is not None
        ]
        if not feasible_trials:
            raise NestedSelectionError(
                f"outer fold {outer_fold} has no feasible inner rule"
            )
        chosen_trial = max(
            feasible_trials,
            key=lambda trial: (
                trial["selected_rule"]["total_utility"],
                -trial["selected_rule"]["catastrophic"],
                -trial["selected_rule"]["authorized"],
                -float(trial["l2"]),
            ),
        )
        chosen = chosen_trial["selected_rule"]
        models = bootstrap_fit(
            matrix[outer_fit], target[outer_fit], scenes[outer_fit],
            datasets[outer_fit], float(chosen_trial["l2"]),
            seed + 50000 + outer_fold * 1009,
            bootstraps=bootstraps,
        )
        outer_expected[outer_eval], outer_harm[outer_eval] = predict_ensemble(
            models, matrix[outer_eval]
        )
        fold_return_safe = (
            outer_expected[outer_eval] >= float(chosen["return_threshold"])
        )
        fold_harm_safe = (
            outer_harm[outer_eval]
            <= float(chosen["harm_probability_threshold"])
        )
        outer_return_safe[outer_eval] = fold_return_safe
        outer_harm_safe[outer_eval] = fold_harm_safe
        outer_authorized[outer_eval] = fold_return_safe & fold_harm_safe
        outer_return_threshold[outer_eval] = float(chosen["return_threshold"])
        outer_harm_threshold[outer_eval] = float(
            chosen["harm_probability_threshold"]
        )
        selected_l2.append(float(chosen_trial["l2"]))
        selected_return.append(float(chosen["return_threshold"]))
        selected_harm.append(float(chosen["harm_probability_threshold"]))
        outer_records.append({
            "outer_fold": int(outer_fold),
            "fit_scenes": fit_scenes,
            "evaluation_scenes": eval_scenes,
            "scene_overlap": [],
            "inner_fold_count": int(inner_fold_count),
            "inner_scene_assignment": inner_mapping,
            "trials": l2_trials,
            "selected_l2": float(chosen_trial["l2"]),
            "selected_rule": chosen,
            "outer_evaluation_rows": int(outer_eval.sum()),
            "outer_evaluation_evidence": outcome_evidence(
                outer_authorized[outer_eval], target[outer_eval],
                scenes[outer_eval],
            ),
        })

    final_l2 = _modal_numeric(selected_l2)
    final_rule = {
        "l2": final_l2,
        "return_threshold": float(np.median(selected_return)),
        "harm_probability_threshold": float(np.median(selected_harm)),
        "aggregation": "modal_l2_median_outer_inner_thresholds",
        "selection_source": "inner_scene_oof_only",
        "outer_fold_l2_values": selected_l2,
        "outer_fold_return_thresholds": selected_return,
        "outer_fold_harm_thresholds": selected_harm,
    }
    # Do not apply the final aggregated rule back to outer predictions when
    # reporting development evidence: that rule can depend on information
    # from other outer folds.  It remains the frozen rule for the subsequent
    # all-data fit and fresh confirmation.
    aggregated_mask = (
        (outer_expected >= final_rule["return_threshold"])
        & (outer_harm <= final_rule["harm_probability_threshold"])
    )
    nested_evidence = outcome_evidence(outer_authorized, target, scenes)
    nested_domains = domain_evidence(
        outer_authorized, target, scenes, datasets
    )
    if not (
        np.isfinite(outer_return_threshold).all()
        and np.isfinite(outer_harm_threshold).all()
    ):
        raise NestedSelectionError("outer fold left an unassigned rule")
    aggregated_evidence = outcome_evidence(aggregated_mask, target, scenes)
    aggregated_domains = domain_evidence(
        aggregated_mask, target, scenes, datasets
    )
    final_models = bootstrap_fit(
        matrix, target, scenes, datasets, final_l2, seed + 900000,
        bootstraps=bootstraps,
    )
    return {
        "status": "NESTED_SELECTION_PASS",
        "final_rule": final_rule,
        "outer_oof": {
            "expected": outer_expected,
            "upper_harm": outer_harm,
            "return_safe_mask": outer_return_safe,
            "harm_safe_mask": outer_harm_safe,
            "authorized_mask": outer_authorized,
            "row_return_threshold": outer_return_threshold,
            "row_harm_threshold": outer_harm_threshold,
            "aggregated_rule_mask": aggregated_mask,
            "evidence": nested_evidence,
            "domains": nested_domains,
            "aggregated_rule_evidence": aggregated_evidence,
            "aggregated_rule_domains": aggregated_domains,
        },
        "outer_folds": outer_records,
        "final_models": final_models,
        "modal_l2": final_l2,
    }


def risk_coverage_curve(
    expected: np.ndarray,
    harm: np.ndarray,
    target: np.ndarray,
    *,
    points: int = 21,
) -> list[dict]:
    """Report a fixed diagnostic curve; it never selects an operating point."""

    if points < 2 or len(expected) == 0:
        raise ValueError("risk-coverage curve needs at least two points")
    values = np.asarray(expected, dtype=np.float64)
    rows = []
    for quantile in np.linspace(0.0, 1.0, points):
        threshold = float(np.quantile(values, quantile))
        mask = values >= threshold
        evidence = outcome_evidence(mask, target, np.arange(len(target)))
        selected_harm = np.asarray(harm)[mask]
        rows.append({
            "quantile": float(quantile),
            "score_name": "robust_expected_utility",
            "threshold": threshold,
            "coverage": evidence["coverage"],
            "authorized": evidence["authorized"],
            "deployed_mean_utility": evidence["deployed_mean_utility"],
            "catastrophic_rate": evidence["catastrophic_rate"],
            "predicted_harm_safe_rate": (
                float((selected_harm <= 0.5).mean())
                if len(selected_harm) else 0.0
            ),
            "selection_used": False,
        })
    return rows


def _proposal_candidate_mask(rows: Sequence[dict]) -> np.ndarray:
    """Validate the proposal-side fields available on supervised rows."""

    candidate = []
    for row in rows:
        decision = row.get("decision", {})
        ids = decision.get("current_local_action_ids", ())
        score = float(decision.get("policy_risk_adjusted_score", np.nan))
        candidate.append(
            isinstance(ids, (list, tuple))
            and len(ids) >= 2
            and len({str(value) for value in ids}) == len(ids)
            and math.isfinite(score)
            and math.isfinite(float(decision.get("native_margin", np.nan)))
        )
    return np.asarray(candidate, dtype=bool)


def equal_budget_baselines(
    rows: Sequence[dict],
    target: np.ndarray,
    gate_mask: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Evaluate simple proposal baselines at the gate's exact budget.

    The candidate population and intervention count are fixed before scoring;
    no baseline parameter is selected from the target outcomes.  This exposes
    whether a gate gain is merely a coverage/uncertainty ranking effect.
    """

    target = np.asarray(target, dtype=np.float64)
    gate_mask = _mask(gate_mask, len(target), "equal-budget gate")
    candidates = _proposal_candidate_mask(rows)
    gate_mask &= candidates
    candidate_indices = np.flatnonzero(candidates)
    budget = int(gate_mask.sum())
    if budget > len(candidate_indices):
        raise ValueError("gate budget exceeds proposal candidate population")

    def select_order(order: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(rows), dtype=bool)
        mask[order[:budget]] = True
        return mask

    native_margin = np.full(len(rows), np.nan, dtype=np.float64)
    proposal_score = np.full(len(rows), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        if candidates[index]:
            decision = row["decision"]
            native_margin[index] = float(decision["native_margin"])
            proposal_score[index] = float(
                decision["policy_risk_adjusted_score"]
            )
    orders = {
        "native_margin_low": candidate_indices[np.argsort(
            native_margin[candidate_indices], kind="stable"
        )],
        "proposal_score_high": candidate_indices[np.argsort(
            -proposal_score[candidate_indices], kind="stable"
        )],
    }
    rng = np.random.default_rng(int(seed))
    orders["uniform_random"] = rng.permutation(candidate_indices)
    masks = {"nested_gate": gate_mask}
    masks.update({name: select_order(order) for name, order in orders.items()})
    scenes = np.asarray([str(row["scene_id"]) for row in rows])
    result = {}
    for name, mask in masks.items():
        evidence = outcome_evidence(mask, target, scenes)
        result[name] = {
            **evidence,
            "budget": budget,
            "candidate_population": int(len(candidate_indices)),
            "selection_used": False,
        }
    return {
        "scope": "outer_oof_proposal_candidates",
        "budget_definition": "exact nested_gate_authorized_count",
        "candidate_definition": "finite native margin/policy score and >=2 unique IDs",
        "selection_used": False,
        "seed": int(seed),
        "comparisons": result,
    }


def coverage_funnel(
    rows: Sequence[dict],
    expected: np.ndarray,
    harm: np.ndarray,
    rule: dict,
    hierarchy: dict,
    *,
    source_population: dict | None = None,
    return_safe_mask: np.ndarray | None = None,
    harm_safe_mask: np.ndarray | None = None,
    authorized_mask: np.ndarray | None = None,
    actually_changed_mask: np.ndarray | None = None,
) -> dict:
    """Summarize proposal-to-outcome coverage without inventing missing rows.

    The default masks apply the final all-data rule.  Nested training passes
    the outer-fold masks instead, so the funnel remains an honest development
    diagnostic.  ``rows`` are exact one-switch records; when no explicit
    change mask is supplied, that source contract is recorded rather than
    inferred from the utility target.
    """

    expected = np.asarray(expected, dtype=np.float64)
    harm = np.asarray(harm, dtype=np.float64)
    if len(rows) != len(expected) or len(rows) != len(harm):
        raise ValueError("coverage rows and predictions are misaligned")
    if expected.ndim != 1 or harm.ndim != 1 or not (
        np.isfinite(expected).all() and np.isfinite(harm).all()
    ):
        raise ValueError("coverage predictions must be finite vectors")
    target = np.asarray(
        [float(row["target"]) for row in rows], dtype=np.float64
    )
    if not np.isfinite(target).all():
        raise ValueError("coverage targets must be finite")
    row_scenes = np.asarray([str(row["scene_id"]) for row in rows])
    tiers = []
    scores = []
    for row in rows:
        decision = row.get("decision", {})
        score = float(decision.get("policy_risk_adjusted_score", np.nan))
        tiers.append(str(row.get("tier", "unknown")))
        scores.append(score)
    candidate_mask = _proposal_candidate_mask(rows)
    tiers_array = np.asarray(tiers)
    scores_array = np.asarray(scores, dtype=np.float64)
    return_threshold = float(rule["return_threshold"])
    harm_threshold = float(rule["harm_probability_threshold"])
    if not (math.isfinite(return_threshold) and math.isfinite(harm_threshold)):
        raise ValueError("coverage rule thresholds must be finite")
    return_safe = (
        candidate_mask & (expected >= return_threshold)
        if return_safe_mask is None
        else _mask(return_safe_mask, len(rows), "return-safe") & candidate_mask
    )
    harm_safe = (
        candidate_mask & (harm <= harm_threshold)
        if harm_safe_mask is None
        else _mask(harm_safe_mask, len(rows), "harm-safe") & candidate_mask
    )
    if authorized_mask is None:
        authorized = return_safe & harm_safe
        authorization_source = "aggregated_final_rule"
    else:
        authorized = _mask(authorized_mask, len(rows), "authorization")
        authorized &= candidate_mask
        authorization_source = "outer_fold_inner_selected_rule"
        if np.any(authorized & ~return_safe) or np.any(authorized & ~harm_safe):
            raise ValueError("authorized rows are not return/harm safe")
    if actually_changed_mask is None:
        source_changed = np.ones(len(rows), dtype=bool)
        change_definition = "source_manifest_exactly_one_changed_action"
    else:
        source_changed = _mask(
            actually_changed_mask, len(rows), "actually-changed"
        )
        change_definition = "explicit_row_change_field"
    actually_changed = authorized & source_changed
    stages = {
        "eligible_decisions": np.ones(len(rows), dtype=bool),
        "proposal_candidates": candidate_mask,
        "core_proposals": candidate_mask & (tiers_array == "core"),
        "expansion_proposals": candidate_mask & (tiers_array == "expansion"),
        "actually_changed": actually_changed,
        "return_safe": return_safe,
        "harm_safe": harm_safe,
        "authorized": authorized,
        "positive_realized_delta": actually_changed & (target > 1e-8),
        "negative_realized_delta": actually_changed & (target < -1e-8),
        "catastrophic_realized_delta": (
            actually_changed & (target <= CATASTROPHIC_THRESHOLD)
        ),
    }
    stage_report = {
        name: outcome_evidence(mask, target, row_scenes)
        for name, mask in stages.items()
    }
    hierarchy_violations = 0
    expansion_threshold = float(hierarchy["expansion_score_threshold"])
    core_threshold = float(hierarchy["core_score_threshold"])
    upper_threshold = float(hierarchy["score_upper_threshold"])
    for tier, score, valid in zip(tiers, scores_array, candidate_mask):
        if not valid:
            continue
        expected_tier = (
            "core" if core_threshold < score <= upper_threshold
            else "expansion" if expansion_threshold < score <= core_threshold
            else None
        )
        if expected_tier != tier:
            hierarchy_violations += 1
    grouped = {}
    for domain in sorted(set(str(row["dataset"]) for row in rows)):
        for tier in sorted(set(tiers)):
            group = np.asarray([
                str(row["dataset"]) == domain and row.get("tier") == tier
                for row in rows
            ], dtype=bool)
            if group.any():
                group_targets = target[group]
                group_scenes = row_scenes[group]
                grouped[f"{domain}/{tier}"] = {
                    name: outcome_evidence(
                        mask[group], group_targets, group_scenes
                    )
                    for name, mask in stages.items()
                }
    return {
        "scope": "exact_one_switch_supervised_rows",
        "row_count": len(rows),
        "stage_order": list(stages),
        "stages": stage_report,
        "by_domain_and_tier": grouped,
        "authorization_source": authorization_source,
        "actually_changed_definition": change_definition,
        "hierarchy_range_violations": hierarchy_violations,
        "hierarchy": {
            "expansion_score_threshold": expansion_threshold,
            "core_score_threshold": core_threshold,
            "score_upper_threshold": upper_threshold,
        },
        "source_population": source_population or {},
        "selection_used": False,
    }
