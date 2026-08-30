"""Scene-disjoint selection and diagnostics for MF3ZK-DSR v1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import hashlib
import math

import numpy as np
import torch

from revealnav_mf3.distributional_switch import (
    QUANTILES,
    DistributionalSwitchCritic,
    quantile_switch_loss,
)
from revealnav_mf3.nested_selection import (
    CATASTROPHIC_THRESHOLD,
    NestedSelectionError,
    deterministic_scene_folds,
    domain_evidence,
    outcome_evidence,
)


def domain_scene_weights(
    scenes: np.ndarray, datasets: np.ndarray,
) -> np.ndarray:
    """Balance domains, then scenes within domains, then rows within scenes."""

    scenes = np.asarray([str(value) for value in scenes])
    datasets = np.asarray([str(value) for value in datasets])
    if (
        scenes.ndim != 1
        or datasets.ndim != 1
        or len(scenes) == 0
        or len(scenes) != len(datasets)
    ):
        raise ValueError("invalid DSR scene/domain arrays")
    domains = sorted(set(datasets))
    result = np.zeros(len(scenes), dtype=np.float64)
    for domain in domains:
        domain_mask = datasets == domain
        domain_scenes = sorted(set(scenes[domain_mask]))
        if not domain_scenes:
            raise ValueError("empty DSR domain")
        for scene in domain_scenes:
            mask = domain_mask & (scenes == scene)
            result[mask] = (
                len(scenes)
                / (len(domains) * len(domain_scenes) * int(mask.sum()))
            )
    if not np.isfinite(result).all() or not np.all(result > 0):
        raise ValueError("invalid DSR domain-scene weights")
    if not math.isclose(float(result.sum()), float(len(result)), rel_tol=1e-12):
        raise ValueError("DSR weights do not preserve the row-count scale")
    return result


def _validate_rows_and_folds(
    rows: Sequence[dict], outer_folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("DSR proposal audit has no rows")
    target = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    scenes = np.asarray([str(row["scene_id"]) for row in rows])
    datasets = np.asarray([str(row["dataset"]) for row in rows])
    outer_folds = np.asarray(outer_folds, dtype=np.int64)
    if (
        len(outer_folds) != len(rows)
        or not np.isfinite(target).all()
        or len(set(datasets)) == 0
    ):
        raise ValueError("invalid DSR proposal-audit arrays")
    assignment: dict[str, set[int]] = {}
    for scene, fold in zip(scenes, outer_folds, strict=True):
        assignment.setdefault(scene, set()).add(int(fold))
    if any(len(folds) != 1 for folds in assignment.values()):
        raise NestedSelectionError("a shared MP3D scene spans outer folds")
    return target, scenes, datasets


def _target_summary(target: np.ndarray, scenes: np.ndarray) -> dict:
    positive_scene_utility: dict[str, float] = {}
    for scene in sorted(set(scenes)):
        value = target[scenes == scene]
        positive_scene_utility[scene] = float(value[value > 0].sum())
    positive_total = sum(positive_scene_utility.values())
    largest_share = (
        max(positive_scene_utility.values()) / positive_total
        if positive_total > 0 else 0.0
    )
    return {
        "rows": int(len(target)),
        "scenes": int(len(set(scenes))),
        "positive": int((target > 0).sum()),
        "negative": int((target < 0).sum()),
        "ties": int((target == 0).sum()),
        "catastrophic": int((target <= CATASTROPHIC_THRESHOLD).sum()),
        "positive_prevalence": float((target > 0).mean()),
        "catastrophic_prevalence": float(
            (target <= CATASTROPHIC_THRESHOLD).mean()
        ),
        "positive_scene_count": int(sum(
            value > 0 for value in positive_scene_utility.values()
        )),
        "largest_scene_share_of_positive_utility": float(largest_share),
    }


def _fixed_coverage(
    target: np.ndarray, order: np.ndarray, coverage: float,
) -> dict:
    budget = min(len(target), max(1, int(math.ceil(len(target) * coverage))))
    selected = np.asarray(order[:budget], dtype=np.int64)
    values = target[selected]
    return {
        "coverage": float(coverage),
        "budget": int(budget),
        "total_utility": float(values.sum()),
        "mean_utility": float(values.mean()),
        "positive": int((values > 0).sum()),
        "negative": int((values < 0).sum()),
        "catastrophic": int((values <= CATASTROPHIC_THRESHOLD).sum()),
    }


def proposal_support_audit(
    rows: list[dict], outer_folds: np.ndarray,
) -> dict:
    """Target-aware upper-bound audit that cannot select a DSR model."""

    target, scenes, datasets = _validate_rows_and_folds(rows, outer_folds)
    result = {}
    failures = []
    for domain in sorted(set(datasets)):
        mask = datasets == domain
        values = target[mask]
        domain_scenes = scenes[mask]
        domain_folds = np.asarray(outer_folds, dtype=np.int64)[mask]
        domain_rows = [row for row, keep in zip(rows, mask, strict=True) if keep]
        oracle_order = np.argsort(-values, kind="stable")
        margin = np.asarray([
            float(row["decision"]["native_margin"]) for row in domain_rows
        ], dtype=np.float64)
        if not np.isfinite(margin).all():
            raise ValueError("proposal audit found non-finite native margin")
        low_margin_order = np.argsort(margin, kind="stable")
        fixed = {
            str(int(coverage * 100)): _fixed_coverage(
                values, oracle_order, coverage
            )
            for coverage in (0.05, 0.10, 0.20)
        }
        low_margin = {
            str(int(coverage * 100)): _fixed_coverage(
                values, low_margin_order, coverage
            )
            for coverage in (0.05, 0.10, 0.20)
        }
        deciles = []
        for index, selected in enumerate(np.array_split(low_margin_order, 10)):
            selected_values = values[selected]
            deciles.append({
                "decile": index + 1,
                "rows": int(len(selected)),
                "native_margin_mean": (
                    float(margin[selected].mean()) if len(selected) else None
                ),
                "total_utility": float(selected_values.sum()),
                "mean_utility": (
                    float(selected_values.mean()) if len(selected) else None
                ),
            })
        summary = _target_summary(values, domain_scenes)
        required_positive_scenes = max(
            5, int(math.ceil(0.20 * summary["scenes"]))
        )
        domain_failures = []
        if (
            fixed["10"]["total_utility"] <= 0
            and fixed["20"]["total_utility"] <= 0
        ):
            domain_failures.append("oracle_10_and_20_percent_nonpositive")
        if summary["positive_scene_count"] < required_positive_scenes:
            domain_failures.append("positive_switch_scene_support_too_small")
        if domain_failures:
            failures.extend(f"{domain}:{reason}" for reason in domain_failures)
        result[domain] = {
            **summary,
            "required_positive_scene_count": required_positive_scenes,
            "oracle_fixed_coverage": fixed,
            "low_native_margin_fixed_coverage": low_margin,
            "low_native_margin_deciles": deciles,
            "outer_fold_support": [
                {
                    "outer_fold": int(fold),
                    **_target_summary(
                        values[domain_folds == fold],
                        domain_scenes[domain_folds == fold],
                    ),
                }
                for fold in sorted(set(domain_folds))
            ],
            "failure_reasons": domain_failures,
        }
    return {
        "status": "PROPOSAL_SUPPORT_AUDIT_PASS" if not failures
        else "PROPOSAL_SUPPORT_AUDIT_FAIL",
        "selection_use": False,
        "architecture_use": False,
        "threshold_use": False,
        "domains": result,
        "failure_reasons": failures,
    }


def _weighted_standardization(
    matrix: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum())
    mean = (matrix * weights[:, None]).sum(axis=0) / total
    variance = (((matrix - mean) ** 2) * weights[:, None]).sum(axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _state_hash(model: DistributionalSwitchCritic) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.asarray(value.detach().cpu()).tobytes())
    return digest.hexdigest()


def _fit_ensemble(
    matrix: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    *,
    weight_decay: float,
    seeds: Sequence[int],
    learning_rate: float,
    training_steps: int,
) -> tuple[list[DistributionalSwitchCritic], list[str], list[float]]:
    weights = domain_scene_weights(scenes, datasets)
    mean, scale = _weighted_standardization(matrix, weights)
    features = torch.as_tensor(matrix, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
    models = []
    initial_hashes = []
    final_losses = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = DistributionalSwitchCritic()
        model.set_standardization(mean, scale)
        initial_hashes.append(_state_hash(model))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        model.train()
        for _ in range(int(training_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = quantile_switch_loss(
                model(features), target_tensor, weight_tensor
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final_loss = quantile_switch_loss(
                model(features), target_tensor, weight_tensor
            )
        if not torch.isfinite(final_loss):
            raise NestedSelectionError("DSR optimization produced non-finite loss")
        models.append(model)
        final_losses.append(float(final_loss))
    return models, initial_hashes, final_losses


def _predict_ensemble(
    models: Sequence[DistributionalSwitchCritic], matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = torch.as_tensor(matrix, dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for model in models:
            predictions.append(model(features))
    result = []
    for name in ("lower_q20", "median_q50", "upper_q80"):
        members = torch.stack([value[name] for value in predictions], dim=1)
        result.append(torch.median(members, dim=1).values.cpu().numpy().astype(
            np.float64
        ))
    lower, median, upper = result
    if (
        not all(np.isfinite(value).all() for value in result)
        or np.any(lower > median)
        or np.any(median > upper)
    ):
        raise NestedSelectionError("DSR ensemble prediction drift")
    return lower, median, upper


def _pinball_score(
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    losses = []
    for quantile, predicted in zip(
        QUANTILES, (lower, median, upper), strict=True,
    ):
        error = target - predicted
        losses.append(np.maximum(quantile * error, (quantile - 1.0) * error))
    row_loss = np.stack(losses, axis=1).mean(axis=1)
    return float(np.sum(row_loss * weights) / np.sum(weights))


def _fold_domain_zero_coverage(
    mask: np.ndarray, datasets: np.ndarray, folds: np.ndarray,
) -> list[str]:
    failures = []
    for fold in sorted(set(int(value) for value in folds)):
        for domain in sorted(set(datasets[folds == fold])):
            stratum = (folds == fold) & (datasets == domain)
            if stratum.any() and not (mask & stratum).any():
                failures.append(f"fold_{fold}:{domain}:zero_intervention")
    return failures


def _candidate_feasibility(
    mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> tuple[bool, list[str], dict]:
    failures = _fold_domain_zero_coverage(mask, datasets, folds)
    domains = domain_evidence(mask, target, scenes, datasets)
    ungated = domain_evidence(
        np.ones(len(target), dtype=bool), target, scenes, datasets
    )
    for domain, evidence in domains.items():
        if evidence["total_utility"] <= 0:
            failures.append(f"{domain}:nonpositive_utility")
        if evidence["catastrophic_rate"] > ungated[domain]["catastrophic_rate"]:
            failures.append(f"{domain}:catastrophic_rate_above_ungated")
    return not failures, failures, {
        "evidence": outcome_evidence(mask, target, scenes),
        "domains": domains,
        "ungated_domains": ungated,
    }


def _modal_smallest(values: Sequence[float]) -> float:
    counts = Counter(float(value) for value in values)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def nested_distributional_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    outer_folds: np.ndarray,
    config: dict,
) -> dict:
    """Fit DSR with inner-loss WD selection and unbiased outer predictions."""

    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    datasets = np.asarray([str(value) for value in datasets])
    outer_folds = np.asarray(outer_folds, dtype=np.int64)
    if (
        matrix.ndim != 2
        or matrix.shape != (len(target), 28)
        or not np.isfinite(matrix).all()
        or not np.isfinite(target).all()
        or not len(target) == len(scenes) == len(datasets) == len(outer_folds)
    ):
        raise ValueError("invalid DSR nested-fit arrays")
    outer_count = int(config["outer_folds"])
    inner_count = int(config["inner_folds"])
    if set(outer_folds) != set(range(outer_count)):
        raise NestedSelectionError("DSR outer folds are incomplete")
    scene_assignment: dict[str, set[int]] = {}
    for scene, fold in zip(scenes, outer_folds, strict=True):
        scene_assignment.setdefault(scene, set()).add(int(fold))
    if any(len(folds) != 1 for folds in scene_assignment.values()):
        raise NestedSelectionError("DSR outer split divided an MP3D scene")
    weight_decay_grid = tuple(float(value) for value in config["weight_decay_grid"])
    seeds = tuple(int(value) for value in config["seeds"])
    learning_rate = float(config["learning_rate"])
    training_steps = int(config["training_steps"])
    fold_salt = str(config["inner_fold_salt"])
    if (
        weight_decay_grid != (0.0001, 0.001, 0.01)
        or len(seeds) != 3
        or inner_count != 4
        or outer_count != 5
        or learning_rate <= 0
        or training_steps < 1
    ):
        raise ValueError("DSR v1 training configuration drift")

    outer_lower = np.full(len(target), np.nan, dtype=np.float64)
    outer_median = np.full(len(target), np.nan, dtype=np.float64)
    outer_upper = np.full(len(target), np.nan, dtype=np.float64)
    outer_records = []
    selected_weight_decay = []
    failures = []
    for outer_fold in range(outer_count):
        evaluate = outer_folds == outer_fold
        fit = ~evaluate
        fit_scenes = sorted(set(scenes[fit]))
        evaluation_scenes = sorted(set(scenes[evaluate]))
        if not fit.any() or not evaluate.any() or set(fit_scenes) & set(evaluation_scenes):
            raise NestedSelectionError(f"invalid DSR outer fold {outer_fold}")
        _, inner_mapping = deterministic_scene_folds(
            fit_scenes, inner_count,
            salt=f"{fold_salt}:outer:{outer_fold}",
        )
        inner_folds = np.asarray([
            inner_mapping[scene] for scene in scenes[fit]
        ], dtype=np.int64)
        trials = []
        candidate_initialization: dict[float, list[list[str]]] = {}
        for weight_decay in weight_decay_grid:
            inner_lower = np.full(int(fit.sum()), np.nan, dtype=np.float64)
            inner_median = np.full(int(fit.sum()), np.nan, dtype=np.float64)
            inner_upper = np.full(int(fit.sum()), np.nan, dtype=np.float64)
            initial_by_fold = []
            fold_records = []
            for inner_fold in range(inner_count):
                inner_evaluate = inner_folds == inner_fold
                inner_fit = ~inner_evaluate
                inner_fit_scenes = set(scenes[fit][inner_fit])
                inner_eval_scenes = set(scenes[fit][inner_evaluate])
                if (
                    not inner_fit.any()
                    or not inner_evaluate.any()
                    or inner_fit_scenes & inner_eval_scenes
                ):
                    raise NestedSelectionError(
                        f"invalid DSR inner fold {outer_fold}/{inner_fold}"
                    )
                models, initial_hashes, final_losses = _fit_ensemble(
                    matrix[fit][inner_fit], target[fit][inner_fit],
                    scenes[fit][inner_fit], datasets[fit][inner_fit],
                    weight_decay=weight_decay, seeds=seeds,
                    learning_rate=learning_rate, training_steps=training_steps,
                )
                predictions = _predict_ensemble(models, matrix[fit][inner_evaluate])
                inner_lower[inner_evaluate] = predictions[0]
                inner_median[inner_evaluate] = predictions[1]
                inner_upper[inner_evaluate] = predictions[2]
                initial_by_fold.append(initial_hashes)
                fold_records.append({
                    "inner_fold": inner_fold,
                    "fit_scenes": sorted(inner_fit_scenes),
                    "evaluation_scenes": sorted(inner_eval_scenes),
                    "scene_overlap": [],
                    "initialization_hashes": initial_hashes,
                    "final_training_losses": final_losses,
                })
            if not all(np.isfinite(value).all() for value in (
                inner_lower, inner_median, inner_upper,
            )):
                raise NestedSelectionError("DSR inner OOF prediction is incomplete")
            inner_mask = inner_lower > 0.0
            feasible, reasons, evidence = _candidate_feasibility(
                inner_mask, target[fit], scenes[fit], datasets[fit], inner_folds
            )
            loss = _pinball_score(
                inner_lower, inner_median, inner_upper, target[fit],
                domain_scene_weights(scenes[fit], datasets[fit]),
            )
            trials.append({
                "weight_decay": weight_decay,
                "inner_oof_quantile_loss": loss,
                "feasible": feasible,
                "failure_reasons": reasons,
                "inner_cv": fold_records,
                **evidence,
            })
            candidate_initialization[weight_decay] = initial_by_fold
        reference_hashes = candidate_initialization[weight_decay_grid[0]]
        if any(
            candidate_initialization[value] != reference_hashes
            for value in weight_decay_grid[1:]
        ):
            raise NestedSelectionError(
                "DSR weight-decay candidates did not share initialization"
            )
        feasible_trials = [trial for trial in trials if trial["feasible"]]
        if not feasible_trials:
            failures.append(f"outer_fold_{outer_fold}:no_feasible_inner_candidate")
            outer_records.append({
                "outer_fold": outer_fold,
                "fit_scenes": fit_scenes,
                "evaluation_scenes": evaluation_scenes,
                "scene_overlap": [],
                "inner_scene_assignment": inner_mapping,
                "common_random_numbers_verified": True,
                "trials": trials,
                "selected_weight_decay": None,
            })
            break
        selected = min(
            feasible_trials,
            key=lambda trial: (
                trial["inner_oof_quantile_loss"], trial["weight_decay"]
            ),
        )
        models, _, training_losses = _fit_ensemble(
            matrix[fit], target[fit], scenes[fit], datasets[fit],
            weight_decay=float(selected["weight_decay"]), seeds=seeds,
            learning_rate=learning_rate, training_steps=training_steps,
        )
        predictions = _predict_ensemble(models, matrix[evaluate])
        outer_lower[evaluate], outer_median[evaluate], outer_upper[evaluate] = predictions
        selected_weight_decay.append(float(selected["weight_decay"]))
        outer_mask = predictions[0] > 0.0
        outer_records.append({
            "outer_fold": outer_fold,
            "fit_scenes": fit_scenes,
            "evaluation_scenes": evaluation_scenes,
            "scene_overlap": [],
            "inner_scene_assignment": inner_mapping,
            "common_random_numbers_verified": True,
            "trials": trials,
            "selected_weight_decay": float(selected["weight_decay"]),
            "outer_training_losses": training_losses,
            "outer_evidence": outcome_evidence(
                outer_mask, target[evaluate], scenes[evaluate]
            ),
            "outer_domains": domain_evidence(
                outer_mask, target[evaluate], scenes[evaluate], datasets[evaluate]
            ),
        })

    if failures:
        return {
            "status": "NESTED_DSR_FAIL",
            "failure_reasons": failures,
            "outer_folds": outer_records,
            "final_models": [],
        }
    if not all(np.isfinite(value).all() for value in (
        outer_lower, outer_median, outer_upper,
    )):
        raise NestedSelectionError("DSR outer OOF prediction is incomplete")
    outer_mask = outer_lower > 0.0
    scientific_failures = _fold_domain_zero_coverage(
        outer_mask, datasets, outer_folds
    )
    evidence = outcome_evidence(outer_mask, target, scenes)
    domains = domain_evidence(outer_mask, target, scenes, datasets)
    ungated_domains = domain_evidence(
        np.ones(len(target), dtype=bool), target, scenes, datasets
    )
    for domain, value in domains.items():
        if value["total_utility"] <= 0:
            scientific_failures.append(f"{domain}:nonpositive_outer_utility")
        if value["minimum_leave_one_selected_scene_out_total"] <= 0:
            scientific_failures.append(f"{domain}:nonpositive_leave_one_scene")
        if value["catastrophic_rate"] > ungated_domains[domain]["catastrophic_rate"]:
            scientific_failures.append(f"{domain}:catastrophic_rate_above_ungated")
    modal_weight_decay = _modal_smallest(selected_weight_decay)
    final_models = []
    final_losses = []
    if not scientific_failures:
        final_models, _, final_losses = _fit_ensemble(
            matrix, target, scenes, datasets,
            weight_decay=modal_weight_decay, seeds=seeds,
            learning_rate=learning_rate, training_steps=training_steps,
        )
    return {
        "status": "NESTED_DSR_PASS" if not scientific_failures
        else "NESTED_DSR_FAIL",
        "failure_reasons": scientific_failures,
        "decision_rule": "lower_q20_utility > 0",
        "selected_weight_decay": modal_weight_decay,
        "outer_fold_weight_decay": selected_weight_decay,
        "outer_folds": outer_records,
        "outer_oof": {
            "lower_q20": outer_lower,
            "median_q50": outer_median,
            "upper_q80": outer_upper,
            "authorized_mask": outer_mask,
            "evidence": evidence,
            "domains": domains,
            "ungated_domains": ungated_domains,
        },
        "final_training_losses": final_losses,
        "final_models": final_models,
    }


def _proposal_fields(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    margin = np.asarray([
        float(row["decision"]["native_margin"]) for row in rows
    ], dtype=np.float64)
    score = np.asarray([
        float(row["decision"]["policy_risk_adjusted_score"]) for row in rows
    ], dtype=np.float64)
    if not np.isfinite(margin).all() or not np.isfinite(score).all():
        raise ValueError("non-finite proposal-side baseline field")
    return margin, score


def _stable_random_key(seed: int, row: dict) -> str:
    identity = (
        f"{row['dataset']}:{row['scene_id']}:{row['episode_id']}:"
        f"{row['decision']['step']}"
    )
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _baseline_evidence(
    mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
) -> dict:
    return {
        "overall": outcome_evidence(mask, target, scenes),
        "domains": domain_evidence(mask, target, scenes, datasets),
    }


def stratified_equal_budget_baselines(
    rows: list[dict],
    target: np.ndarray,
    gate_mask: np.ndarray,
    outer_folds: np.ndarray,
    *,
    seed: int = 20260830,
) -> dict:
    """Target-blind baselines at global and fold/domain-matched budgets."""

    target, scenes, datasets = _validate_rows_and_folds(rows, outer_folds)
    gate_mask = np.asarray(gate_mask, dtype=bool)
    outer_folds = np.asarray(outer_folds, dtype=np.int64)
    if gate_mask.shape != (len(rows),):
        raise ValueError("invalid DSR gate mask")
    margin, score = _proposal_fields(rows)
    random_key = np.asarray([_stable_random_key(seed, row) for row in rows])

    def choose(indices: np.ndarray, budget: int, mode: str) -> np.ndarray:
        if mode == "low_native_margin":
            order = indices[np.argsort(margin[indices], kind="stable")]
        elif mode == "high_proposal_score":
            order = indices[np.argsort(-score[indices], kind="stable")]
        elif mode == "fixed_seed_random":
            order = indices[np.argsort(random_key[indices], kind="stable")]
        else:
            raise ValueError("unknown DSR baseline")
        return order[:budget]

    modes = ("low_native_margin", "high_proposal_score", "fixed_seed_random")
    global_masks = {}
    budget = int(gate_mask.sum())
    all_indices = np.arange(len(rows), dtype=np.int64)
    for mode in modes:
        selected = np.zeros(len(rows), dtype=bool)
        selected[choose(all_indices, budget, mode)] = True
        global_masks[mode] = selected

    matched_masks = {mode: np.zeros(len(rows), dtype=bool) for mode in modes}
    strata = []
    for fold in sorted(set(outer_folds)):
        for domain in sorted(set(datasets[outer_folds == fold])):
            indices = np.flatnonzero(
                (outer_folds == fold) & (datasets == domain)
            )
            stratum_budget = int(gate_mask[indices].sum())
            strata.append({
                "outer_fold": int(fold), "dataset": domain,
                "eligible": int(len(indices)), "budget": stratum_budget,
            })
            for mode in modes:
                matched_masks[mode][choose(indices, stratum_budget, mode)] = True
    if any(int(mask.sum()) != budget for mask in matched_masks.values()):
        raise RuntimeError("fold/domain matched baseline budget drift")
    return {
        "budget": budget,
        "global": {
            mode: _baseline_evidence(
                mask, target, scenes, datasets
            ) for mode, mask in global_masks.items()
        },
        "fold_domain_matched": {
            "strata": strata,
            "baselines": {
                mode: _baseline_evidence(
                    mask, target, scenes, datasets
                ) for mode, mask in matched_masks.items()
            },
        },
        "internal_masks": {
            "global": global_masks,
            "fold_domain_matched": matched_masks,
        },
    }


def risk_coverage_diagnostic(
    score: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    *,
    coverages: Sequence[float] = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0),
) -> list[dict]:
    """Fixed q20 ranking diagnostic; never an operating-point selector."""

    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    if len(score) == 0 or len(score) != len(target) or not np.isfinite(score).all():
        raise ValueError("invalid DSR risk-coverage inputs")
    order = np.argsort(-score, kind="stable")
    result = []
    for coverage in coverages:
        budget = min(len(score), max(1, int(math.ceil(len(score) * coverage))))
        mask = np.zeros(len(score), dtype=bool)
        mask[order[:budget]] = True
        result.append({
            "requested_coverage": float(coverage),
            "score_name": "lower_q20_utility",
            "selection_used": False,
            **outcome_evidence(mask, target, scenes),
        })
    return result


def scene_cluster_bootstrap(
    gate_mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    *,
    comparator_mask: np.ndarray | None = None,
    replicates: int = 10_000,
    seed: int = 20260830,
) -> dict:
    """Bootstrap raw MP3D scenes, preserving shared cross-benchmark clusters."""

    gate_mask = np.asarray(gate_mask, dtype=bool)
    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    datasets = np.asarray([str(value) for value in datasets])
    if (
        len(target) == 0
        or gate_mask.shape != target.shape
        or scenes.shape != target.shape
        or datasets.shape != target.shape
        or not np.isfinite(target).all()
        or replicates < 1
    ):
        raise ValueError("invalid DSR scene-bootstrap input")
    if comparator_mask is not None:
        comparator_mask = np.asarray(comparator_mask, dtype=bool)
        if comparator_mask.shape != target.shape:
            raise ValueError("invalid DSR bootstrap comparator")
    unique_scenes = np.unique(scenes)
    rng = np.random.default_rng(int(seed))
    gate_totals = np.empty(replicates, dtype=np.float64)
    differences = (
        np.empty(replicates, dtype=np.float64)
        if comparator_mask is not None else None
    )
    domain_totals = {
        domain: np.empty(replicates, dtype=np.float64)
        for domain in sorted(set(datasets))
    }
    scene_indices = {
        scene: np.flatnonzero(scenes == scene) for scene in unique_scenes
    }
    for replicate in range(replicates):
        sampled = rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
        indices = np.concatenate([scene_indices[scene] for scene in sampled])
        gate_total = float(target[indices][gate_mask[indices]].sum())
        gate_totals[replicate] = gate_total
        if differences is not None:
            comparator_total = float(
                target[indices][comparator_mask[indices]].sum()
            )
            differences[replicate] = gate_total - comparator_total
        for domain in domain_totals:
            domain_indices = indices[datasets[indices] == domain]
            domain_totals[domain][replicate] = float(
                target[domain_indices][gate_mask[domain_indices]].sum()
            )

    def interval(values: np.ndarray) -> dict:
        return {
            "mean": float(values.mean()),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "probability_above_zero": float((values > 0).mean()),
        }

    return {
        "cluster": "raw_mp3d_scene_id_shared_across_benchmarks",
        "replicates": int(replicates),
        "seed": int(seed),
        "gate_total_utility": interval(gate_totals),
        "domains": {
            domain: interval(values) for domain, values in domain_totals.items()
        },
        "gate_minus_comparator_total_utility": (
            interval(differences) if differences is not None else None
        ),
    }
