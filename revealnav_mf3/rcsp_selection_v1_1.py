"""Scene-disjoint constrained selection for MF3ZL-RCSP v1.1.

This is a versioned copy of the sealed v1 selection procedure.  The only
algorithmic difference is the v1.1 model backend, which retains events whose
native and runner-up embeddings are identical by masking their relative
semantic direction to zero.  The v1 module remains untouched and auditable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import hashlib
import math

import numpy as np
import torch

from revealnav_mf3.dsr_selection import (
    scene_cluster_bootstrap,
    stratified_equal_budget_baselines,
)
from revealnav_mf3.nested_selection import (
    CATASTROPHIC_THRESHOLD,
    NestedSelectionError,
    deterministic_scene_folds,
    domain_evidence,
    outcome_evidence,
)
from revealnav_mf3.rcsp_v1_1 import (
    ENGINEERED_FEATURE_DIM,
    POLICY_FEATURE_NAMES,
    EngineeredRCSPControl,
    RelativeSemanticSwitchPolicy,
    catastrophic_constraint,
    projected_dual_update,
    utility_weighted_preference_loss,
)


def domain_scene_episode_weights(
    datasets: np.ndarray,
    scenes: np.ndarray,
    episodes: np.ndarray,
) -> np.ndarray:
    """Balance domain, scene, episode, then events within each episode."""

    datasets = np.asarray([str(value) for value in datasets])
    scenes = np.asarray([str(value) for value in scenes])
    episodes = np.asarray([str(value) for value in episodes])
    if (
        datasets.ndim != 1
        or scenes.ndim != 1
        or episodes.ndim != 1
        or len(datasets) == 0
        or not len(datasets) == len(scenes) == len(episodes)
    ):
        raise ValueError("invalid RCSP weighting arrays")
    result = np.zeros(len(datasets), dtype=np.float64)
    domains = sorted(set(datasets))
    for domain in domains:
        domain_mask = datasets == domain
        domain_scenes = sorted(set(scenes[domain_mask]))
        if not domain_scenes:
            raise ValueError("empty RCSP domain")
        for scene in domain_scenes:
            scene_mask = domain_mask & (scenes == scene)
            scene_episodes = sorted(set(episodes[scene_mask]))
            if not scene_episodes:
                raise ValueError("empty RCSP scene")
            for episode in scene_episodes:
                mask = scene_mask & (episodes == episode)
                result[mask] = len(datasets) / (
                    len(domains)
                    * len(domain_scenes)
                    * len(scene_episodes)
                    * int(mask.sum())
                )
    if (
        not np.isfinite(result).all()
        or not np.all(result > 0)
        or not math.isclose(float(result.sum()), float(len(result)), rel_tol=1e-12)
    ):
        raise ValueError("invalid RCSP domain-scene-episode weights")
    return result


def _weighted_policy_standardization(
    policy: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum())
    mean = np.sum(policy * weights[:, None], axis=0) / total
    variance = np.sum((policy - mean) ** 2 * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _state_hash(model: RelativeSemanticSwitchPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _tensor_inputs(inputs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value, dtype=torch.float32)
        for name, value in inputs.items()
    }


def _validate_inputs(
    inputs: dict[str, np.ndarray], rows: int, representation: str,
) -> None:
    if representation == "semantic":
        expected = {
            "policy": (rows, len(POLICY_FEATURE_NAMES)),
            "instruction": (rows, 768),
            "history": (rows, 768),
            "native": (rows, 768),
            "runner": (rows, 768),
        }
    elif representation == "engineered_28d":
        expected = {"engineered": (rows, ENGINEERED_FEATURE_DIM)}
    else:
        raise ValueError("unknown RCSP representation")
    if set(inputs) != set(expected):
        raise ValueError("RCSP input field drift")
    for name, shape in expected.items():
        value = np.asarray(inputs[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid RCSP {name} array")


def fit_primal_dual_policy(
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    episodes: np.ndarray,
    *,
    weight_decay: float,
    seeds: Sequence[int],
    learning_rate: float,
    dual_learning_rate: float,
    training_steps: int,
    risk_constrained: bool = True,
    representation: str = "semantic",
) -> tuple[list[torch.nn.Module], list[str], list[dict]]:
    """Fit fixed-architecture full-batch RCSP ensembles."""

    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    datasets = np.asarray([str(value) for value in datasets])
    episodes = np.asarray([str(value) for value in episodes])
    _validate_inputs(inputs, len(target), representation)
    if (
        len(target) == 0
        or not np.isfinite(target).all()
        or not len(target) == len(scenes) == len(datasets) == len(episodes)
        or float(weight_decay) not in (0.0001, 0.001, 0.01)
        or len(tuple(seeds)) != 3
        or learning_rate <= 0
        or dual_learning_rate <= 0
        or training_steps < 1
    ):
        raise ValueError("invalid MF3ZL-RCSP v1 training configuration")
    weights = domain_scene_episode_weights(datasets, scenes, episodes)
    standardized_field = "policy" if representation == "semantic" else "engineered"
    mean, scale = _weighted_policy_standardization(
        inputs[standardized_field], weights
    )
    tensors = _tensor_inputs(inputs)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
    catastrophic_tensor = (target_tensor <= CATASTROPHIC_THRESHOLD).float()
    domain_indices = {
        domain: torch.as_tensor(np.flatnonzero(datasets == domain), dtype=torch.long)
        for domain in sorted(set(datasets))
    }
    ungated_rates = {
        domain: float(np.sum(
            weights[datasets == domain]
            * (target[datasets == domain] <= CATASTROPHIC_THRESHOLD)
        ) / np.sum(weights[datasets == domain]))
        for domain in domain_indices
    }
    models = []
    initial_hashes = []
    diagnostics = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        if representation == "semantic":
            model = RelativeSemanticSwitchPolicy(len(POLICY_FEATURE_NAMES))
            model.set_policy_standardization(mean, scale)
        else:
            model = EngineeredRCSPControl()
            model.set_standardization(mean, scale)
        initial_hashes.append(_state_hash(model))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        duals = {
            domain: torch.zeros((), dtype=torch.float32)
            for domain in domain_indices
        }
        model.train()
        for _ in range(int(training_steps)):
            optimizer.zero_grad(set_to_none=True)
            logits = (
                model(
                    tensors["policy"], tensors["instruction"], tensors["history"],
                    tensors["native"], tensors["runner"],
                )
                if representation == "semantic"
                else model(tensors["engineered"])
            )
            preference = utility_weighted_preference_loss(
                logits, target_tensor, weight_tensor
            )
            constraints = {}
            objective = preference
            for domain, indices in domain_indices.items():
                constraint = catastrophic_constraint(
                    logits[indices], catastrophic_tensor[indices],
                    weight_tensor[indices], ungated_rates[domain],
                )
                constraints[domain] = constraint
                if risk_constrained:
                    objective = objective + duals[domain] * constraint
            penalty = torch.zeros((), dtype=torch.float32)
            for parameter in model.parameters():
                penalty = penalty + torch.sum(parameter ** 2)
            objective = objective + float(weight_decay) * penalty
            if not torch.isfinite(objective):
                raise NestedSelectionError("RCSP optimization became non-finite")
            objective.backward()
            optimizer.step()
            if risk_constrained:
                duals = {
                    domain: projected_dual_update(
                        dual, constraints[domain], dual_learning_rate
                    )
                    for domain, dual in duals.items()
                }
        model.eval()
        with torch.no_grad():
            logits = (
                model(
                    tensors["policy"], tensors["instruction"], tensors["history"],
                    tensors["native"], tensors["runner"],
                )
                if representation == "semantic"
                else model(tensors["engineered"])
            )
            loss = utility_weighted_preference_loss(
                logits, target_tensor, weight_tensor
            )
            constraints = {
                domain: float(catastrophic_constraint(
                    logits[indices], catastrophic_tensor[indices],
                    weight_tensor[indices], ungated_rates[domain],
                ))
                for domain, indices in domain_indices.items()
            }
        models.append(model)
        diagnostics.append({
            "seed": int(seed),
            "preference_loss": float(loss),
            "dual_variables": {
                domain: float(value) for domain, value in duals.items()
            },
            "soft_constraint_excess": constraints,
            "ungated_weighted_catastrophic_rate": ungated_rates,
        })
    return models, initial_hashes, diagnostics


def _predict(
    models: Sequence[torch.nn.Module],
    inputs: dict[str, np.ndarray],
    representation: str = "semantic",
) -> np.ndarray:
    rows = len(next(iter(inputs.values())))
    _validate_inputs(inputs, rows, representation)
    tensors = _tensor_inputs(inputs)
    with torch.no_grad():
        members = torch.stack([
            (
                model(
                    tensors["policy"], tensors["instruction"], tensors["history"],
                    tensors["native"], tensors["runner"],
                )
                if representation == "semantic"
                else model(tensors["engineered"])
            )
            for model in models
        ], dim=1)
    result = torch.median(members, dim=1).values.cpu().numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise NestedSelectionError("RCSP prediction is non-finite")
    return result


def _subset_inputs(
    inputs: dict[str, np.ndarray], mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask] for name, value in inputs.items()}


def _preference_score(
    logits: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    labels = target > 0
    loss = np.where(labels, np.logaddexp(0.0, -logits), np.logaddexp(0.0, logits))
    magnitude = weights * np.abs(target)
    if not np.isfinite(loss).all() or float(magnitude.sum()) <= 0:
        raise NestedSelectionError("invalid RCSP preference score")
    return float(np.sum(loss * magnitude) / np.sum(magnitude))


def _zero_coverage_failures(
    mask: np.ndarray, datasets: np.ndarray, folds: np.ndarray,
) -> list[str]:
    failures = []
    for fold in sorted(set(int(value) for value in folds)):
        for domain in sorted(set(datasets[folds == fold])):
            stratum = (folds == fold) & (datasets == domain)
            if stratum.any() and not (mask & stratum).any():
                failures.append(f"fold_{fold}:{domain}:zero_intervention")
    return failures


def _scientific_evidence(
    mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> tuple[list[str], dict]:
    failures = _zero_coverage_failures(mask, datasets, folds)
    domains = domain_evidence(mask, target, scenes, datasets)
    ungated = domain_evidence(
        np.ones(len(target), dtype=bool), target, scenes, datasets
    )
    for domain, value in domains.items():
        if value["total_utility"] <= 0:
            failures.append(f"{domain}:nonpositive_utility")
        if value["minimum_leave_one_selected_scene_out_total"] <= 0:
            failures.append(f"{domain}:nonpositive_leave_one_selected_scene")
        if value["catastrophic_rate"] > ungated[domain]["catastrophic_rate"]:
            failures.append(f"{domain}:catastrophic_rate_above_ungated")
    return failures, {
        "overall": outcome_evidence(mask, target, scenes),
        "domains": domains,
        "ungated_domains": ungated,
    }


def _baseline_failures(
    gate_evidence: dict, baselines: dict,
) -> list[str]:
    failures = []
    compared = baselines["fold_domain_matched"]["baselines"]
    for baseline_name in ("low_native_margin", "high_proposal_score"):
        baseline = compared[baseline_name]
        if gate_evidence["overall"]["total_utility"] <= baseline["overall"]["total_utility"]:
            failures.append(f"utility_not_above_{baseline_name}")
        if gate_evidence["overall"]["catastrophic_rate"] > baseline["overall"]["catastrophic_rate"]:
            failures.append(f"catastrophic_rate_above_{baseline_name}")
        for domain, value in gate_evidence["domains"].items():
            control = baseline["domains"][domain]
            if value["total_utility"] <= control["total_utility"]:
                failures.append(f"{domain}:utility_not_above_{baseline_name}")
            if value["catastrophic_rate"] > control["catastrophic_rate"]:
                failures.append(f"{domain}:catastrophic_rate_above_{baseline_name}")
    return failures


def _modal_smallest(values: Sequence[float]) -> float:
    counts = Counter(float(value) for value in values)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def nested_rcsp_fit(
    rows: list[dict],
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    episodes: np.ndarray,
    outer_folds: np.ndarray,
    config: dict,
    *,
    risk_constrained: bool = True,
    representation: str = "semantic",
) -> dict:
    """Nested whole-scene RCSP fitting with fixed zero decision boundary."""

    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    datasets = np.asarray([str(value) for value in datasets])
    episodes = np.asarray([str(value) for value in episodes])
    outer_folds = np.asarray(outer_folds, dtype=np.int64)
    _validate_inputs(inputs, len(target), representation)
    if not (
        len(rows) == len(target) == len(scenes) == len(datasets)
        == len(episodes) == len(outer_folds)
        and np.isfinite(target).all()
    ):
        raise ValueError("invalid RCSP nested-fit arrays")
    outer_count = int(config["outer_folds"])
    inner_count = int(config["inner_folds"])
    weight_decay_grid = tuple(float(value) for value in config["weight_decay_grid"])
    seeds = tuple(int(value) for value in config["seeds"])
    if (
        outer_count != 5
        or inner_count != 4
        or weight_decay_grid != (0.0001, 0.001, 0.01)
        or len(seeds) != 3
        or set(outer_folds) != set(range(outer_count))
    ):
        raise ValueError("MF3ZL-RCSP nested configuration drift")
    scene_assignment: dict[str, set[int]] = {}
    for scene, fold in zip(scenes, outer_folds, strict=True):
        scene_assignment.setdefault(scene, set()).add(int(fold))
    if any(len(value) != 1 for value in scene_assignment.values()):
        raise NestedSelectionError("RCSP outer split divided an MP3D scene")

    fit_kwargs = {
        "seeds": seeds,
        "learning_rate": float(config["learning_rate"]),
        "dual_learning_rate": float(config["dual_learning_rate"]),
        "training_steps": int(config["training_steps"]),
        "risk_constrained": bool(risk_constrained),
        "representation": representation,
    }
    outer_logits = np.full(len(target), np.nan, dtype=np.float64)
    outer_records = []
    selected_wd = []
    failures = []
    for outer_fold in range(outer_count):
        evaluate = outer_folds == outer_fold
        fit = ~evaluate
        fit_scenes = sorted(set(scenes[fit]))
        evaluation_scenes = sorted(set(scenes[evaluate]))
        if not fit.any() or not evaluate.any() or set(fit_scenes) & set(evaluation_scenes):
            raise NestedSelectionError(f"invalid RCSP outer fold {outer_fold}")
        _, mapping = deterministic_scene_folds(
            fit_scenes, inner_count,
            salt=f"{config['inner_fold_salt']}:outer:{outer_fold}",
        )
        inner_folds = np.asarray([mapping[scene] for scene in scenes[fit]])
        trials = []
        initialization = {}
        for weight_decay in weight_decay_grid:
            inner_logits = np.full(int(fit.sum()), np.nan, dtype=np.float64)
            fold_records = []
            hashes_by_fold = []
            for inner_fold in range(inner_count):
                inner_evaluate = inner_folds == inner_fold
                inner_fit = ~inner_evaluate
                train_scenes = set(scenes[fit][inner_fit])
                held_scenes = set(scenes[fit][inner_evaluate])
                if not inner_fit.any() or not inner_evaluate.any() or train_scenes & held_scenes:
                    raise NestedSelectionError(
                        f"invalid RCSP inner fold {outer_fold}/{inner_fold}"
                    )
                models, hashes, diagnostics = fit_primal_dual_policy(
                    _subset_inputs(_subset_inputs(inputs, fit), inner_fit),
                    target[fit][inner_fit], scenes[fit][inner_fit],
                    datasets[fit][inner_fit], episodes[fit][inner_fit],
                    weight_decay=weight_decay, **fit_kwargs,
                )
                inner_logits[inner_evaluate] = _predict(
                    models,
                    _subset_inputs(_subset_inputs(inputs, fit), inner_evaluate),
                    representation,
                )
                hashes_by_fold.append(hashes)
                fold_records.append({
                    "inner_fold": inner_fold,
                    "fit_scenes": sorted(train_scenes),
                    "evaluation_scenes": sorted(held_scenes),
                    "scene_overlap": [],
                    "initialization_hashes": hashes,
                    "training_diagnostics": diagnostics,
                })
            if not np.isfinite(inner_logits).all():
                raise NestedSelectionError("RCSP inner OOF logits incomplete")
            inner_mask = inner_logits > 0.0
            scientific_failures, evidence = _scientific_evidence(
                inner_mask, target[fit], scenes[fit], datasets[fit], inner_folds
            )
            inner_rows = [row for row, keep in zip(rows, fit, strict=True) if keep]
            baselines = rcsp_equal_budget_baselines(
                inner_rows, target[fit], inner_mask, inner_folds
            )
            baseline_failures = _baseline_failures(evidence, baselines)
            candidate_failures = scientific_failures + baseline_failures
            score = _preference_score(
                inner_logits, target[fit],
                domain_scene_episode_weights(
                    datasets[fit], scenes[fit], episodes[fit]
                ),
            )
            trials.append({
                "weight_decay": weight_decay,
                "inner_oof_preference_loss": score,
                "feasible": not candidate_failures,
                "failure_reasons": candidate_failures,
                "inner_cv": fold_records,
                "evidence": evidence,
                "equal_budget": baselines,
            })
            initialization[weight_decay] = hashes_by_fold
        reference = initialization[weight_decay_grid[0]]
        if any(initialization[value] != reference for value in weight_decay_grid[1:]):
            raise NestedSelectionError("RCSP WD candidates did not share initialization")
        feasible = [trial for trial in trials if trial["feasible"]]
        if not feasible:
            failures.append(f"outer_fold_{outer_fold}:no_feasible_inner_candidate")
            outer_records.append({
                "outer_fold": outer_fold,
                "fit_scenes": fit_scenes,
                "evaluation_scenes": evaluation_scenes,
                "inner_scene_assignment": mapping,
                "scene_overlap": [],
                "common_random_numbers_verified": True,
                "trials": trials,
                "selected_weight_decay": None,
            })
            break
        selected = min(
            feasible,
            key=lambda value: (
                value["inner_oof_preference_loss"], value["weight_decay"]
            ),
        )
        models, _, diagnostics = fit_primal_dual_policy(
            _subset_inputs(inputs, fit), target[fit], scenes[fit], datasets[fit],
            episodes[fit], weight_decay=float(selected["weight_decay"]),
            **fit_kwargs,
        )
        predicted = _predict(
            models, _subset_inputs(inputs, evaluate), representation
        )
        outer_logits[evaluate] = predicted
        selected_wd.append(float(selected["weight_decay"]))
        outer_records.append({
            "outer_fold": outer_fold,
            "fit_scenes": fit_scenes,
            "evaluation_scenes": evaluation_scenes,
            "inner_scene_assignment": mapping,
            "scene_overlap": [],
            "common_random_numbers_verified": True,
            "trials": trials,
            "selected_weight_decay": float(selected["weight_decay"]),
            "outer_training_diagnostics": diagnostics,
            "outer_evidence": _scientific_evidence(
                predicted > 0.0, target[evaluate], scenes[evaluate],
                datasets[evaluate], outer_folds[evaluate],
            )[1],
        })

    if failures:
        return {
            "status": "NESTED_RCSP_FAIL",
            "failure_reasons": failures,
            "outer_folds": outer_records,
            "final_models": [],
        }
    if not np.isfinite(outer_logits).all():
        raise NestedSelectionError("RCSP outer OOF logits incomplete")
    gate_mask = outer_logits > 0.0
    scientific_failures, evidence = _scientific_evidence(
        gate_mask, target, scenes, datasets, outer_folds
    )
    baselines = rcsp_equal_budget_baselines(rows, target, gate_mask, outer_folds)
    baseline_failures = _baseline_failures(evidence, baselines)
    failures = scientific_failures + baseline_failures
    modal_wd = _modal_smallest(selected_wd)
    final_models = []
    final_diagnostics = []
    if not failures:
        final_models, _, final_diagnostics = fit_primal_dual_policy(
            inputs, target, scenes, datasets, episodes,
            weight_decay=modal_wd, **fit_kwargs,
        )
    return {
        "status": "NESTED_RCSP_PASS" if not failures else "NESTED_RCSP_FAIL",
        "failure_reasons": failures,
        "decision_rule": "switch_logit > 0",
        "risk_constrained": bool(risk_constrained),
        "representation": representation,
        "selected_weight_decay": modal_wd,
        "outer_fold_weight_decay": selected_wd,
        "outer_folds": outer_records,
        "outer_oof": {
            "switch_logit": outer_logits,
            "authorized_mask": gate_mask,
            **evidence,
        },
        "equal_budget": baselines,
        "final_training_diagnostics": final_diagnostics,
        "final_models": final_models,
    }


def rcsp_equal_budget_baselines(
    rows: list[dict],
    target: np.ndarray,
    gate_mask: np.ndarray,
    outer_folds: np.ndarray,
) -> dict:
    """Target-blind global and fold/domain-matched RCSP controls."""

    target = np.asarray(target, dtype=np.float64)
    if len(rows) != len(target):
        raise ValueError("RCSP baseline row/target cardinality drift")
    baseline_rows = [
        {**row, "target": float(value)}
        for row, value in zip(rows, target, strict=True)
    ]
    result = stratified_equal_budget_baselines(
        baseline_rows, target, gate_mask, outer_folds, seed=20260830
    )
    result.pop("internal_masks", None)
    return result


def rcsp_risk_coverage_diagnostic(
    logits: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    *,
    coverages: Sequence[float] = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0),
) -> list[dict]:
    """Fixed switch-logit ranking diagnostic, never an operating point."""

    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    if (
        len(logits) == 0
        or logits.shape != target.shape
        or scenes.shape != target.shape
        or not np.isfinite(logits).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("invalid RCSP risk-coverage input")
    order = np.argsort(-logits, kind="stable")
    result = []
    for coverage in coverages:
        budget = min(len(logits), max(1, int(math.ceil(len(logits) * coverage))))
        mask = np.zeros(len(logits), dtype=bool)
        mask[order[:budget]] = True
        result.append({
            "requested_coverage": float(coverage),
            "score_name": "switch_logit",
            "selection_used": False,
            **outcome_evidence(mask, target, scenes),
        })
    return result


__all__ = [
    "domain_scene_episode_weights",
    "fit_primal_dual_policy",
    "nested_rcsp_fit",
    "rcsp_equal_budget_baselines",
    "rcsp_risk_coverage_diagnostic",
    "scene_cluster_bootstrap",
]
