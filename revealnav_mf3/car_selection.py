"""Nested train-development fitting for MF3ZM-CAR v1.

The module is deliberately separate from the sealed RCSP selectors.  CAR uses
the same frozen semantic model and exact rows, but aligns the optimization and
evaluation estimands: event-equal, domain-balanced weights; a hard forward
``logit > 0`` gate; event-level catastrophic constraints; and explicit
leave-one-scene utility constraints.  No function in this file can authorize a
public split.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import hashlib
import math

import numpy as np
import torch

from revealnav_mf3.car import (
    CAR_ENGINEERED_FEATURE_DIM,
    CAR_POLICY_FEATURE_NAMES,
    build_model,
    catastrophic_rate_constraint,
    event_domain_weights,
    projected_dual_update,
    selected_utility_constraint,
    utility_weighted_preference_loss,
)
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


REPRESENTATIONS = ("semantic", "engineered_28d", "policy_only")
RISK_MODES = ("hard", "soft", "none")


def _validate_inputs(
    inputs: dict[str, np.ndarray], rows: int, representation: str,
) -> None:
    if representation == "semantic":
        expected = {
            "policy": (rows, len(CAR_POLICY_FEATURE_NAMES)),
            "instruction": (rows, 768),
            "history": (rows, 768),
            "native": (rows, 768),
            "runner": (rows, 768),
        }
    elif representation == "engineered_28d":
        expected = {"engineered": (rows, CAR_ENGINEERED_FEATURE_DIM)}
    elif representation == "policy_only":
        expected = {"policy_only": (rows, len(CAR_POLICY_FEATURE_NAMES))}
    else:
        raise ValueError(f"unknown CAR representation: {representation}")
    if set(inputs) != set(expected):
        raise ValueError("CAR input field drift")
    for name, shape in expected.items():
        value = np.asarray(inputs[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid CAR {name} array")


def _standardize(
    matrix: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        matrix.ndim != 2
        or weights.ndim != 1
        or len(matrix) != len(weights)
        or len(matrix) == 0
        or not np.isfinite(matrix).all()
        or not np.isfinite(weights).all()
        or not np.all(weights > 0)
    ):
        raise ValueError("invalid CAR standardization input")
    total = float(weights.sum())
    mean = np.sum(matrix * weights[:, None], axis=0) / total
    variance = np.sum((matrix - mean) ** 2 * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _tensor_inputs(
    inputs: dict[str, np.ndarray], device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in inputs.items()
    }


def _forward(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    representation: str,
) -> torch.Tensor:
    if representation == "semantic":
        return model(
            tensors["policy"], tensors["instruction"], tensors["history"],
            tensors["native"], tensors["runner"],
        )
    field = "engineered" if representation == "engineered_28d" else "policy_only"
    return model(tensors[field])


def fit_car_ensemble(
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
    dual_cap: float,
    representation: str = "semantic",
    risk_mode: str = "hard",
    scene_constraint: bool = True,
    use_cuda: bool = True,
) -> tuple[list[torch.nn.Module], list[str], list[dict]]:
    """Fit a fixed CAR ensemble on one scene-disjoint training partition."""

    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    datasets = np.asarray([str(value) for value in datasets])
    episodes = np.asarray([str(value) for value in episodes])
    _validate_inputs(inputs, len(target), representation)
    seeds = tuple(int(value) for value in seeds)
    if (
        len(target) == 0
        or not np.isfinite(target).all()
        or not len(target) == len(scenes) == len(datasets) == len(episodes)
        or float(weight_decay) not in (0.0001, 0.001, 0.01)
        or len(seeds) != 3
        or learning_rate <= 0
        or dual_learning_rate <= 0
        or training_steps < 1
        or dual_cap <= 0
        or representation not in REPRESENTATIONS
        or risk_mode not in RISK_MODES
    ):
        raise ValueError("invalid CAR training configuration")

    event_weights = event_domain_weights(datasets)
    field = {
        "semantic": "policy",
        "engineered_28d": "engineered",
        "policy_only": "policy_only",
    }[representation]
    mean, scale = _standardize(inputs[field], event_weights)
    device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )
    tensors = _tensor_inputs(inputs, device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(
        event_weights, dtype=torch.float32, device=device
    )
    catastrophic_tensor = (
        target_tensor <= float(CATASTROPHIC_THRESHOLD)
    ).to(torch.float32)

    domains = sorted(set(datasets))
    domain_indices = {
        domain: np.flatnonzero(datasets == domain) for domain in domains
    }
    base_rates = {
        domain: float(
            np.mean(target[indices] <= float(CATASTROPHIC_THRESHOLD))
        )
        for domain, indices in domain_indices.items()
    }
    local_weights = {
        domain: torch.full(
            (len(indices),), 1.0 / len(indices), dtype=torch.float32,
            device=device,
        )
        for domain, indices in domain_indices.items()
    }
    domain_scene_values = {
        domain: sorted(set(scenes[indices]))
        for domain, indices in domain_indices.items()
    }

    models: list[torch.nn.Module] = []
    initial_hashes: list[str] = []
    diagnostics: list[dict] = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = build_model(representation).to(device)
        if representation in {"semantic", "engineered_28d"}:
            setter = (
                model.set_policy_standardization
                if representation == "semantic"
                else model.set_standardization
            )
            setter(mean, scale)
        else:
            model.set_standardization(mean, scale)
        initial_hashes.append(_state_hash(model))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        risk_duals = {
            domain: torch.zeros((), dtype=torch.float32, device=device)
            for domain in domains
        }
        utility_duals = {
            domain: torch.zeros((), dtype=torch.float32, device=device)
            for domain in domains
        }
        scene_duals = {
            (domain, scene): torch.zeros((), dtype=torch.float32, device=device)
            for domain in domains for scene in domain_scene_values[domain]
        }
        zero_risk_steps = Counter()
        zero_scene_steps = Counter()
        model.train()
        for _step in range(int(training_steps)):
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, tensors, representation)
            preference = utility_weighted_preference_loss(
                logits, target_tensor, weight_tensor
            )
            objective = preference
            constraints: dict[tuple[str, str], torch.Tensor] = {}
            for domain, indices in domain_indices.items():
                local_index = torch.as_tensor(indices, dtype=torch.long, device=device)
                d_logits = logits[local_index]
                d_target = target_tensor[local_index]
                d_catastrophic = catastrophic_tensor[local_index]
                d_weights = local_weights[domain]
                if risk_mode != "none":
                    risk_constraint, zero = catastrophic_rate_constraint(
                        d_logits, d_catastrophic, d_weights, base_rates[domain],
                        hard_forward=(risk_mode == "hard"),
                    )
                    constraints[(domain, "risk")] = risk_constraint
                    if zero:
                        zero_risk_steps[domain] += 1
                    else:
                        objective = objective + risk_duals[domain] * risk_constraint
                all_mask = torch.ones(
                    len(indices), dtype=torch.bool, device=device
                )
                utility_constraint, zero = selected_utility_constraint(
                    d_logits, d_target, d_weights, all_mask, hard_forward=True,
                )
                constraints[(domain, "utility")] = utility_constraint
                if zero:
                    zero_scene_steps[(domain, "domain")] += 1
                objective = objective + utility_duals[domain] * utility_constraint
                if scene_constraint:
                    for scene in domain_scene_values[domain]:
                        subset = torch.as_tensor(
                            scenes[indices] != scene,
                            dtype=torch.bool, device=device,
                        )
                        if not bool(subset.any()):
                            continue
                        scene_constraint_value, zero = selected_utility_constraint(
                            d_logits, d_target, d_weights, subset,
                            hard_forward=True,
                        )
                        constraints[(domain, f"scene:{scene}")] = (
                            scene_constraint_value
                        )
                        if zero:
                            zero_scene_steps[(domain, scene)] += 1
                        objective = (
                            objective
                            + scene_duals[(domain, scene)] * scene_constraint_value
                        )
            penalty = torch.zeros((), dtype=torch.float32, device=device)
            for parameter in model.parameters():
                penalty = penalty + torch.sum(parameter ** 2)
            objective = objective + float(weight_decay) * penalty
            if not bool(torch.isfinite(objective)):
                raise NestedSelectionError("CAR optimization became non-finite")
            objective.backward()
            optimizer.step()
            if risk_mode != "none":
                for domain in domains:
                    key = (domain, "risk")
                    if key in constraints:
                        risk_duals[domain] = projected_dual_update(
                            risk_duals[domain], constraints[key],
                            dual_learning_rate, maximum=dual_cap,
                        )
            for domain in domains:
                utility_duals[domain] = projected_dual_update(
                    utility_duals[domain], constraints[(domain, "utility")],
                    dual_learning_rate, maximum=dual_cap,
                )
            if scene_constraint:
                for key, value in constraints.items():
                    if key[1].startswith("scene:"):
                        scene_key = (key[0], key[1].split(":", 1)[1])
                        scene_duals[scene_key] = projected_dual_update(
                            scene_duals[scene_key], value, dual_learning_rate,
                            maximum=dual_cap,
                        )

        model.eval()
        with torch.no_grad():
            final_logits = _forward(model, tensors, representation)
            final_loss = utility_weighted_preference_loss(
                final_logits, target_tensor, weight_tensor
            )
        final_logits_np = final_logits.detach().cpu().numpy()
        hard = final_logits_np > 0.0
        models.append(model.cpu())
        diagnostics.append({
            "seed": seed,
            "device": str(device),
            "preference_loss": float(final_loss.detach().cpu()),
            "risk_mode": risk_mode,
            "scene_constraint": bool(scene_constraint),
            "zero_risk_steps": dict(zero_risk_steps),
            "zero_scene_steps": {
                f"{key[0]}:{key[1]}": int(value)
                for key, value in zero_scene_steps.items()
            },
            "hard_selected_by_domain": {
                domain: int(hard[indices].sum())
                for domain, indices in domain_indices.items()
            },
            "dual_variables": {
                "risk": {domain: float(value.detach().cpu())
                          for domain, value in risk_duals.items()},
                "utility": {domain: float(value.detach().cpu())
                             for domain, value in utility_duals.items()},
                "scene_max": float(max(
                    (value.detach().cpu().item() for value in scene_duals.values()),
                    default=0.0,
                )),
            },
            "ungated_event_catastrophic_rate": base_rates,
        })
    return models, initial_hashes, diagnostics


def predict_car_ensemble(
    models: Sequence[torch.nn.Module],
    inputs: dict[str, np.ndarray],
    representation: str,
) -> np.ndarray:
    rows = len(next(iter(inputs.values())))
    _validate_inputs(inputs, rows, representation)
    tensors = _tensor_inputs(inputs, torch.device("cpu"))
    with torch.no_grad():
        members = torch.stack([
            _forward(model, tensors, representation) for model in models
        ], dim=1)
    result = torch.median(members, dim=1).values.numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise NestedSelectionError("CAR prediction is non-finite")
    return result


def _preference_score(
    logits: np.ndarray, target: np.ndarray, weights: np.ndarray,
) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        logits.ndim != 1
        or target.ndim != 1
        or weights.ndim != 1
        or logits.shape != target.shape
        or target.shape != weights.shape
        or not np.isfinite(logits).all()
        or not np.isfinite(target).all()
        or not np.isfinite(weights).all()
        or not np.all(weights > 0)
    ):
        raise ValueError("invalid CAR preference score")
    labels = target > 0.0
    losses = np.where(
        labels, np.logaddexp(0.0, -logits), np.logaddexp(0.0, logits)
    )
    magnitude = weights * np.abs(target)
    if float(magnitude.sum()) <= 0:
        raise NestedSelectionError("CAR preference score has zero magnitude")
    return float(np.sum(losses * magnitude) / np.sum(magnitude))


def _fold_domain_evidence(
    mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> dict:
    result = {}
    for fold in sorted(set(int(value) for value in folds)):
        for domain in sorted(set(str(value) for value in datasets[folds == fold])):
            stratum = (folds == fold) & (datasets == domain)
            result[f"fold_{fold}/{domain}"] = {
                "outer_fold": int(fold),
                "dataset": domain,
                **outcome_evidence(
                    mask[stratum], target[stratum], scenes[stratum]
                ),
            }
    return result


def _scientific_evidence(
    mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> tuple[list[str], dict]:
    mask = np.asarray(mask, dtype=bool)
    target = np.asarray(target, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scenes])
    datasets = np.asarray([str(value) for value in datasets])
    folds = np.asarray(folds, dtype=np.int64)
    if (
        mask.shape != target.shape
        or scenes.shape != target.shape
        or datasets.shape != target.shape
        or folds.shape != target.shape
    ):
        raise ValueError("CAR evidence array shape drift")
    domains = domain_evidence(mask, target, scenes, datasets)
    ungated = domain_evidence(
        np.ones(len(target), dtype=bool), target, scenes, datasets
    )
    failures: list[str] = []
    for domain, value in domains.items():
        if value["authorized"] == 0:
            failures.append(f"{domain}:zero_intervention")
        if value["total_utility"] <= 0.0:
            failures.append(f"{domain}:nonpositive_utility")
        if value["minimum_leave_one_selected_scene_out_total"] <= 0.0:
            failures.append(f"{domain}:nonpositive_leave_one_selected_scene")
        if value["catastrophic_rate"] > ungated[domain]["catastrophic_rate"]:
            failures.append(f"{domain}:catastrophic_rate_above_ungated")
    strata = _fold_domain_evidence(mask, target, scenes, datasets, folds)
    for key, value in strata.items():
        if value["authorized"] == 0:
            failures.append(f"{key}:zero_intervention")
        if value["total_utility"] < 0.0:
            failures.append(f"{key}:negative_utility")
    return failures, {
        "overall": outcome_evidence(mask, target, scenes),
        "domains": domains,
        "ungated_domains": ungated,
        "fold_domain": strata,
    }


def _baseline_failures(
    gate_evidence: dict,
    baselines: dict,
    gate_mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> list[str]:
    failures: list[str] = []
    matched = baselines["fold_domain_matched"]["baselines"]
    for baseline_name in ("low_native_margin", "high_proposal_score"):
        baseline = matched[baseline_name]
        if gate_evidence["overall"]["total_utility"] <= baseline["overall"]["total_utility"]:
            failures.append(f"overall:utility_not_above_{baseline_name}")
        if gate_evidence["overall"]["catastrophic_rate"] > baseline["overall"]["catastrophic_rate"]:
            failures.append(f"overall:catastrophic_rate_above_{baseline_name}")
        for domain, value in gate_evidence["domains"].items():
            control = baseline["domains"][domain]
            if value["total_utility"] <= control["total_utility"]:
                failures.append(f"{domain}:utility_not_above_{baseline_name}")
            if value["catastrophic_rate"] > control["catastrophic_rate"]:
                failures.append(f"{domain}:catastrophic_rate_above_{baseline_name}")
        matched_mask = np.asarray(
            baselines["internal_masks"]["fold_domain_matched"][baseline_name],
            dtype=bool,
        )
        gate_mask = np.asarray(gate_mask, dtype=bool)
        target = np.asarray(target, dtype=np.float64)
        scenes = np.asarray([str(value) for value in scenes])
        datasets = np.asarray([str(value) for value in datasets])
        folds = np.asarray(folds, dtype=np.int64)
        for key, value in gate_evidence["fold_domain"].items():
            fold = int(key.split("/", 1)[0].split("_", 1)[1])
            domain = key.split("/", 1)[1]
            stratum = (folds == fold) & (datasets == domain)
            control = outcome_evidence(
                matched_mask[stratum], target[stratum], scenes[stratum]
            )
            if value["total_utility"] <= control["total_utility"]:
                failures.append(f"{key}:utility_not_above_{baseline_name}")
            if value["catastrophic_rate"] > control["catastrophic_rate"]:
                failures.append(f"{key}:catastrophic_rate_above_{baseline_name}")
    return failures


def _evaluate_candidate(
    rows: list[dict],
    mask: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> tuple[list[str], dict, dict]:
    failures, evidence = _scientific_evidence(
        mask, target, scenes, datasets, folds
    )
    baseline_rows = [
        {**row, "target": float(value)}
        for row, value in zip(rows, target, strict=True)
    ]
    baselines = stratified_equal_budget_baselines(
        baseline_rows, target, mask, folds, seed=20260830
    )
    failures.extend(_baseline_failures(
        evidence, baselines, mask, target, scenes, datasets, folds
    ))
    return failures, evidence, baselines


def _subset_inputs(
    inputs: dict[str, np.ndarray], mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask] for name, value in inputs.items()}


def nested_car_fit(
    rows: list[dict],
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    episodes: np.ndarray,
    outer_folds: np.ndarray,
    config: dict,
    *,
    representation: str = "semantic",
    risk_mode: str = "hard",
    scene_constraint: bool = True,
    continue_after_fold_failure: bool = True,
) -> dict:
    """Perform nested CAR selection without using outer targets for choice."""

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
        and representation in REPRESENTATIONS
        and risk_mode in RISK_MODES
    ):
        raise ValueError("invalid CAR nested-fit arrays")
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
        raise ValueError("CAR nested configuration drift")

    scene_assignment: dict[str, set[int]] = {}
    for scene, fold in zip(scenes, outer_folds, strict=True):
        scene_assignment.setdefault(scene, set()).add(int(fold))
    if any(len(value) != 1 for value in scene_assignment.values()):
        raise NestedSelectionError("CAR outer split divided an MP3D scene")

    fit_kwargs = {
        "seeds": seeds,
        "learning_rate": float(config["learning_rate"]),
        "dual_learning_rate": float(config["dual_learning_rate"]),
        "training_steps": int(config["training_steps"]),
        "dual_cap": float(config["dual_cap"]),
        "representation": representation,
        "risk_mode": risk_mode,
        "scene_constraint": bool(scene_constraint),
        "use_cuda": bool(config.get("use_cuda", True)),
    }
    outer_logits = np.full(len(target), np.nan, dtype=np.float64)
    outer_records: list[dict] = []
    selected_wd: list[float] = []
    failures: list[str] = []
    for outer_fold in range(outer_count):
        evaluate = outer_folds == outer_fold
        fit = ~evaluate
        fit_scenes = sorted(set(scenes[fit]))
        evaluation_scenes = sorted(set(scenes[evaluate]))
        if not fit.any() or not evaluate.any() or set(fit_scenes) & set(evaluation_scenes):
            raise NestedSelectionError(f"invalid CAR outer fold {outer_fold}")
        _, mapping = deterministic_scene_folds(
            fit_scenes, inner_count,
            salt=f"{config['inner_fold_salt']}:outer:{outer_fold}",
        )
        inner_folds = np.asarray([mapping[scene] for scene in scenes[fit]])
        trials: list[dict] = []
        initialization: dict[float, list[list[str]]] = {}
        for weight_decay in weight_decay_grid:
            inner_logits = np.full(int(fit.sum()), np.nan, dtype=np.float64)
            fold_records: list[dict] = []
            hashes_by_fold: list[list[str]] = []
            for inner_fold in range(inner_count):
                inner_evaluate = inner_folds == inner_fold
                inner_fit = ~inner_evaluate
                train_scenes = set(scenes[fit][inner_fit])
                held_scenes = set(scenes[fit][inner_evaluate])
                if not inner_fit.any() or not inner_evaluate.any() or train_scenes & held_scenes:
                    raise NestedSelectionError(
                        f"invalid CAR inner fold {outer_fold}/{inner_fold}"
                    )
                models, hashes, diagnostics = fit_car_ensemble(
                    _subset_inputs(_subset_inputs(inputs, fit), inner_fit),
                    target[fit][inner_fit], scenes[fit][inner_fit],
                    datasets[fit][inner_fit], episodes[fit][inner_fit],
                    weight_decay=weight_decay, **fit_kwargs,
                )
                inner_logits[inner_evaluate] = predict_car_ensemble(
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
                raise NestedSelectionError("CAR inner OOF logits incomplete")
            inner_mask = inner_logits > 0.0
            inner_rows = [
                row for row, keep in zip(rows, fit, strict=True) if keep
            ]
            candidate_failures, evidence, baselines = _evaluate_candidate(
                inner_rows, inner_mask, target[fit], scenes[fit],
                datasets[fit], inner_folds,
            )
            score = _preference_score(
                inner_logits, target[fit], event_domain_weights(datasets[fit])
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
            raise NestedSelectionError("CAR WD candidates did not share initialization")
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
            if not continue_after_fold_failure:
                break
            continue
        selected = min(
            feasible,
            key=lambda value: (
                value["inner_oof_preference_loss"], value["weight_decay"]
            ),
        )
        models, _, diagnostics = fit_car_ensemble(
            _subset_inputs(inputs, fit), target[fit], scenes[fit],
            datasets[fit], episodes[fit],
            weight_decay=float(selected["weight_decay"]), **fit_kwargs,
        )
        outer_logits[evaluate] = predict_car_ensemble(
            models, _subset_inputs(inputs, evaluate), representation
        )
        selected_wd.append(float(selected["weight_decay"]))
        outer_gate = outer_logits[evaluate] > 0.0
        _, outer_evidence = _scientific_evidence(
            outer_gate, target[evaluate], scenes[evaluate],
            datasets[evaluate], outer_folds[evaluate],
        )
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
            "outer_evidence": outer_evidence,
        })

    if not np.isfinite(outer_logits).all():
        return {
            "status": "NESTED_CAR_FAIL",
            "failure_reasons": failures + ["outer_oof_incomplete"],
            "decision_rule": "switch_logit > 0",
            "representation": representation,
            "risk_mode": risk_mode,
            "scene_constraint": bool(scene_constraint),
            "outer_folds": outer_records,
            "final_models": [],
        }

    gate_mask = outer_logits > 0.0
    scientific_failures, evidence = _scientific_evidence(
        gate_mask, target, scenes, datasets, outer_folds
    )
    baseline_rows = [
        {**row, "target": float(value)}
        for row, value in zip(rows, target, strict=True)
    ]
    baselines = stratified_equal_budget_baselines(
        baseline_rows, target, gate_mask, outer_folds, seed=20260830
    )
    failures.extend(scientific_failures)
    failures.extend(_baseline_failures(
        evidence, baselines, gate_mask, target, scenes, datasets, outer_folds
    ))
    modal_wd = (
        min(
            value for value, count in Counter(selected_wd).items()
            if count == max(Counter(selected_wd).values())
        ) if selected_wd else None
    )
    final_models: list[torch.nn.Module] = []
    final_diagnostics: list[dict] = []
    if not failures and modal_wd is not None:
        final_models, _, final_diagnostics = fit_car_ensemble(
            inputs, target, scenes, datasets, episodes,
            weight_decay=modal_wd, **fit_kwargs,
        )
    return {
        "status": "NESTED_CAR_PASS" if not failures else "NESTED_CAR_FAIL",
        "failure_reasons": failures,
        "decision_rule": "switch_logit > 0",
        "representation": representation,
        "risk_mode": risk_mode,
        "scene_constraint": bool(scene_constraint),
        "selected_weight_decay": modal_wd,
        "outer_fold_weight_decay": selected_wd,
        "outer_folds": outer_records,
        "outer_oof": {
            "switch_logit": outer_logits,
            "authorized_mask": gate_mask,
            "evidence": evidence,
            "equal_budget": baselines,
            "scene_cluster_bootstrap": scene_cluster_bootstrap(
                gate_mask, target, scenes, datasets,
                comparator_mask=baselines["internal_masks"]["fold_domain_matched"][
                    "low_native_margin"
                ], replicates=10_000, seed=20260830,
            ),
        },
        "final_training_diagnostics": final_diagnostics,
        "final_models": final_models,
    }


__all__ = [
    "REPRESENTATIONS",
    "RISK_MODES",
    "event_domain_weights",
    "fit_car_ensemble",
    "nested_car_fit",
    "predict_car_ensemble",
]
