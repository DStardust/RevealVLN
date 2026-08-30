#!/usr/bin/env python3
"""Fit the MF3ZK action-aligned return gate on RxR/R2R train data.

The implementation keeps the proposal ranker frozen and fits only the linear
return/harm screen.  All model selection is scene-disjoint OOF on train data;
the sealed confirmation cohort is never consulted by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2r6.protocol import scene_fold  # noqa: E402
from revealnav_mf3.action_aligned import (  # noqa: E402
    FEATURE_NAMES,
    action_aligned_features,
)


PROTOCOL = ROOT / "artifacts/training/mf3zk_joint_v1/MF3ZK_JOINT_PROTOCOL.json"
R2R_MANIFEST = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_collection/"
    "MF3ZK_R2R_DIRECT_SWITCH_MANIFEST.json"
)
RXR_CORE = ROOT / (
    "artifacts/phase1/mf3zd_direct_switch_returns_v1/"
    "MF3ZD_DIRECT_SWITCH_MANIFEST.json"
)
RXR_EXPANSION = ROOT / (
    "artifacts/phase1/mf3zf_expanded_direct_switch_returns_v1/"
    "MF3ZF_DIRECT_SWITCH_MANIFEST.json"
)
OUT = ROOT / "artifacts/training/mf3zk_joint_v1/gates"
SEED = 20260830
BOOTSTRAPS = 24
L2_GRID = (0.1, 1.0, 10.0, 100.0)
HARM_THRESHOLD = -0.05
CATASTROPHIC_THRESHOLD = -0.10
MIN_AUTHORIZED_PER_DOMAIN = 8
MIN_AUTHORIZED_SINGLE_DOMAIN = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    """Return a stable project-relative path when possible.

    The production output is always below ``ROOT``.  Keeping the fallback
    makes the fitting helper independently testable in a temporary directory
    without weakening the production path checks.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def _load_feature(path: Path) -> dict[str, np.ndarray]:
    resolved = path.resolve()
    if (
        ROOT not in resolved.parents or resolved.is_symlink()
        or not resolved.is_file()
    ):
        raise RuntimeError(f"unsafe action-aligned feature: {path}")
    with np.load(resolved, allow_pickle=False) as payload:
        if set(payload.files) != {"instruction", "checkpoint", "native", "alternative"}:
            raise RuntimeError(f"feature schema drift: {path}")
        arrays = {key: payload[key].astype(np.float64) for key in payload.files}
    if any(value.shape != (768,) or not np.isfinite(value).all()
           for value in arrays.values()):
        raise RuntimeError(f"feature value drift: {path}")
    return arrays


def _load_rxr(path: Path, tier: str, allowed_indices: set[int] | None) -> list[dict]:
    value = json.loads(path.read_text())
    if value.get("status") != "DIRECT_SWITCH_RETURN_DATASET_READY":
        raise RuntimeError(f"RxR source is not ready: {path}")
    if value.get("unseen_or_test_read") is not False:
        raise RuntimeError(f"RxR source crossed public split: {path}")
    rows = []
    for row in value.get("records", []):
        index = int(row["row_index"])
        if allowed_indices is not None and index not in allowed_indices:
            continue
        feature = (ROOT / row["feature"]["path"]).resolve()
        arrays = _load_feature(feature)
        rows.append({
            "dataset": "RxR", "tier": tier,
            "scene_id": str(row["scene_id"]),
            "episode_id": str(row["episode_id"]),
            "source_manifest": str(path.relative_to(ROOT)),
            "source_row_index": index,
            "decision": row["decision"],
            "target": float(row["delta"]["utility"]),
            "arrays": arrays,
            "feature": {
                "path": str(feature.relative_to(ROOT)),
                "bytes": feature.stat().st_size,
                "sha256": sha256_file(feature),
            },
        })
    return rows


def _load_r2r(
    path: Path, allowed_scenes: set[str], excluded_scenes: set[str] | None = None,
) -> list[dict]:
    value = json.loads(path.read_text())
    if value.get("status") != "R2R_DIRECT_SWITCH_RETURN_DATASET_READY":
        raise RuntimeError("R2R exact return manifest is not ready")
    if value.get("split") != "train" or value.get("unseen_or_test_read") is not False:
        raise RuntimeError("R2R exact return manifest split boundary drift")
    rows = []
    for row in value.get("records", []):
        scene = str(row["scene_id"])
        if excluded_scenes is not None and scene in excluded_scenes:
            continue
        if scene not in allowed_scenes:
            raise RuntimeError("R2R collection row entered a held-out confirmation scene")
        feature = (ROOT / row["feature"]["path"]).resolve()
        arrays = _load_feature(feature)
        rows.append({
            "dataset": "R2R", "tier": str(row["tier"]),
            "scene_id": str(row["scene_id"]),
            "episode_id": str(row["episode_id"]),
            "source_manifest": str(path.relative_to(ROOT)),
            "source_row_index": int(row["row_index"]),
            "decision": row["decision"],
            "target": float(row["delta"]["utility"]),
            "arrays": arrays,
            "feature": {
                "path": str(feature.relative_to(ROOT)),
                "bytes": feature.stat().st_size,
                "sha256": sha256_file(feature),
            },
        })
    return rows


def _vector(row: dict) -> np.ndarray:
    vector = action_aligned_features(
        row["decision"], row["arrays"]["instruction"],
        row["arrays"]["checkpoint"], row["arrays"]["native"],
        row["arrays"]["alternative"],
    )
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise RuntimeError("action-aligned feature construction drift")
    return vector


def _weights(datasets: np.ndarray) -> np.ndarray:
    unique = sorted(set(str(value) for value in datasets))
    if len(unique) == 1:
        return np.ones(len(datasets), dtype=np.float64)
    result = np.zeros(len(datasets), dtype=np.float64)
    for domain in unique:
        mask = datasets == domain
        result[mask] = len(datasets) / (len(unique) * int(mask.sum()))
    return result


def _standardize(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = max(float(weights.sum()), 1e-12)
    mean = (matrix * weights[:, None]).sum(0) / total
    variance = (((matrix - mean) ** 2) * weights[:, None]).sum(0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[scale < 1e-6] = 1.0
    return (matrix - mean) / scale, mean, scale


def _design(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((np.ones((len(matrix), 1)), matrix), axis=1)


def _ridge_fit(matrix: np.ndarray, target: np.ndarray, weights: np.ndarray, l2: float) -> np.ndarray:
    design = _design(matrix)
    weighted = weights[:, None] * design
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 1e-8
    return np.linalg.solve(design.T @ weighted + penalty, design.T @ (weights * target))


def _logistic_fit(matrix: np.ndarray, target: np.ndarray, weights: np.ndarray, l2: float) -> np.ndarray:
    design = _design(matrix)
    prior = (float((weights * target).sum()) + 0.5) / (float(weights.sum()) + 1.0)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prior / (1.0 - prior))
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 1e-8
    for _ in range(80):
        logits = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.maximum(probability * (1.0 - probability), 1e-5)
        weighted_error = weights * (probability - target)
        gradient = design.T @ weighted_error + penalty @ coefficients
        hessian = design.T @ ((weights * curvature)[:, None] * design) + penalty
        update = np.linalg.solve(hessian, gradient)
        coefficients -= update
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return coefficients


def _bootstrap_fit(
    matrix: np.ndarray, target: np.ndarray, scenes: np.ndarray,
    datasets: np.ndarray, l2: float, seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    models = []
    domains = sorted(set(str(value) for value in datasets))
    harm = (target <= HARM_THRESHOLD).astype(np.float64)
    for _ in range(BOOTSTRAPS):
        indices = []
        # Draw scene clusters independently per domain.  Combined fitting is
        # thus balanced even when RxR and R2R have different row counts.
        for domain in domains:
            domain_scenes = np.unique(scenes[datasets == domain])
            sampled = rng.choice(domain_scenes, len(domain_scenes), replace=True)
            indices.extend(
                int(index)
                for scene in sampled
                for index in np.flatnonzero((datasets == domain) & (scenes == scene))
            )
        selected = np.asarray(indices, dtype=np.int64)
        weights = _weights(datasets[selected])
        standardized, mean, scale = _standardize(matrix[selected], weights)
        models.append((
            mean, scale,
            _ridge_fit(standardized, target[selected], weights, l2),
            _logistic_fit(standardized, harm[selected], weights, l2),
        ))
    return models


def _predict(
    models: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected, harm = [], []
    for mean, scale, return_coef, harm_coef in models:
        design = _design((matrix - mean) / scale)
        expected.append(design @ return_coef)
        logits = np.clip(design @ harm_coef, -30.0, 30.0)
        harm.append(1.0 / (1.0 + np.exp(-logits)))
    expected_array = np.stack(expected, axis=1)
    median = np.median(expected_array, axis=1)
    robust = median - 0.5 * np.median(np.abs(expected_array - median[:, None]), axis=1)
    upper_harm = np.quantile(np.stack(harm, axis=1), 0.75, axis=1)
    return robust, upper_harm


def _crossfit(
    matrix: np.ndarray, target: np.ndarray, scenes: np.ndarray,
    datasets: np.ndarray, l2: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    folds = np.asarray([scene_fold(str(scene)) for scene in scenes])
    expected = np.zeros(len(target), dtype=np.float64)
    harm = np.zeros(len(target), dtype=np.float64)
    evidence = []
    for fold in range(5):
        fit = folds != fold
        evaluate = folds == fold
        if not bool(evaluate.any()) or not bool(fit.any()):
            evidence.append({"fold": fold, "fit_rows": int(fit.sum()), "evaluation_rows": int(evaluate.sum()), "skipped": True})
            continue
        models = _bootstrap_fit(
            matrix[fit], target[fit], scenes[fit], datasets[fit],
            l2, SEED + fold * 1000,
        )
        expected[evaluate], harm[evaluate] = _predict(models, matrix[evaluate])
        evidence.append({
            "fold": fold, "fit_rows": int(fit.sum()),
            "evaluation_rows": int(evaluate.sum()),
            "fit_scenes": int(len(set(scenes[fit]))),
            "evaluation_scenes": int(len(set(scenes[evaluate]))),
            "scene_overlap": sorted(set(scenes[fit]) & set(scenes[evaluate])),
            "fit_domains": sorted(set(datasets[fit])),
            "evaluation_domains": sorted(set(datasets[evaluate])),
        })
    if not np.isfinite(expected).all() or not np.isfinite(harm).all():
        raise RuntimeError("non-finite OOF prediction")
    return expected, harm, evidence


def _rule_grid(expected: np.ndarray, harm: np.ndarray):
    return_thresholds = np.unique(np.quantile(expected, np.linspace(0.0, 0.90, 19)))
    harm_thresholds = np.unique(np.quantile(harm, np.linspace(0.10, 1.0, 19)))
    for return_threshold in return_thresholds:
        for harm_threshold in harm_thresholds:
            yield float(return_threshold), float(harm_threshold), (
                (expected >= return_threshold) & (harm <= harm_threshold)
            )


def _evidence(mask: np.ndarray, target: np.ndarray, scenes: np.ndarray) -> dict:
    selected = target[mask]
    selected_scenes = np.unique(scenes[mask])
    totals = [float(target[mask & (scenes != scene)].sum()) for scene in selected_scenes]
    return {
        "authorized": int(mask.sum()),
        "positive": int((selected > 1e-8).sum()),
        "negative": int((selected < -1e-8).sum()),
        "ties": int((np.abs(selected) <= 1e-8).sum()),
        "catastrophic": int((selected <= CATASTROPHIC_THRESHOLD).sum()),
        "total_utility": float(selected.sum()),
        "deployed_mean_utility": float(selected.sum() / len(target)),
        "selected_mean_utility": float(selected.mean()) if len(selected) else 0.0,
        "minimum_leave_one_selected_scene_out_total": min(totals) if totals else 0.0,
    }


def _domain_evidence(mask: np.ndarray, target: np.ndarray, scenes: np.ndarray, datasets: np.ndarray) -> dict:
    result = {}
    for domain in sorted(set(datasets)):
        domain_mask = datasets == domain
        result[domain] = _evidence(
            mask[domain_mask], target[domain_mask], scenes[domain_mask]
        )
    return result


def _fit_arm(tier: str, arm: str, rows: list[dict], output: Path) -> dict:
    selected = [row for row in rows if row["tier"] == tier and (arm == "joint" or row["dataset"] == arm)]
    if not selected:
        raise RuntimeError(f"no rows for {tier}/{arm}")
    matrix = np.stack([_vector(row) for row in selected])
    target = np.asarray([row["target"] for row in selected], dtype=np.float64)
    scenes = np.asarray([row["scene_id"] for row in selected])
    datasets = np.asarray([row["dataset"] for row in selected])
    if len(set(datasets)) == 2 and arm == "joint":
        required = {"RxR", "R2R"}
        if set(datasets) != required:
            raise RuntimeError("joint arm lost one benchmark")
    if len(set(scenes)) < 5:
        raise RuntimeError(f"insufficient scene diversity for {tier}/{arm}")
    ungated = _evidence(np.ones(len(target), dtype=bool), target, scenes)
    searches = []
    accepted = []
    for l2 in L2_GRID:
        expected, harm, folds = _crossfit(matrix, target, scenes, datasets, l2)
        best = None
        for return_threshold, harm_threshold, mask in _rule_grid(expected, harm):
            evidence = _evidence(mask, target, scenes)
            domains = _domain_evidence(mask, target, scenes, datasets)
            ungated_domains = {
                domain: _evidence(
                    datasets == domain, target, scenes
                ) for domain in sorted(set(datasets))
            }
            minimum = MIN_AUTHORIZED_PER_DOMAIN if arm == "joint" else MIN_AUTHORIZED_SINGLE_DOMAIN
            feasible = (
                evidence["authorized"] >= minimum
                and evidence["total_utility"] > 0.0
                and evidence["catastrophic"] < ungated["catastrophic"]
                and evidence["minimum_leave_one_selected_scene_out_total"] > 0.0
                and all(
                    value["authorized"] >= MIN_AUTHORIZED_PER_DOMAIN
                    and value["total_utility"] > 0.0
                    and value["catastrophic"] < ungated_domains[domain]["catastrophic"]
                    and value["minimum_leave_one_selected_scene_out_total"] > 0.0
                    for domain, value in domains.items()
                )
            )
            candidate = {
                "l2": l2, "return_threshold": return_threshold,
                "harm_probability_threshold": harm_threshold,
                "feasible": feasible, **evidence, "domains": domains,
            }
            if best is None or (
                candidate["feasible"], candidate["total_utility"],
                -candidate["catastrophic"], -candidate["authorized"],
                candidate["return_threshold"],
            ) > (
                best["feasible"], best["total_utility"],
                -best["catastrophic"], -best["authorized"],
                best["return_threshold"],
            ):
                best = candidate
            if feasible:
                accepted.append((candidate, expected.copy(), harm.copy(), folds))
        searches.append(best)
    selected_rule = None
    if accepted:
        selected_rule, expected, harm, folds = max(
            accepted,
            key=lambda value: (
                value[0]["total_utility"], -value[0]["catastrophic"],
                -value[0]["authorized"], value[0]["return_threshold"],
            ),
        )
    status = "TRAIN_RETURN_GATE_PASS" if selected_rule is not None else "TRAIN_RETURN_GATE_FAIL"
    result = {
        "schema_version": f"revealnav-mf3zk-{tier.lower()}-{arm}-gate/1",
        "status": status,
        "task_metric_run_authorized": False,
        "public_unseen_authorized": False,
        "tier": tier, "arm": arm,
        "rows": len(selected), "scenes": len(set(scenes)),
        "datasets": sorted(set(datasets)),
        "ungated_oof_cohort": ungated,
        "search_best_by_regularization": searches,
        "feature_names": list(FEATURE_NAMES),
        "controls": {
            "scene_folds": 5, "scene_overlap_all_folds": 0,
            "bootstraps_per_fit": BOOTSTRAPS,
            "harm_label_threshold": HARM_THRESHOLD,
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "dataset_balanced_effective_weight": arm == "joint",
            "unseen_or_test_read": False,
        },
    }
    if selected_rule is not None:
        result["selected_rule"] = selected_rule
        result["oof_rows"] = [
            {
                "row_index": index,
                "dataset": selected[index]["dataset"],
                "tier": tier,
                "episode_id": selected[index]["episode_id"],
                "scene_id": selected[index]["scene_id"],
                "target_utility": float(target[index]),
                "robust_expected_utility": float(expected[index]),
                "upper_harm_probability": float(harm[index]),
                "authorized": bool(
                    expected[index] >= selected_rule["return_threshold"]
                    and harm[index] <= selected_rule["harm_probability_threshold"]
                ),
            }
            for index in range(len(selected))
        ]
        final_models = _bootstrap_fit(
            matrix, target, scenes, datasets, selected_rule["l2"], SEED + 9000
        )
        model_path = output / f"MF3ZK_{tier.upper()}_{arm.upper()}_GATE_MODELS.npz"
        part = model_path.with_name(model_path.name + ".part")
        with part.open("wb") as stream:
            np.savez(
                stream,
                means=np.stack([model[0] for model in final_models]),
                scales=np.stack([model[1] for model in final_models]),
                return_coefficients=np.stack([model[2] for model in final_models]),
                harm_coefficients=np.stack([model[3] for model in final_models]),
                feature_names=np.asarray(FEATURE_NAMES),
            )
        os.replace(part, model_path)
        result["model"] = {
            "path": provenance_path(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
            "members": len(final_models),
        }
    gate_path = output / f"MF3ZK_{tier.upper()}_{arm.upper()}_GATE.json"
    atomic_json(gate_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit",))
    args = parser.parse_args()
    if args.command != "fit":
        raise AssertionError
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "SEALED_BEFORE_MF3ZK_JOINT_TRAINING":
        raise RuntimeError("MF3ZK protocol is not sealed")
    holdout = set(protocol["strict_scene_holdout"]["confirmation_scenes"])
    rxr_allowed = {
        (row["tier"], int(row["source_row_index"]))
        for row in protocol["rxr_sources"]["fit_rows"]
    }
    rows = []
    rows.extend(_load_rxr(
        RXR_CORE, "core", {index for tier, index in rxr_allowed if tier == "core"}
    ))
    rows.extend(_load_rxr(
        RXR_EXPANSION, "expansion", {index for tier, index in rxr_allowed if tier == "expansion"}
    ))
    excluded_r2r_holdout_rows = 0
    r2r_fit_scenes = set(protocol["r2r_train"]["fit_scenes"]) - holdout
    if R2R_MANIFEST.is_file():
        r2r_rows = _load_r2r(R2R_MANIFEST, r2r_fit_scenes, holdout)
        excluded_r2r_holdout_rows = sum(
            row.get("scene_id") in holdout
            for row in json.loads(R2R_MANIFEST.read_text()).get("records", [])
        )
        rows.extend(r2r_rows)
    if any(row["scene_id"] in holdout for row in rows):
        raise RuntimeError("joint training row entered strict confirmation holdout")
    if OUT.exists():
        raise RuntimeError("refusing to overwrite MF3ZK gate directory")
    OUT.mkdir(parents=True)
    results = []
    for tier in ("core", "expansion"):
        for arm in ("joint", "RxR", "R2R"):
            try:
                result = _fit_arm(tier, arm, rows, OUT)
            except RuntimeError as error:
                result = {
                    "schema_version": f"revealnav-mf3zk-{tier.lower()}-{arm.lower()}-gate/1",
                    "status": "TRAIN_RETURN_GATE_FAIL",
                    "task_metric_run_authorized": False,
                    "tier": tier, "arm": arm,
                    "error": str(error),
                }
                atomic_json(OUT / f"MF3ZK_{tier.upper()}_{arm.upper()}_GATE.json", result)
            results.append(result)
    summary = {
        "schema_version": "revealnav-mf3zk-joint-training-result/1",
        "status": "PASS" if all(
            next(row for row in results
                 if row.get("tier") == tier and row.get("arm") == "joint")["status"]
            == "TRAIN_RETURN_GATE_PASS"
            for tier in ("core", "expansion")
        ) else "FAIL",
        "mainline": "joint",
        "results": results,
        "rows": len(rows),
        "scene_count": len({row["scene_id"] for row in rows}),
        "strict_holdout_exclusions": {
            "r2r_rows_excluded_due_to_rxr_or_r2r_confirmation_scene": int(
                excluded_r2r_holdout_rows
            ),
            "holdout_scene_count": len(holdout),
        },
        "source_hashes": {
            "protocol": sha256_file(PROTOCOL),
            "r2r_manifest": sha256_file(R2R_MANIFEST) if R2R_MANIFEST.is_file() else None,
            "rxr_core": sha256_file(RXR_CORE),
            "rxr_expansion": sha256_file(RXR_EXPANSION),
        },
        "unseen_or_test_read": False,
    }
    atomic_json(OUT / "MF3ZK_JOINT_TRAINING_RESULT.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "rows": len(rows),
        "gates": {f"{row['tier']}/{row['arm']}": row["status"] for row in results},
    }, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
