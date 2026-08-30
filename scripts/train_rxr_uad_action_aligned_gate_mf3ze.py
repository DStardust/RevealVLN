#!/usr/bin/env python3
"""Fit the MF3ZE action-aligned return gate with scene-disjoint OOF evidence."""

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

from revealnav_mf2r6.protocol import scene_fold
from revealnav_mf3.action_aligned import FEATURE_NAMES, action_aligned_features
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


SOURCE = ROOT / (
    "artifacts/phase1/mf3zd_direct_switch_returns_v1/"
    "MF3ZD_DIRECT_SWITCH_MANIFEST.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZE_ACTION_ALIGNED_RETURN_GATE.md"
OUT = ROOT / "artifacts/training/mf3ze_action_aligned_return_gate_v1"
GATE = OUT / "MF3ZE_CROSSFIT_GATE.json"
MODEL = OUT / "MF3ZE_GATE_MODELS.npz"
SCHEMA_TAG = "mf3ze"
SOURCE_STATUS = "DIRECT_SWITCH_RETURN_DATASET_READY"
EXPECTED_ROWS = 93
EXPECTED_SCENES = 50
MIN_AUTHORIZED = 12
SEED = 20260830
BOOTSTRAPS = 24
L2_GRID = (0.1, 1.0, 10.0, 100.0)
HARM_LABEL_THRESHOLD = -0.05
CATASTROPHIC_THRESHOLD = -0.10


def feature_vector(row: dict, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    vector = action_aligned_features(
        row["decision"], arrays["instruction"], arrays["checkpoint"],
        arrays["native"], arrays["alternative"],
    )
    return vector, list(FEATURE_NAMES)


def load() -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]:
    source = json.loads(SOURCE.read_text())
    records = source["records"]
    if not (
        source.get("status") == SOURCE_STATUS
        and source.get("unseen_or_test_read") is False
        and len(records) == EXPECTED_ROWS
        and source.get("counts", {}).get("scenes") == EXPECTED_SCENES
    ):
        raise RuntimeError("MF3ZE direct-switch manifest drift")
    features = []
    names = None
    for index, row in enumerate(records):
        if row.get("row_index") != index:
            raise RuntimeError("MF3ZE row alignment drift")
        path = (ROOT / row["feature"]["path"]).resolve()
        if not (
            ROOT in path.parents and path.is_file() and not path.is_symlink()
            and path.stat().st_size == row["feature"]["bytes"]
            and sha256_file(path) == row["feature"]["sha256"]
        ):
            raise RuntimeError("MF3ZE feature provenance drift")
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key].copy() for key in payload.files}
        if set(arrays) != {"instruction", "checkpoint", "native", "alternative"}:
            raise RuntimeError("MF3ZE feature schema drift")
        vector, current_names = feature_vector(row, arrays)
        names = current_names if names is None else names
        if names != current_names:
            raise RuntimeError("MF3ZE feature-name drift")
        features.append(vector)
    target = np.asarray([row["delta"]["utility"] for row in records], dtype=np.float64)
    matrix = np.stack(features)
    if matrix.shape != (EXPECTED_ROWS, len(names)) or not np.isfinite(target).all():
        raise RuntimeError("MF3ZE training tensor drift")
    return matrix, target, records, names


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(0)
    scale = matrix.std(0)
    scale[scale < 1e-6] = 1.0
    return (matrix - mean) / scale, mean, scale


def add_intercept(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((np.ones((len(matrix), 1)), matrix), axis=1)


def ridge_fit(matrix: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    design = add_intercept(matrix)
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 1e-8
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def logistic_fit(matrix: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    design = add_intercept(matrix)
    prior = (target.sum() + 0.5) / (len(target) + 1.0)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prior / (1.0 - prior))
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 1e-8
    for _ in range(80):
        logits = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.maximum(probability * (1.0 - probability), 1e-5)
        gradient = design.T @ (probability - target) + penalty @ coefficients
        hessian = design.T @ (curvature[:, None] * design) + penalty
        update = np.linalg.solve(hessian, gradient)
        coefficients -= update
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return coefficients


def predict_model(
    matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    return_coef: np.ndarray, harm_coef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = add_intercept((matrix - mean) / scale)
    expected = design @ return_coef
    logits = np.clip(design @ harm_coef, -30.0, 30.0)
    return expected, 1.0 / (1.0 + np.exp(-logits))


def bootstrap_fit(
    matrix: np.ndarray, target: np.ndarray, scenes: np.ndarray, l2: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(scenes)
    models = []
    harm = (target <= HARM_LABEL_THRESHOLD).astype(np.float64)
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(scenes == scene) for scene in sampled])
        standardized, mean, scale = standardize(matrix[indices])
        models.append((
            mean, scale,
            ridge_fit(standardized, target[indices], l2),
            logistic_fit(standardized, harm[indices], l2),
        ))
    return models


def ensemble_predict(
    models: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected = []
    harm = []
    for mean, scale, return_coef, harm_coef in models:
        current_expected, current_harm = predict_model(
            matrix, mean, scale, return_coef, harm_coef
        )
        expected.append(current_expected)
        harm.append(current_harm)
    expected_array = np.stack(expected, axis=1)
    median = np.median(expected_array, axis=1)
    robust = median - 0.5 * np.median(np.abs(expected_array - median[:, None]), axis=1)
    return robust, np.quantile(np.stack(harm, axis=1), 0.75, axis=1)


def crossfit(
    matrix: np.ndarray, target: np.ndarray, scenes: np.ndarray, folds: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    expected = np.zeros(len(target), dtype=np.float64)
    harm = np.zeros(len(target), dtype=np.float64)
    evidence = []
    for fold in range(5):
        fit = folds != fold
        evaluate = folds == fold
        models = bootstrap_fit(
            matrix[fit], target[fit], scenes[fit], l2, SEED + fold * 1000
        )
        expected[evaluate], harm[evaluate] = ensemble_predict(models, matrix[evaluate])
        evidence.append({
            "fold": fold, "fit_rows": int(fit.sum()),
            "evaluation_rows": int(evaluate.sum()),
            "fit_scenes": int(len(set(scenes[fit]))),
            "evaluation_scenes": int(len(set(scenes[evaluate]))),
            "scene_overlap": sorted(set(scenes[fit]) & set(scenes[evaluate])),
        })
    return expected, harm, evidence


def candidate_rules(expected: np.ndarray, harm: np.ndarray):
    return_thresholds = np.unique(np.quantile(expected, np.linspace(0.0, 0.90, 19)))
    harm_thresholds = np.unique(np.quantile(harm, np.linspace(0.10, 1.0, 19)))
    for return_threshold in return_thresholds:
        for harm_threshold in harm_thresholds:
            yield float(return_threshold), float(harm_threshold), (
                (expected >= return_threshold) & (harm <= harm_threshold)
            )


def rule_evidence(mask: np.ndarray, target: np.ndarray, scenes: np.ndarray) -> dict:
    selected = target[mask]
    totals_without_scene = [
        float(target[mask & (scenes != scene)].sum()) for scene in np.unique(scenes[mask])
    ]
    return {
        "authorized": int(mask.sum()),
        "positive": int((selected > 1e-8).sum()),
        "negative": int((selected < -1e-8).sum()),
        "ties": int((np.abs(selected) <= 1e-8).sum()),
        "catastrophic": int((selected <= CATASTROPHIC_THRESHOLD).sum()),
        "total_utility": float(selected.sum()),
        "deployed_mean_utility": float(selected.sum() / len(target)),
        "selected_mean_utility": float(selected.mean()) if len(selected) else 0.0,
        "minimum_leave_one_selected_scene_out_total": (
            min(totals_without_scene) if totals_without_scene else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit",))
    args = parser.parse_args()
    if args.command != "fit":
        raise AssertionError
    if OUT.exists():
        raise RuntimeError("refusing to overwrite MF3ZE training output")
    OUT.mkdir(parents=True)
    matrix, target, records, names = load()
    scenes = np.asarray([str(row["scene_id"]) for row in records])
    folds = np.asarray([scene_fold(scene) for scene in scenes])
    ungated = rule_evidence(np.ones(len(target), dtype=bool), target, scenes)
    searches = []
    accepted = []
    for l2 in L2_GRID:
        expected, harm, fold_evidence = crossfit(matrix, target, scenes, folds, l2)
        best = None
        for return_threshold, harm_threshold, mask in candidate_rules(expected, harm):
            evidence = rule_evidence(mask, target, scenes)
            feasible = (
                evidence["authorized"] >= MIN_AUTHORIZED
                and evidence["total_utility"] > 0.0
                and evidence["catastrophic"] < ungated["catastrophic"]
                and evidence["minimum_leave_one_selected_scene_out_total"] > 0.0
            )
            candidate = {
                "l2": l2, "return_threshold": return_threshold,
                "harm_probability_threshold": harm_threshold,
                "feasible": feasible, **evidence,
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
                accepted.append((candidate, expected.copy(), harm.copy(), fold_evidence))
        searches.append(best)
    selected = None
    if accepted:
        selected = max(
            accepted,
            key=lambda value: (
                value[0]["total_utility"], -value[0]["catastrophic"],
                -value[0]["authorized"], value[0]["return_threshold"],
            ),
        )
    status = "SHADOW_GATE_PASS" if selected is not None else "SHADOW_GATE_FAIL"
    result = {
        "schema_version": f"revealnav-{SCHEMA_TAG}-action-aligned-gate/1",
        "status": status,
        "task_metric_run_authorized": selected is not None,
        "ungated_mf3v_oof_cohort": ungated,
        "search_best_by_regularization": searches,
        "feature_names": names,
        "controls": {
            "rows": len(records), "scenes": len(set(scenes)),
            "scene_folds": 5, "scene_overlap_all_folds": 0,
            "bootstraps_per_fit": BOOTSTRAPS,
            "harm_label_threshold": HARM_LABEL_THRESHOLD,
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "unseen_or_test_read": False,
        },
        "sources": {
            "direct_switch_manifest": sha256_file(SOURCE),
            "design": sha256_file(DESIGN),
        },
    }
    if selected is not None:
        rule, expected, harm, fold_evidence = selected
        result["selected_rule"] = rule
        result["folds"] = fold_evidence
        result["oof_rows"] = [
            {
                "row_index": index, "episode_id": records[index]["episode_id"],
                "scene_id": records[index]["scene_id"],
                "target_utility": float(target[index]),
                "robust_expected_utility": float(expected[index]),
                "upper_harm_probability": float(harm[index]),
                "authorized": bool(
                    expected[index] >= rule["return_threshold"]
                    and harm[index] <= rule["harm_probability_threshold"]
                ),
            }
            for index in range(len(records))
        ]
        final_models = bootstrap_fit(matrix, target, scenes, rule["l2"], SEED + 9000)
        payload = {
            "means": np.stack([model[0] for model in final_models]),
            "scales": np.stack([model[1] for model in final_models]),
            "return_coefficients": np.stack([model[2] for model in final_models]),
            "harm_coefficients": np.stack([model[3] for model in final_models]),
            "feature_names": np.asarray(names),
        }
        part = MODEL.with_name(MODEL.name + ".part")
        with part.open("wb") as stream:
            np.savez(stream, **payload)
        os.replace(part, MODEL)
        result["model"] = {
            "path": str(MODEL.relative_to(ROOT)), "bytes": MODEL.stat().st_size,
            "sha256": sha256_file(MODEL), "members": len(final_models),
        }
    atomic_json(GATE, result)
    print(json.dumps({
        "status": status, "ungated": ungated,
        "selected_rule": result.get("selected_rule"),
    }, indent=2, sort_keys=True))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
