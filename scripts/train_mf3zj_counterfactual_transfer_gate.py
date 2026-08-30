#!/usr/bin/env python3
"""Fit MF3ZJ's train-only proposal-invariant fallback safety gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2r6.protocol import scene_fold  # noqa: E402
from revealnav_mf3.uncertainty_gate import (  # noqa: E402
    TRANSFER_FEATURE_NAMES,
    transfer_action_features,
)
from scripts.train_rxr_uad_action_aligned_gate_mf3ze import (  # noqa: E402
    add_intercept,
    atomic_json,
    ensemble_predict,
    logistic_fit,
    ridge_fit,
    sha256_file,
    standardize,
)


LEARNED_SOURCE = ROOT / (
    "artifacts/phase1/mf3zf_expanded_direct_switch_returns_v1/"
    "MF3ZF_DIRECT_SWITCH_MANIFEST.json"
)
FALLBACK_SOURCE = ROOT / (
    "artifacts/phase1/mf3zi_uncertainty_direct_switch_returns_v1/"
    "MF3ZI_UNCERTAINTY_MANIFEST.json"
)
DESIGN = ROOT / (
    "artifacts/design/METHOD_FREEZE_3ZJ_COUNTERFACTUAL_TRANSFER_ARBITRATION.md"
)
OUT = ROOT / "artifacts/training/mf3zj_counterfactual_transfer_gate_v1"
GATE = OUT / "MF3ZJ_CROSSFIT_GATE.json"
MODEL = OUT / "MF3ZJ_TRANSFER_GATE_MODELS.npz"
SEED = 20260830
BOOTSTRAPS = 24
L2 = 0.1
RETURN_CLIP = 0.05
HARM_THRESHOLD = -0.05
CATASTROPHIC_THRESHOLD = -0.10
MIN_AUTHORIZED = 12


def checked_records(path: Path, status: str, rows: int, scenes: int) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload.get("records", [])
    if not (
        payload.get("status") == status
        and len(records) == rows
        and payload.get("counts", {}).get("scenes") == scenes
    ):
        raise RuntimeError(f"MF3ZJ source manifest drift: {path}")
    return records


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    inputs = (
        ("learned_uad", checked_records(
            LEARNED_SOURCE, "DIRECT_SWITCH_RETURN_DATASET_READY", 217, 58
        )),
        ("native_margin", checked_records(
            FALLBACK_SOURCE,
            "UNCERTAINTY_DIRECT_SWITCH_RETURN_DATASET_READY", 126, 46,
        )),
    )
    features: list[np.ndarray] = []
    target: list[float] = []
    scenes: list[str] = []
    sources: list[str] = []
    metadata: list[dict] = []
    for source, records in inputs:
        for row in records:
            path = (ROOT / row["feature"]["path"]).resolve()
            if not (
                ROOT in path.parents
                and path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == row["feature"]["bytes"]
                and sha256_file(path) == row["feature"]["sha256"]
            ):
                raise RuntimeError("MF3ZJ feature provenance drift")
            with np.load(path, allow_pickle=False) as payload:
                if set(payload.files) != {
                    "instruction", "checkpoint", "native", "alternative"
                }:
                    raise RuntimeError("MF3ZJ feature schema drift")
                vector = transfer_action_features(
                    row["decision"], payload["instruction"],
                    payload["checkpoint"], payload["native"],
                    payload["alternative"],
                )
            features.append(vector)
            target.append(float(row["delta"]["utility"]))
            scenes.append(str(row["scene_id"]))
            sources.append(source)
            metadata.append({
                "source": source,
                "episode_id": str(row["episode_id"]),
                "scene_id": str(row["scene_id"]),
            })
    matrix = np.stack(features)
    values = np.asarray(target, dtype=np.float64)
    scene_array = np.asarray(scenes)
    source_array = np.asarray(sources)
    if matrix.shape != (343, len(TRANSFER_FEATURE_NAMES)) or not (
        np.isfinite(matrix).all() and np.isfinite(values).all()
    ):
        raise RuntimeError("MF3ZJ training tensor drift")
    return matrix, values, scene_array, source_array, metadata


def bootstrap_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(scenes)
    bounded = np.clip(target, -RETURN_CLIP, RETURN_CLIP)
    harm = (target <= HARM_THRESHOLD).astype(np.float64)
    models = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([
            np.flatnonzero(scenes == scene) for scene in sampled
        ])
        normalized, mean, scale = standardize(matrix[indices])
        models.append((
            mean,
            scale,
            ridge_fit(normalized, bounded[indices], L2),
            logistic_fit(normalized, harm[indices], L2),
        ))
    return models


def crossfit(
    matrix: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    folds = np.asarray([scene_fold(scene) for scene in scenes])
    expected = np.zeros(len(target), dtype=np.float64)
    harm = np.zeros(len(target), dtype=np.float64)
    evidence = []
    for fold in range(5):
        fit = folds != fold
        evaluate = folds == fold
        models = bootstrap_fit(
            matrix[fit], target[fit], scenes[fit], SEED + fold * 1000
        )
        expected[evaluate], harm[evaluate] = ensemble_predict(
            models, matrix[evaluate]
        )
        evidence.append({
            "fold": fold,
            "fit_rows": int(fit.sum()),
            "evaluation_rows": int(evaluate.sum()),
            "fit_scenes": int(len(set(scenes[fit]))),
            "evaluation_scenes": int(len(set(scenes[evaluate]))),
            "scene_overlap": sorted(set(scenes[fit]) & set(scenes[evaluate])),
        })
    return expected, harm, evidence


def rule_evidence(mask: np.ndarray, target: np.ndarray, scenes: np.ndarray) -> dict:
    selected = target[mask]
    selected_scenes = np.unique(scenes[mask])
    leave_one_out = [
        float(target[mask & (scenes != scene)].sum())
        for scene in selected_scenes
    ]
    return {
        "authorized": int(mask.sum()),
        "positive": int((selected > 1e-8).sum()),
        "negative": int((selected < -1e-8).sum()),
        "ties": int((np.abs(selected) <= 1e-8).sum()),
        "catastrophic": int((selected <= CATASTROPHIC_THRESHOLD).sum()),
        "total_utility": float(selected.sum()),
        "selected_mean_utility": float(selected.mean()) if len(selected) else 0.0,
        "minimum_leave_one_selected_scene_out_total": (
            min(leave_one_out) if leave_one_out else 0.0
        ),
    }


def select_rule(
    expected: np.ndarray,
    harm: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    fallback: np.ndarray,
) -> tuple[dict | None, np.ndarray | None]:
    return_thresholds = np.unique(np.concatenate((
        np.asarray([0.0]),
        np.quantile(expected[fallback], np.linspace(0.0, 0.90, 19)),
    )))
    harm_thresholds = np.unique(
        np.quantile(harm[fallback], np.linspace(0.10, 1.0, 19))
    )
    accepted = []
    for return_threshold in return_thresholds:
        if return_threshold < 0.0:
            continue
        for harm_threshold in harm_thresholds:
            mask = fallback & (expected >= return_threshold) & (
                harm <= harm_threshold
            )
            evidence = rule_evidence(mask, target, scenes)
            feasible = (
                evidence["authorized"] >= MIN_AUTHORIZED
                and evidence["total_utility"] > 0.0
                and evidence["catastrophic"] == 0
                and evidence["minimum_leave_one_selected_scene_out_total"] > 0.0
            )
            if feasible:
                accepted.append(({
                    "l2": L2,
                    "return_threshold": float(return_threshold),
                    "harm_probability_threshold": float(harm_threshold),
                    "feasible": True,
                    **evidence,
                }, mask))
    if not accepted:
        return None, None
    return max(
        accepted,
        key=lambda item: (
            item[0]["total_utility"],
            -item[0]["authorized"],
            item[0]["return_threshold"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit",))
    parser.parse_args()
    if OUT.exists():
        raise RuntimeError("refusing to overwrite MF3ZJ training output")
    OUT.mkdir(parents=True)
    matrix, target, scenes, sources, metadata = load()
    expected, harm, folds = crossfit(matrix, target, scenes)
    fallback = sources == "native_margin"
    selected_rule, selected_mask = select_rule(
        expected, harm, target, scenes, fallback
    )
    passed = selected_rule is not None
    result = {
        "schema_version": "revealnav-mf3zj-counterfactual-transfer-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "task_metric_run_authorized": passed,
        "feature_names": list(TRANSFER_FEATURE_NAMES),
        "folds": folds,
        "controls": {
            "pooled_rows": len(target),
            "learned_uad_rows": int((sources == "learned_uad").sum()),
            "native_margin_rows": int(fallback.sum()),
            "scenes": int(len(set(scenes))),
            "scene_folds": 5,
            "scene_overlap_all_folds": 0,
            "bootstraps_per_fit": BOOTSTRAPS,
            "l2": L2,
            "return_target_clip": [-RETURN_CLIP, RETURN_CLIP],
            "harm_label_threshold": HARM_THRESHOLD,
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "proposal_source_is_model_input": False,
            "unseen_or_test_read": False,
        },
        "sources": {
            "learned_direct_switch_manifest": sha256_file(LEARNED_SOURCE),
            "fallback_direct_switch_manifest": sha256_file(FALLBACK_SOURCE),
            "design": sha256_file(DESIGN),
        },
    }
    if passed:
        assert selected_mask is not None
        result["selected_rule"] = selected_rule
        result["oof_fallback_rows"] = [
            {
                **metadata[index],
                "target_utility": float(target[index]),
                "robust_expected_utility": float(expected[index]),
                "upper_harm_probability": float(harm[index]),
                "authorized": bool(selected_mask[index]),
            }
            for index in np.flatnonzero(fallback)
        ]
        models = bootstrap_fit(matrix, target, scenes, SEED + 9000)
        payload = {
            "means": np.stack([model[0] for model in models]),
            "scales": np.stack([model[1] for model in models]),
            "return_coefficients": np.stack([model[2] for model in models]),
            "harm_coefficients": np.stack([model[3] for model in models]),
            "feature_names": np.asarray(TRANSFER_FEATURE_NAMES),
        }
        part = MODEL.with_name(MODEL.name + ".part")
        with part.open("wb") as stream:
            np.savez(stream, **payload)
        os.replace(part, MODEL)
        result["model"] = {
            "path": str(MODEL.relative_to(ROOT)),
            "bytes": MODEL.stat().st_size,
            "sha256": sha256_file(MODEL),
            "members": len(models),
        }
    atomic_json(GATE, result)
    print(json.dumps({
        "status": result["status"],
        "selected_rule": result.get("selected_rule"),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
