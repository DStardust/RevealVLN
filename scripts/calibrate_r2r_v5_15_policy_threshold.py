#!/usr/bin/env python3
"""Calibrate the frozen Net-Advantage ensemble on R2R-train V5.6 proposals."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import train_r2r_sparse_net_advantage as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENSEMBLE_RESULT = ROOT / (
    "artifacts/phase1/r2r_train_net_advantage/full/training_v5_14/"
    "R2R_SPARSE_NET_ADVANTAGE_TRAINING_RESULT.json"
)
SELECTION = ROOT / (
    "artifacts/phase1/r2r_train_policy_calibration_v5_15/"
    "R2R_TRAIN_V5_15_POLICY_SELECTION.json"
)
PROGRESS = ROOT / (
    "artifacts/phase1/r2r_train_policy_calibration_v5_15/"
    "R2R_TRAIN_V5_15_POLICY_PROGRESS.json"
)
SEEDS = (20260826, 20260827, 20260828)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def policy_without_indices(records: list[dict], scores: np.ndarray, threshold: float) -> dict:
    return {
        key: value for key, value in training.event_policy(
            records, scores, threshold
        ).items() if key != "selected_indices"
    }


def select_threshold(records: list[dict], scores: np.ndarray) -> tuple[float, dict]:
    finite = scores[np.isfinite(scores)]
    if len(finite) != len(scores) or len(finite) < 5:
        raise RuntimeError("insufficient finite policy calibration scores")
    thresholds = np.concatenate((
        [np.nextafter(float(finite.min()), -math.inf)], np.unique(finite)
    ))
    candidates = []
    for threshold in thresholds:
        policy = policy_without_indices(records, scores, float(threshold))
        if policy["activated"] < 5 or policy["activation_rate"] > 0.10:
            continue
        candidates.append((
            policy["mean_net_per_event_m"], policy["positive_precision"],
            -policy["activation_rate"], float(threshold), policy,
        ))
    if not candidates:
        raise RuntimeError("no policy threshold satisfies frozen coverage constraints")
    _, _, _, threshold, policy = max(candidates, key=lambda row: row[:3])
    return threshold, policy


def score_quantiles(scores: np.ndarray) -> dict[str, float]:
    return {
        str(q): float(np.quantile(scores, q))
        for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ensemble-result", type=Path, default=DEFAULT_ENSEMBLE_RESULT,
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    ensemble_result_path = args.ensemble_result.resolve()
    if any(
        ROOT not in path.parents
        for path in (manifest_path, output_dir, ensemble_result_path)
    ):
        raise SystemExit("calibration paths must remain inside the project")
    manifest = load(manifest_path)
    selection = load(SELECTION)
    progress = load(PROGRESS)
    ensemble_result = load(ensemble_result_path)
    if (
        manifest.get("status")
        != "R2R_TRAIN_POLICY_INDUCED_NET_ADVANTAGE_DATASET_READY"
        or manifest.get("schema_version")
        != "revealnav-r2r-train-policy-induced-net-advantage-dataset/1"
        or manifest.get("controller_seeds") != list(SEEDS)
        or manifest.get("unseen_or_test_read") is not False
        or manifest.get("task_metric_payload_read") is not False
        or selection.get("selected_runs") != 10809
        or len({row["scene_id"] for row in selection["episodes"]}) != 61
        or progress.get("completed") != 10809
        or progress.get("exhausted_failures")
        or ensemble_result.get("status")
        != "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_PASS"
        or ensemble_result.get("unseen_or_test_read") is not False
    ):
        raise RuntimeError("V5.15 calibration provenance gate failed")
    array_path = (ROOT / manifest["arrays"]["path"]).resolve()
    if (
        ROOT not in array_path.parents or array_path.is_symlink()
        or not array_path.is_file()
        or array_path.stat().st_size != manifest["arrays"]["bytes"]
        or training.sha256_file(array_path) != manifest["arrays"]["sha256"]
    ):
        raise RuntimeError("policy-induced calibration array provenance drift")
    with np.load(array_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    records = manifest["records"]
    if len(records) != len(arrays["better"]):
        raise RuntimeError("policy calibration record alignment drift")
    deployment = ensemble_result["deployment_checkpoint"]
    checkpoint_path = (ROOT / deployment["path"]).resolve()
    if (
        ROOT not in checkpoint_path.parents or checkpoint_path.is_symlink()
        or checkpoint_path.stat().st_size != deployment["bytes"]
        or training.sha256_file(checkpoint_path) != deployment["sha256"]
    ):
        raise RuntimeError("frozen ensemble checkpoint provenance drift")
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        payload.get("schema_version")
        != "revealnav-pairwise-net-advantage-ensemble/1"
        or tuple(payload.get("member_seeds", ())) != SEEDS
        or payload.get("aggregation")
        != "mean_probability_and_mean_positive_gain"
    ):
        raise RuntimeError("frozen ensemble payload drift")
    models = []
    for state in payload["model_state_dicts"]:
        model = training.PairwiseNetAdvantageHead(
            payload["input_dim"], payload["projection_dim"]
        ).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    indices = {
        partition: np.asarray([
            row["row_index"] for row in records
            if row["partition"] == partition
        ], dtype=np.int64)
        for partition in ("train", "calibration", "dev")
    }
    predictions = {}
    partition_records = {}
    for partition, local_indices in indices.items():
        if not len(local_indices):
            raise RuntimeError(f"empty policy calibration partition: {partition}")
        batch = training.tensors(arrays, local_indices, device)
        predictions[partition] = training.predict_ensemble(
            models, batch, arrays, local_indices
        )
        partition_records[partition] = [records[index] for index in local_indices]
    selection_error = None
    try:
        threshold, calibration_policy = select_threshold(
            partition_records["calibration"], predictions["calibration"][2]
        )
    except RuntimeError as error:
        selection_error = str(error)
        threshold = float(np.max(predictions["calibration"][2]))
        calibration_policy = policy_without_indices(
            partition_records["calibration"],
            predictions["calibration"][2], threshold,
        )
    policies = {
        "train": policy_without_indices(
            partition_records["train"], predictions["train"][2], threshold
        ),
        "calibration": calibration_policy,
        "dev": policy_without_indices(
            partition_records["dev"], predictions["dev"][2], threshold
        ),
    }
    positives = {
        partition: int(arrays["better"][local_indices].sum())
        for partition, local_indices in indices.items()
    }
    finite = all(
        all(np.isfinite(value).all() for value in prediction)
        for prediction in predictions.values()
    )
    gates = {
        "at_least_300_policy_rows": len(records) >= 300,
        "all_61_train_scenes_in_run_coverage": (
            len({row["scene_id"] for row in selection["episodes"]}) == 61
        ),
        "calibration_and_dev_each_have_five_positives": (
            positives["calibration"] >= 5 and positives["dev"] >= 5
        ),
        "all_scores_finite": finite,
        "calibration_activations_at_least_five": (
            policies["calibration"]["activated"] >= 5
        ),
        "calibration_sparse_net_positive": (
            policies["calibration"]["mean_net_per_event_m"] > 0
        ),
        "calibration_activation_rate_bounded": (
            policies["calibration"]["activation_rate"] <= 0.10
        ),
        "internal_dev_activations_at_least_five": (
            policies["dev"]["activated"] >= 5
        ),
        "internal_dev_sparse_net_positive": (
            policies["dev"]["mean_net_per_event_m"] > 0
        ),
        "internal_dev_positive_precision_above_half": (
            policies["dev"]["positive_precision"] > 0.5
        ),
        "internal_dev_activation_rate_bounded": (
            policies["dev"]["activation_rate"] <= 0.10
        ),
    }
    calibrated = dict(payload)
    calibrated.update({
        "calibrated_score_threshold": threshold,
        "policy_calibration": {
            "version": "V5.15",
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": training.sha256_file(manifest_path),
            "selection": str(SELECTION.relative_to(ROOT)),
            "selection_sha256": training.sha256_file(SELECTION),
            "threshold_partition": "R2R-train policy-induced calibration scenes",
        },
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "sparse_net_advantage_v5_15_policy_calibrated.pt"
    part = checkpoint.with_name(checkpoint.name + ".part")
    torch.save(calibrated, part)
    os.replace(part, checkpoint)
    value = {
        "schema_version": "revealnav-r2r-v5.15-policy-calibration-result/1",
        "status": (
            "R2R_V5_15_POLICY_CALIBRATION_PASS"
            if all(gates.values()) else "R2R_V5_15_POLICY_CALIBRATION_FAIL"
        ),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": training.sha256_file(manifest_path),
        "frozen_ensemble_result": str(ensemble_result_path.relative_to(ROOT)),
        "frozen_ensemble_result_sha256": training.sha256_file(
            ensemble_result_path
        ),
        "threshold": threshold,
        "threshold_selection_error": selection_error,
        "threshold_selection_partition": "calibration",
        "policies": policies,
        "positive_rows": positives,
        "score_quantiles": {
            partition: score_quantiles(prediction[2])
            for partition, prediction in predictions.items()
        },
        "gates": gates,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": training.sha256_file(checkpoint),
            "member_seeds": list(SEEDS),
        },
        "candidate_ensemble_weights_changed": False,
        "task_metric_payload_read": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    result_path = output_dir / "R2R_V5_15_POLICY_CALIBRATION_RESULT.json"
    atomic_json(result_path, value)
    print(json.dumps({
        "status": value["status"], "threshold": threshold, "gates": gates,
        "calibration_policy": policies["calibration"],
        "dev_policy": policies["dev"],
    }, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
