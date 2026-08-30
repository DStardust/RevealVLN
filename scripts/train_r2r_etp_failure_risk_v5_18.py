#!/usr/bin/env python3
"""Build and train the train-only V5.18 frozen-ETP failure-risk gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from revealnav_failure_risk import ETPFailureRiskHead  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
SOURCE = ROOT / "artifacts/phase1/r2r_train_policy_calibration_v5_15"
RUNS = SOURCE / "runs"
MANIFEST = SOURCE / "labels/R2R_TRAIN_NET_ADVANTAGE_MANIFEST.json"
DEFAULT_OUT = ROOT / "artifacts/phase1/r2r_etp_failure_risk_v5_18"
EVENT = re.compile(r"^r2r_ep(.+)_seed(\d+)_s\d+_p\d+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(part, path)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[labels.astype(bool)].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _partition(scene: str, manifest: dict) -> str:
    matches = [
        name for name in ("train", "calibration", "dev")
        if scene in manifest[f"{name}_scenes"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"scene partition is not unique: {scene}")
    return matches[0]


def build_dataset(output: Path) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    manifest = json.loads(MANIFEST.read_text())
    if not (
        manifest.get("status")
        == "R2R_TRAIN_POLICY_INDUCED_NET_ADVANTAGE_DATASET_READY"
        and manifest.get("source_feature_events") == 591
        and manifest.get("unseen_or_test_read") is False
    ):
        raise RuntimeError("V5.15 train feature provenance is not ready")
    event_ids = {
        row["event_id"] for row in manifest["records"]
    } | {
        row["event_id"] for row in manifest["unreachable_rows"]
    }
    if len(event_ids) != manifest["source_feature_events"]:
        raise RuntimeError("V5.15 event inventory is not one-to-one")
    arrays: dict[str, list] = {
        key: [] for key in (
            "instruction", "current_history", "temporal_history", "native",
            "alternative", "immediate_costs", "failure", "group_weight",
        )
    }
    records = []
    pending_weights = Counter()
    for event_id in sorted(event_ids):
        match = EVENT.fullmatch(event_id)
        if match is None:
            raise RuntimeError(f"unexpected event id: {event_id}")
        episode, seed_text = match.groups()
        seed = int(seed_text)
        run = RUNS / f"seed_{seed}" / f"ep_{episode}"
        summary = json.loads((run / "RUN_SUMMARY.json").read_text())
        events = [
            row for row in summary["feature_events"]
            if row["event_id"] == event_id
        ]
        stats_paths = list((run / "etp_output").rglob("stats_ckpt_270_train.json"))
        if not (
            summary.get("status") == "PASS"
            and summary.get("split") == "train"
            and summary.get("native_action_overridden") is False
            and summary.get("unseen_or_test_read") is False
            and len(events) == 1
            and len(stats_paths) == 1
        ):
            raise RuntimeError(f"invalid train run closure: {event_id}")
        event = events[0]
        feature_path = ROOT / event["feature_path"]
        if (
            feature_path.is_symlink()
            or not feature_path.is_file()
            or feature_path.stat().st_size != event["feature_bytes"]
            or sha256_file(feature_path) != event["feature_sha256"]
        ):
            raise RuntimeError(f"feature provenance drift: {event_id}")
        stats = json.loads(stats_paths[0].read_text())
        success = float(stats["success"])
        if success not in (0.0, 1.0):
            raise RuntimeError("frozen ETP success label is not binary")
        branches = event["candidate_branch_ids"]
        native_id = event["native_branch_id"]
        alternatives = [value for value in branches if value != native_id]
        if len(branches) != 2 or len(alternatives) != 1:
            raise RuntimeError("failure-risk event is not a causal pair")
        alternative_id = alternatives[0]
        native_index = branches.index(native_id)
        alternative_index = branches.index(alternative_id)
        with np.load(feature_path, allow_pickle=False) as feature:
            instruction = feature["instruction_embedding"].astype(np.float32)
            history = feature["history_embeddings"].astype(np.float32)
            candidates = feature["candidate_embeddings"].astype(np.float32)
            mask = feature["candidate_mask"].astype(bool)
        if not bool(mask[-1, [native_index, alternative_index]].all()):
            raise RuntimeError("current failure-risk embeddings are incomplete")
        checkpoint = np.asarray(event["checkpoint_position"], dtype=np.float32)
        distances = []
        for branch in (native_id, alternative_id):
            position = np.asarray(event["candidate_positions"][branch], dtype=np.float32)
            distances.append(float(np.linalg.norm(position - checkpoint)) / 10.0)
        group = f"{seed}:{episode}"
        partition = _partition(event["scene_id"], manifest)
        pending_weights[group] += 1
        arrays["instruction"].append(instruction)
        arrays["current_history"].append(history[-1])
        arrays["temporal_history"].append(history.mean(0))
        arrays["native"].append(candidates[-1, native_index])
        arrays["alternative"].append(candidates[-1, alternative_index])
        arrays["immediate_costs"].append(distances)
        arrays["failure"].append(1.0 - success)
        arrays["group_weight"].append(0.0)
        records.append({
            "row_index": len(records),
            "event_id": event_id,
            "episode_id": episode,
            "controller_seed": seed,
            "scene_id": event["scene_id"],
            "partition": partition,
            "group_id": group,
            "navigation_step": event["navigation_step"],
            "failure": bool(1.0 - success),
            "native_branch_id": native_id,
            "alternative_branch_id": alternative_id,
        })
    for row in records:
        arrays["group_weight"][row["row_index"]] = 1.0 / pending_weights[row["group_id"]]
    tensor_arrays = {
        key: np.asarray(value, dtype=(
            np.float16 if key in {
                "instruction", "current_history", "temporal_history", "native",
                "alternative",
            } else np.float32
        ))
        for key, value in arrays.items()
    }
    dataset_path = output / "R2R_ETP_FAILURE_RISK_DATASET.npz"
    atomic_npz(dataset_path, tensor_arrays)
    counts = {
        split: {
            "events": sum(row["partition"] == split for row in records),
            "groups": len({
                row["group_id"] for row in records if row["partition"] == split
            }),
            "failed_events": sum(
                row["partition"] == split and row["failure"] for row in records
            ),
            "failed_groups": len({
                row["group_id"] for row in records
                if row["partition"] == split and row["failure"]
            }),
        }
        for split in ("train", "calibration", "dev")
    }
    value = {
        "schema_version": "revealnav-r2r-etp-failure-risk-dataset/1",
        "status": "R2R_TRAIN_ONLY_FAILURE_RISK_DATASET_READY",
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256_file(MANIFEST),
        "events": len(records),
        "groups": len({row["group_id"] for row in records}),
        "counts": counts,
        "label": "1 - final frozen ETP-R1 train episode success",
        "group_weight": "inverse number of events in the seed-by-episode run",
        "records": records,
        "arrays": {
            "path": str(dataset_path.relative_to(ROOT)),
            "bytes": dataset_path.stat().st_size,
            "sha256": sha256_file(dataset_path),
            "shapes": {key: list(value.shape) for key, value in tensor_arrays.items()},
        },
        "causal_inputs_only": True,
        "scene_disjoint_partitions": True,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    atomic_json(output / "R2R_ETP_FAILURE_RISK_DATASET.json", value)
    return records, tensor_arrays, value


def _batch(arrays: dict[str, np.ndarray], indices: np.ndarray, device) -> dict:
    keys = (
        "instruction", "current_history", "temporal_history", "native",
        "alternative", "immediate_costs", "failure", "group_weight",
    )
    return {
        key: torch.from_numpy(arrays[key][indices].astype(np.float32)).to(device)
        for key in keys
    }


def _forward(model: ETPFailureRiskHead, batch: dict) -> torch.Tensor:
    return model(
        batch["instruction"], batch["current_history"],
        batch["temporal_history"], batch["native"], batch["alternative"],
        batch["immediate_costs"],
    )


def _group_scores(records: list[dict], scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        grouped.setdefault(row["group_id"], []).append(index)
    ids = sorted(grouped)
    labels = np.asarray([
        records[grouped[group][0]]["failure"] for group in ids
    ], dtype=np.float32)
    values = np.asarray([
        max(float(scores[index]) for index in grouped[group]) for group in ids
    ], dtype=np.float64)
    return labels, values, ids


def _policy(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores > threshold
    positive = labels.astype(bool)
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    return {
        "groups": len(labels), "activated": int(predicted.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tp / max(1, tp + fn),
        "fpr": fp / max(1, fp + tn),
        "precision": tp / max(1, tp + fp),
        "activation_rate": float(predicted.mean()),
    }


def _threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    candidates = [float(scores.max()) + 1e-6, *sorted(set(map(float, scores)))]
    feasible = []
    for threshold in candidates:
        policy = _policy(labels, scores, threshold)
        if policy["fpr"] <= 0.10:
            feasible.append((
                policy["tpr"], policy["precision"],
                -policy["activation_rate"], threshold, policy,
            ))
    if not feasible:
        raise RuntimeError("no calibration threshold satisfies the FPR budget")
    _, _, _, threshold, policy = max(feasible, key=lambda row: row[:4])
    return threshold, policy


def train(records, arrays, output: Path, device: torch.device) -> dict:
    indices = {
        split: np.asarray([
            row["row_index"] for row in records if row["partition"] == split
        ], dtype=np.int64)
        for split in ("train", "calibration", "dev")
    }
    batches = {split: _batch(arrays, value, device) for split, value in indices.items()}
    states = []
    members = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = ETPFailureRiskHead().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        train_batch = batches["train"]
        labels = train_batch["failure"]
        group_weight = train_batch["group_weight"]
        weighted_positive = float((group_weight * labels).sum())
        weighted_negative = float((group_weight * (1.0 - labels)).sum())
        pos_weight = torch.tensor(
            weighted_negative / max(weighted_positive, 1e-6), device=device
        )
        generator = torch.Generator().manual_seed(seed)
        for _ in range(80):
            model.train()
            order = torch.randperm(len(labels), generator=generator)
            for start in range(0, len(order), 64):
                local = order[start:start + 64].to(device)
                logits = _forward(model, {
                    key: value[local] for key, value in train_batch.items()
                })
                losses = F.binary_cross_entropy_with_logits(
                    logits, labels[local], pos_weight=pos_weight,
                    reduction="none",
                )
                loss = (losses * group_weight[local]).sum() / group_weight[local].sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        model.eval()
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        member = {"seed": seed}
        with torch.no_grad():
            for split in indices:
                probability = torch.sigmoid(_forward(model, batches[split])).cpu().numpy()
                labels_g, scores_g, _ = _group_scores(
                    [records[index] for index in indices[split]], probability
                )
                member[f"{split}_group_auc"] = auc(labels_g, scores_g)
        members.append(member)
    models = []
    for state in states:
        model = ETPFailureRiskHead().to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    predictions = {}
    group_values = {}
    for split in indices:
        with torch.no_grad():
            probability = torch.stack([
                torch.sigmoid(_forward(model, batches[split])) for model in models
            ]).mean(0).cpu().numpy()
        labels_g, scores_g, group_ids = _group_scores(
            [records[index] for index in indices[split]], probability
        )
        predictions[split] = probability
        group_values[split] = (labels_g, scores_g, group_ids)
    threshold, calibration_policy = _threshold(*group_values["calibration"][:2])
    policies = {
        split: _policy(group_values[split][0], group_values[split][1], threshold)
        for split in group_values
    }
    group_auc = {
        split: auc(group_values[split][0], group_values[split][1])
        for split in group_values
    }
    checkpoint = output / "etp_failure_risk_ensemble_v5_18.pt"
    payload = {
        "schema_version": "revealnav-etp-failure-risk-ensemble/1",
        "member_seeds": list(SEEDS),
        "model_state_dicts": states,
        "aggregation": "mean_failure_probability",
        "threshold": threshold,
        "threshold_rule": (
            "maximize calibration group failure recall subject to frozen-ETP "
            "successful-group false-positive rate <= 0.10; tie-break by "
            "precision, sparsity, then higher threshold"
        ),
        "input_dim": 768,
        "projection_dim": 64,
        "immediate_cost_scale_m": 10.0,
        "label": "1 - final frozen ETP-R1 train episode success",
    }
    part = checkpoint.with_name(checkpoint.name + ".part")
    torch.save(payload, part)
    os.replace(part, checkpoint)
    calibration_failures = int(group_values["calibration"][0].sum())
    dev_failures = int(group_values["dev"][0].sum())
    gates = {
        "three_members": len(states) == 3,
        "calibration_contains_both_classes": 0 < calibration_failures < len(group_values["calibration"][0]),
        "dev_contains_both_classes": 0 < dev_failures < len(group_values["dev"][0]),
        "calibration_fpr_at_most_0_10": calibration_policy["fpr"] <= 0.10,
        "calibration_detects_failure": calibration_policy["tp"] > 0,
        "dev_auc_above_random": group_auc["dev"] > 0.5,
        "dev_detects_failure": policies["dev"]["tp"] > 0,
        "dev_fpr_at_most_0_20": policies["dev"]["fpr"] <= 0.20,
    }
    result = {
        "schema_version": "revealnav-r2r-etp-failure-risk-training/1",
        "status": "R2R_ETP_FAILURE_RISK_PASS" if all(gates.values()) else "R2R_ETP_FAILURE_RISK_FAIL",
        "fixed_epochs": 80,
        "members": members,
        "ensemble_group_auc": group_auc,
        "threshold": threshold,
        "policies": policies,
        "gates": gates,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "calibration_selected_threshold": True,
        "dev_used_for_threshold_or_training": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    atomic_json(output / "R2R_ETP_FAILURE_RISK_TRAINING.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT not in output.parents or output.exists():
        raise SystemExit("output directory must be new and inside the project")
    output.mkdir(parents=True)
    records, arrays, dataset = build_dataset(output)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = train(records, arrays, output, device)
    print(json.dumps({
        "dataset": dataset["counts"],
        "status": result["status"],
        "auc": result["ensemble_group_auc"],
        "threshold": result["threshold"],
        "policies": result["policies"],
        "gates": result["gates"],
    }, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
