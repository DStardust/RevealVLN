#!/usr/bin/env python3
"""Train and calibrate the sparse R2R net-advantage trigger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_net_advantage import PairwiseNetAdvantageHead  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)


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


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return float((ranks[labels.astype(bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def event_policy(
    records: list[dict], scores: np.ndarray, threshold: float,
) -> dict:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        grouped.setdefault(row["event_id"], []).append(index)
    selected = []
    for event_id, indices in grouped.items():
        best = max(indices, key=lambda index: (float(scores[index]), -index))
        if float(scores[best]) > threshold:
            selected.append(best)
    realized = np.asarray([
        records[index]["realized_trial_net_m"] for index in selected
    ], dtype=np.float64)
    positives = np.asarray([
        records[index]["better_by_margin"] for index in selected
    ], dtype=bool)
    return {
        "events": len(grouped),
        "activated": len(selected),
        "activation_rate": len(selected) / max(1, len(grouped)),
        "mean_net_per_event_m": float(realized.sum() / max(1, len(grouped))),
        "mean_net_per_activation_m": float(realized.mean()) if len(realized) else 0.0,
        "positive_precision": float(positives.mean()) if len(positives) else 0.0,
        "selected_indices": selected,
    }


def calibrate(records: list[dict], scores: np.ndarray) -> tuple[float, dict]:
    candidates = sorted(set(float(value) for value in scores))
    candidates = [min(candidates) - 1e-6, *candidates]
    rows = []
    for threshold in candidates:
        policy = event_policy(records, scores, threshold)
        if policy["activation_rate"] <= 0.20:
            rows.append((policy["mean_net_per_event_m"], policy["positive_precision"], -policy["activation_rate"], threshold, policy))
    if not rows:
        raise RuntimeError("no sparse calibration threshold")
    _, _, _, threshold, policy = max(rows, key=lambda row: row[:3])
    return threshold, policy


def tensors(arrays, indices, device):
    result = {}
    for key in (
        "instruction", "current_history", "temporal_history", "native", "alternative",
    ):
        result[key] = torch.from_numpy(arrays[key][indices].astype(np.float32)).to(device)
    result["immediate_costs"] = torch.from_numpy(
        arrays["immediate_costs"][indices].astype(np.float32) / 10.0
    ).to(device)
    result["better"] = torch.from_numpy(arrays["better"][indices].astype(np.float32)).to(device)
    result["gain"] = torch.from_numpy(arrays["positive_gain"][indices].astype(np.float32) / 10.0).to(device)
    return result


def predict(model, batch, arrays, indices) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits, gain = model(
            batch["instruction"], batch["current_history"], batch["temporal_history"],
            batch["native"], batch["alternative"], batch["immediate_costs"],
        )
    probability = torch.sigmoid(logits).cpu().numpy()
    gain_m = gain.cpu().numpy() * 10.0
    penalty = arrays["round_trip_cost"][indices].astype(np.float32)
    score = probability * gain_m - (1.0 - probability) * penalty
    return probability, gain_m, score


def one_seed(
    seed: int, arrays, records: list[dict], train_indices: np.ndarray,
    dev_indices: np.ndarray, output_dir: Path, device: torch.device,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = PairwiseNetAdvantageHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    train = tensors(arrays, train_indices, device)
    dev = tensors(arrays, dev_indices, device)
    positives = float(train["better"].sum())
    negatives = float(len(train_indices) - positives)
    pos_weight = torch.tensor(negatives / max(1.0, positives), device=device)
    generator = torch.Generator().manual_seed(seed)
    best = None
    patience = 0
    for epoch in range(160):
        model.train()
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), 64):
            local = order[start:start + 64].to(device)
            logits, gain = model(
                train["instruction"][local], train["current_history"][local],
                train["temporal_history"][local], train["native"][local],
                train["alternative"][local], train["immediate_costs"][local],
            )
            labels = train["better"][local]
            classification = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
            positive = labels > 0.5
            regression = (
                F.smooth_l1_loss(gain[positive], train["gain"][local][positive])
                if bool(positive.any()) else logits.sum() * 0.0
            )
            loss = classification + 0.5 * regression
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, gain = model(
                dev["instruction"], dev["current_history"], dev["temporal_history"],
                dev["native"], dev["alternative"], dev["immediate_costs"],
            )
            val = float(F.binary_cross_entropy_with_logits(logits, dev["better"]))
            positive = dev["better"] > 0.5
            if bool(positive.any()):
                val += 0.5 * float(F.smooth_l1_loss(gain[positive], dev["gain"][positive]))
        if best is None or val < best[0] - 1e-5:
            best = (val, epoch, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if patience >= 20:
            break
    model.load_state_dict(best[2], strict=True)
    model.eval()
    dev_probability, dev_gain, dev_score = predict(model, dev, arrays, dev_indices)
    dev_records = [records[index] for index in dev_indices]
    threshold, policy = calibrate(dev_records, dev_score)
    train_probability, train_gain, train_score = predict(model, train, arrays, train_indices)
    train_records = [records[index] for index in train_indices]
    checkpoint = output_dir / f"seed_{seed}" / "sparse_net_advantage.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "revealnav-pairwise-net-advantage-checkpoint/1",
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "calibrated_score_threshold": threshold,
        "score_definition": "p_better*positive_gain-(1-p_better)*round_trip_cost",
        "immediate_cost_scale_m": 10.0,
        "input_dim": 768,
        "projection_dim": 96,
    }
    part = checkpoint.with_name(checkpoint.name + ".part")
    torch.save(payload, part)
    os.replace(part, checkpoint)
    result = {
        "seed": seed,
        "best_epoch": best[1],
        "train_rows": len(train_indices),
        "dev_rows": len(dev_indices),
        "train_auc": auc(arrays["better"][train_indices], train_probability),
        "dev_auc": auc(arrays["better"][dev_indices], dev_probability),
        "calibrated_score_threshold": threshold,
        "train_policy": {key: value for key, value in event_policy(train_records, train_score, threshold).items() if key != "selected_indices"},
        "dev_policy": {key: value for key, value in policy.items() if key != "selected_indices"},
        "dev_predicted_gain_mean_m": float(dev_gain.mean()),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if ROOT not in manifest_path.parents or ROOT not in output_dir.parents:
        raise SystemExit("training paths must remain inside the project")
    manifest = json.loads(manifest_path.read_text())
    array_path = (ROOT / manifest["arrays"]["path"]).resolve()
    if (
        array_path.is_symlink() or array_path.stat().st_size != manifest["arrays"]["bytes"]
        or sha256_file(array_path) != manifest["arrays"]["sha256"]
    ):
        raise RuntimeError("training array provenance drift")
    with np.load(array_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    records = manifest["records"]
    if len(records) != len(arrays["better"]):
        raise RuntimeError("record/array alignment drift")
    train_indices = np.asarray([row["row_index"] for row in records if row["partition"] == "train"], dtype=np.int64)
    dev_indices = np.asarray([row["row_index"] for row in records if row["partition"] == "dev"], dtype=np.int64)
    if len(train_indices) < 16 or len(dev_indices) < 8:
        raise RuntimeError("insufficient scene-disjoint rows for learnability gate")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = [one_seed(
        seed, arrays, records, train_indices, dev_indices, output_dir, device
    ) for seed in SEEDS]
    finite_auc = [row["dev_auc"] for row in results if math.isfinite(row["dev_auc"])]
    gates = {
        "all_three_seeds_finite": len(finite_auc) == 3,
        "median_dev_auc_above_chance": len(finite_auc) == 3 and float(np.median(finite_auc)) >= 0.55,
        "mean_sparse_net_positive": float(np.mean([row["dev_policy"]["mean_net_per_event_m"] for row in results])) > 0.0,
        "mean_sparse_precision_above_half": float(np.mean([row["dev_policy"]["positive_precision"] for row in results])) > 0.5,
        "all_activation_rates_bounded": all(row["dev_policy"]["activation_rate"] <= 0.20 for row in results),
    }
    value = {
        "schema_version": "revealnav-r2r-sparse-net-advantage-training/1",
        "status": "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_PASS" if all(gates.values()) else "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_FAIL",
        "dataset_manifest": str(manifest_path.relative_to(ROOT)),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "train_scenes": manifest["train_scenes"],
        "dev_scenes": manifest["dev_scenes"],
        "results": results,
        "gates": gates,
        "selection_rule": "highest calibrated dev mean net, tie broken by AUC then seed",
        "selected_seed": max(results, key=lambda row: (
            row["dev_policy"]["mean_net_per_event_m"], row["dev_auc"], -row["seed"]
        ))["seed"],
        "task_metric_payload_read": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    result_path = output_dir / "R2R_SPARSE_NET_ADVANTAGE_TRAINING_RESULT.json"
    atomic_json(result_path, value)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "selected_seed": value["selected_seed"],
        "dev_auc": [row["dev_auc"] for row in results],
        "dev_net": [row["dev_policy"]["mean_net_per_event_m"] for row in results],
    }, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
