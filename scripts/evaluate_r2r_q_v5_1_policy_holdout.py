#!/usr/bin/env python3
"""Seal and evaluate expanded Q on R2R-train policy-induced proposals."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r4 import BranchExcursionQHead  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, rank_auc, sha256_file  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
TRAINING = ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v5_1"
COMPARISON = TRAINING / "RXR_BRANCH_EXCURSION_Q_COMPARISON_V5_1.json"
SOURCE = ROOT / (
    "artifacts/phase1/r2r_train_policy_calibration_v5_15/labels/"
    "R2R_TRAIN_NET_ADVANTAGE_MANIFEST.json"
)
RUNS = ROOT / "artifacts/phase1/r2r_train_policy_calibration_v5_15/runs"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_q_v5_1_policy_holdout"
PROTOCOL = OUT / "R2R_Q_V5_1_POLICY_HOLDOUT_PROTOCOL.json"
RESULT = OUT / "R2R_Q_V5_1_POLICY_HOLDOUT_RESULT.json"


def checkpoints() -> list[dict]:
    comparison = json.loads(COMPARISON.read_text())
    if not (
        comparison.get("status") == "BRANCH_EXCURSION_Q_OFFLINE_GATE_PASS"
        and comparison.get("passing_variants") == ["source_balanced"]
    ):
        raise RuntimeError("source-balanced expanded Q did not pass offline gate")
    rows = []
    source = comparison["variants"]["source_balanced"]["runs"]
    for seed in SEEDS:
        checkpoint = source[str(seed)]["checkpoint"]
        path = (ROOT / checkpoint["path"]).resolve()
        if (
            ROOT not in path.parents
            or path.is_symlink()
            or path.stat().st_size != checkpoint["bytes"]
            or sha256_file(path) != checkpoint["sha256"]
        ):
            raise RuntimeError("expanded Q checkpoint provenance drift")
        rows.append({"seed": seed, **checkpoint})
    return rows


def source_manifest() -> dict:
    value = json.loads(SOURCE.read_text())
    records = value.get("records", [])
    if not (
        value.get("status")
        == "R2R_TRAIN_POLICY_INDUCED_NET_ADVANTAGE_DATASET_READY"
        and value.get("schema_version")
        == "revealnav-r2r-train-policy-induced-net-advantage-dataset/1"
        and value.get("split") == "train_only_with_scene_disjoint_internal_partition"
        and value.get("source_policy") == "V5.6 shadow proposals"
        and value.get("task_metric_payload_read") is False
        and value.get("unseen_or_test_read") is False
        and len(records) == 576
        and len({row["event_id"] for row in records}) == 576
        and sum(row["better_by_margin"] for row in records) == 30
    ):
        raise RuntimeError("R2R policy-induced holdout source drift")
    return value


def protocol_value() -> dict:
    source = source_manifest()
    locked = checkpoints()
    return {
        "schema_version": "revealnav-r2r-q-policy-holdout-protocol/5.1",
        "status": "LOCKED_BEFORE_R2R_POLICY_HOLDOUT_Q_INFERENCE",
        "scope": {
            "dataset": "R2R train only",
            "source_policy": "V5.6 shadow proposals; native ETP action unchanged",
            "rows": 576,
            "unique_episodes": 172,
            "scenes": 48,
            "positive_rows": 30,
            "negative_rows": 546,
            "task_metrics_used": False,
        },
        "model": {
            "variant": "source_balanced",
            "checkpoints": locked,
            "ensemble": "coordinate-wise median of three Q cost predictions",
            "candidate_score": (
                "minimum predicted native action cost minus minimum predicted "
                "proposed-alternative action cost"
            ),
            "activation": "candidate_score > 0; no calibrated threshold",
        },
        "realized_reward": (
            "0 when preserving native; source realized_trial_net_m when activating"
        ),
        "regret": "max(0, realized_trial_net_m) minus selected realized reward",
        "success_gates": {
            "all_576_rows_validate": True,
            "positive_and_negative_rows_present": True,
            "score_auc_above_0_5": True,
            "nonzero_activations": True,
            "activation_positive_precision_above_0_5": True,
            "mean_realized_net_per_activation_positive": True,
            "mean_regret_below_always_native": True,
        },
        "sources": {
            str(SOURCE.relative_to(ROOT)): sha256_file(SOURCE),
            str(COMPARISON.relative_to(ROOT)): sha256_file(COMPARISON),
            "revealnav_mf2r4/model.py": sha256_file(
                ROOT / "revealnav_mf2r4/model.py"
            ),
            "scripts/evaluate_r2r_q_v5_1_policy_holdout.py": sha256_file(
                ROOT / "scripts/evaluate_r2r_q_v5_1_policy_holdout.py"
            ),
        },
        "val_seen_used_for_threshold_or_selection": False,
        "val_unseen_or_test_read": False,
        "gold_payload_read": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed R2R policy holdout protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "rows": value["scope"]["rows"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def load_models(device: torch.device) -> list[BranchExcursionQHead]:
    models = []
    for row in checkpoints():
        payload = torch.load(
            ROOT / row["path"], map_location="cpu", weights_only=False
        )
        if not (
            payload.get("schema_version")
            == "revealnav-mf2-branch-excursion-q-checkpoint/5.1"
            and payload.get("variant") == "source_balanced"
            and payload.get("seed") == row["seed"]
        ):
            raise RuntimeError("expanded Q checkpoint schema drift")
        model = BranchExcursionQHead(768, 96, 128.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
    return models


def source_event(record: dict) -> tuple[dict, Path]:
    summary_path = (
        RUNS / f"seed_{record['controller_seed']}" / f"ep_{record['episode_id']}"
        / "RUN_SUMMARY.json"
    )
    summary = json.loads(summary_path.read_text())
    events = [
        row for row in summary.get("feature_events", [])
        if row["event_id"] == record["event_id"]
    ]
    if not (
        summary.get("status") == "PASS"
        and summary.get("split") == "train"
        and summary.get("native_action_overridden") is False
        and summary.get("task_metric_payload_read") is False
        and len(events) == 1
    ):
        raise RuntimeError("policy-induced source event closure failure")
    event = events[0]
    if not (
        event["candidate_branch_ids"]
        == [record["native_branch_id"], record["alternative_branch_id"]]
        and event["proposed_branch_id"] == record["alternative_branch_id"]
    ):
        raise RuntimeError("policy-induced candidate order drift")
    feature = (ROOT / event["feature_path"]).resolve()
    if (
        ROOT not in feature.parents
        or feature.is_symlink()
        or feature.stat().st_size != event["feature_bytes"]
        or sha256_file(feature) != event["feature_sha256"]
    ):
        raise RuntimeError("policy-induced feature provenance drift")
    return event, feature


def predict(
    models: list[BranchExcursionQHead], feature: Path, device: torch.device,
) -> tuple[float, list[float]]:
    with np.load(feature, allow_pickle=False) as shard:
        instruction = torch.from_numpy(
            shard["instruction_embedding"].astype(np.float32)
        ).unsqueeze(0).to(device)
        history = torch.from_numpy(
            shard["history_embeddings"].astype(np.float32)
        ).unsqueeze(0).to(device)
        candidates = torch.from_numpy(
            shard["candidate_embeddings"].astype(np.float32)
        ).unsqueeze(0).to(device)
        mask = torch.from_numpy(
            shard["candidate_mask"].astype(np.bool_)
        ).unsqueeze(0).to(device)
    steps = history.shape[1]
    if candidates.shape[2] != 2 or not bool(mask[0, -1].all()):
        raise RuntimeError("policy-induced feature shape drift")
    decision = torch.tensor([steps - 1], dtype=torch.long, device=device)
    per_model = []
    with torch.no_grad():
        for model in models:
            output = model(history, candidates, mask, instruction, decision)
            action = torch.stack((output.commit_cost[0], output.excursion_cost[0]))
            candidate_cost = action.min(0).values
            per_model.append(float(candidate_cost[0] - candidate_cost[1]))
    return float(np.median(per_model)), per_model


def evaluate(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("R2R policy holdout protocol must be sealed")
    source = source_manifest()
    models = load_models(device)
    labels, scores, realized = [], [], []
    rows = []
    for record in source["records"]:
        _, feature = source_event(record)
        score, members = predict(models, feature, device)
        label = bool(record["better_by_margin"])
        reward = float(record["realized_trial_net_m"])
        activate = score > 0.0
        labels.append(int(label)); scores.append(score); realized.append(reward)
        rows.append({
            "event_id": record["event_id"],
            "episode_id": record["episode_id"],
            "scene_id": record["scene_id"],
            "label_better_by_margin": label,
            "ensemble_score": score,
            "member_scores": members,
            "activated": activate,
            "realized_reward_if_activated_m": reward,
        })
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    rewards_array = np.asarray(realized, dtype=np.float64)
    activated = scores_array > 0.0
    selected_reward = np.where(activated, rewards_array, 0.0)
    oracle_reward = np.maximum(rewards_array, 0.0)
    true_positive = int(np.sum(activated & (labels_array == 1)))
    activation_count = int(activated.sum())
    positive_count = int(labels_array.sum())
    metrics = {
        "rows": len(rows),
        "positive_rows": positive_count,
        "negative_rows": len(rows) - positive_count,
        "score_auc": rank_auc(labels_array, scores_array),
        "activations": activation_count,
        "activation_rate": activation_count / len(rows),
        "activation_positive_precision": (
            true_positive / activation_count if activation_count else 0.0
        ),
        "positive_recall": true_positive / positive_count,
        "mean_realized_net_per_activation_m": (
            float(selected_reward[activated].mean()) if activation_count else 0.0
        ),
        "mean_realized_net_per_event_m": float(selected_reward.mean()),
        "mean_regret_m": float((oracle_reward - selected_reward).mean()),
        "always_native_mean_regret_m": float(oracle_reward.mean()),
        "member_activation_counts": [
            sum(row["member_scores"][index] > 0.0 for row in rows)
            for index in range(3)
        ],
        "unanimous_member_decisions": sum(
            len({value > 0.0 for value in row["member_scores"]}) == 1
            for row in rows
        ),
    }
    gates = {
        "all_576_rows_validate": len(rows) == 576,
        "positive_and_negative_rows_present": 0 < positive_count < len(rows),
        "score_auc_above_0_5": metrics["score_auc"] > 0.5,
        "nonzero_activations": activation_count > 0,
        "activation_positive_precision_above_0_5": (
            metrics["activation_positive_precision"] > 0.5
        ),
        "mean_realized_net_per_activation_positive": (
            metrics["mean_realized_net_per_activation_m"] > 0.0
        ),
        "mean_regret_below_always_native": (
            metrics["mean_regret_m"] < metrics["always_native_mean_regret_m"]
        ),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-r2r-q-policy-holdout-result/5.1",
        "status": "R2R_Q_V5_1_POLICY_HOLDOUT_PASS" if passed
                  else "R2R_Q_V5_1_POLICY_HOLDOUT_FAIL",
        "metrics": metrics,
        "gates": gates,
        "records": rows,
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_manifest_sha256": sha256_file(SOURCE),
        "task_metric_payload_read": False,
        "val_seen_used_for_threshold_or_selection": False,
        "val_unseen_or_test_read": False,
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": "conservative controller integration" if passed
                     else "cross-dataset proposal diagnosis",
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "metrics": metrics, "gates": gates,
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    return seal() if args.seal else evaluate(torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
