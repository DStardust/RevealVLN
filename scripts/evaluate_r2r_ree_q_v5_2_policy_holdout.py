#!/usr/bin/env python3
"""Evaluate the pre-existing REE-Q fusion with expanded Q checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_r2r_q_v5_1_policy_holdout as q_holdout  # noqa: E402
from revealnav_mf2r3 import RelationalRevealExpiryHeads  # noqa: E402
from revealnav_mf2r4 import BranchExcursionQHead  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, rank_auc, sha256_file  # noqa: E402


SEEDS = q_holdout.SEEDS
Q_COMPARISON = q_holdout.COMPARISON
Q_HOLDOUT_RESULT = q_holdout.RESULT
SOURCE = q_holdout.SOURCE
OLD_LOCK = ROOT / "locks/REE_Q_FUSION_CONTROLLER_V4_4.json"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_ree_q_v5_2_policy_holdout"
PROTOCOL = OUT / "R2R_REE_Q_V5_2_POLICY_HOLDOUT_PROTOCOL.json"
RESULT = OUT / "R2R_REE_Q_V5_2_POLICY_HOLDOUT_RESULT.json"
WRONG_COMMITMENT_WEIGHT = 5.0


def checkpoint_pairs() -> list[dict]:
    old_lock = json.loads(OLD_LOCK.read_text())
    q_rows = {row["seed"]: row for row in q_holdout.checkpoints()}
    if not (
        old_lock.get("status") == "LOCKED_BEFORE_FRESH_R2R_UNSEEN_CONFIRMATION"
        and old_lock["fixed_composition"]["wrong_commitment_weight"]
        == WRONG_COMMITMENT_WEIGHT
        and old_lock["fixed_composition"]["seed_pairing"]
        == "pair REE and Q checkpoints with the same seed; report all three seeds independently"
        and old_lock["fixed_composition"]["ensemble_allowed"] is False
    ):
        raise RuntimeError("pre-existing REE-Q composition lock drift")
    pairs = []
    for row in old_lock["checkpoint_pairs"]:
        seed = row["seed"]
        ree = row["ree"]
        ree_path = (ROOT / ree["path"]).resolve()
        if (
            ROOT not in ree_path.parents
            or ree_path.is_symlink()
            or ree_path.stat().st_size != ree["bytes"]
            or sha256_file(ree_path) != ree["sha256"]
        ):
            raise RuntimeError("REE checkpoint provenance drift")
        pairs.append({"seed": seed, "ree": ree, "q": q_rows[seed]})
    if [row["seed"] for row in pairs] != list(SEEDS):
        raise RuntimeError("same-seed REE-Q triplet drift")
    return pairs


def protocol_value() -> dict:
    source = q_holdout.source_manifest()
    failed_q = json.loads(Q_HOLDOUT_RESULT.read_text())
    if failed_q.get("status") != "R2R_Q_V5_1_POLICY_HOLDOUT_FAIL":
        raise RuntimeError("Q-only failure evidence is absent")
    return {
        "schema_version": "revealnav-r2r-ree-q-policy-holdout-protocol/5.2",
        "status": "LOCKED_BEFORE_FIXED_REE_Q_POLICY_HOLDOUT_INFERENCE",
        "motivation": (
            "Q-only cross-dataset ranking failed; test the already-frozen online "
            "composition rather than tune a new threshold"
        ),
        "scope": {
            "dataset": "R2R train only",
            "rows": len(source["records"]),
            "positive_rows": source["positive_rows"],
            "negative_rows": source["negative_rows"],
            "native_action_overridden_during_collection": False,
        },
        "composition": {
            "seed_pairing": "same-seed REE and source-balanced expanded Q",
            "pairs": checkpoint_pairs(),
            "target_probabilities": "softmax REE target logits over native and proposed branch",
            "fused_action_cost": "Q action cost + 5.0 * (1 - REE target probability)",
            "candidate_score": (
                "minimum fused native action cost minus minimum fused proposed action cost"
            ),
            "activation": "candidate_score > 0; no calibrated threshold",
            "aggregation": "report each seed independently; no ensemble",
        },
        "per_seed_success_gates": {
            "score_auc_above_0_5": True,
            "nonzero_activations": True,
            "activation_positive_precision_above_0_5": True,
            "mean_realized_net_per_activation_positive": True,
            "mean_regret_below_always_native": True,
        },
        "overall_gate": "at least two of the three same-seed pairs pass every per-seed gate",
        "sources": {
            str(SOURCE.relative_to(ROOT)): sha256_file(SOURCE),
            str(Q_COMPARISON.relative_to(ROOT)): sha256_file(Q_COMPARISON),
            str(Q_HOLDOUT_RESULT.relative_to(ROOT)): sha256_file(Q_HOLDOUT_RESULT),
            str(OLD_LOCK.relative_to(ROOT)): sha256_file(OLD_LOCK),
            "revealnav_mf2r3/model.py": sha256_file(ROOT / "revealnav_mf2r3/model.py"),
            "revealnav_mf2r4/model.py": sha256_file(ROOT / "revealnav_mf2r4/model.py"),
            "scripts/evaluate_r2r_ree_q_v5_2_policy_holdout.py": sha256_file(
                ROOT / "scripts/evaluate_r2r_ree_q_v5_2_policy_holdout.py"
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
        raise RuntimeError("sealed REE-Q holdout protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "rows": value["scope"]["rows"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def load_pairs(device: torch.device) -> list[tuple[int, torch.nn.Module, torch.nn.Module]]:
    loaded = []
    for row in checkpoint_pairs():
        ree_payload = torch.load(
            ROOT / row["ree"]["path"], map_location="cpu", weights_only=False
        )
        q_payload = torch.load(
            ROOT / row["q"]["path"], map_location="cpu", weights_only=False
        )
        if not (
            ree_payload.get("schema_version") == "revealnav-mf2-expiry-checkpoint/3.1"
            and ree_payload.get("condition") == "augmented"
            and ree_payload.get("seed") == row["seed"]
            and q_payload.get("schema_version")
            == "revealnav-mf2-branch-excursion-q-checkpoint/5.1"
            and q_payload.get("variant") == "source_balanced"
            and q_payload.get("seed") == row["seed"]
        ):
            raise RuntimeError("same-seed REE-Q checkpoint schema drift")
        ree = RelationalRevealExpiryHeads(768, 128, 4).to(device)
        ree.load_state_dict(ree_payload["model_state_dict"], strict=True)
        ree.eval()
        q = BranchExcursionQHead(768, 96, 128.0).to(device)
        q.load_state_dict(q_payload["model_state_dict"], strict=True)
        q.eval()
        loaded.append((row["seed"], ree, q))
    return loaded


def inputs(feature: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
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
    if candidates.shape[2] != 2 or not bool(mask[0, -1].all()):
        raise RuntimeError("REE-Q policy feature shape drift")
    decision = torch.tensor([history.shape[1] - 1], device=device)
    budgets = torch.tensor(
        [1.5, 2.0, 3.0, 4.0], device=device
    ).view(1, 1, 4).expand(1, history.shape[1], 4)
    return history, candidates, mask, instruction, decision, budgets


def scores(
    pairs: list[tuple[int, torch.nn.Module, torch.nn.Module]],
    feature: Path,
    device: torch.device,
) -> list[float]:
    history, candidates, mask, instruction, decision, budgets = inputs(feature, device)
    values = []
    with torch.no_grad():
        for _, ree, q in pairs:
            ree_output = ree(history, candidates, mask, budgets, instruction)
            q_output = q(history, candidates, mask, instruction, decision)
            probabilities = torch.softmax(ree_output.target_logits[0, -1], dim=-1)
            penalty = WRONG_COMMITMENT_WEIGHT * (1.0 - probabilities)
            fused = torch.stack((q_output.commit_cost[0], q_output.excursion_cost[0]))
            fused = fused + penalty.unsqueeze(0)
            candidate_cost = fused.min(0).values
            values.append(float(candidate_cost[0] - candidate_cost[1]))
    return values


def metrics(labels: np.ndarray, rewards: np.ndarray, score: np.ndarray) -> dict:
    activated = score > 0.0
    selected_reward = np.where(activated, rewards, 0.0)
    oracle_reward = np.maximum(rewards, 0.0)
    true_positive = int(np.sum(activated & (labels == 1)))
    count = int(activated.sum())
    positive = int(labels.sum())
    return {
        "score_auc": rank_auc(labels, score),
        "activations": count,
        "activation_rate": count / len(labels),
        "activation_positive_precision": true_positive / count if count else 0.0,
        "positive_recall": true_positive / positive,
        "mean_realized_net_per_activation_m": (
            float(selected_reward[activated].mean()) if count else 0.0
        ),
        "mean_realized_net_per_event_m": float(selected_reward.mean()),
        "mean_regret_m": float((oracle_reward - selected_reward).mean()),
        "always_native_mean_regret_m": float(oracle_reward.mean()),
    }


def gates(value: dict) -> dict:
    return {
        "score_auc_above_0_5": value["score_auc"] > 0.5,
        "nonzero_activations": value["activations"] > 0,
        "activation_positive_precision_above_0_5": (
            value["activation_positive_precision"] > 0.5
        ),
        "mean_realized_net_per_activation_positive": (
            value["mean_realized_net_per_activation_m"] > 0.0
        ),
        "mean_regret_below_always_native": (
            value["mean_regret_m"] < value["always_native_mean_regret_m"]
        ),
    }


def evaluate(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("fixed REE-Q holdout protocol must be sealed")
    source = q_holdout.source_manifest()
    pairs = load_pairs(device)
    records = []
    labels, rewards, score_rows = [], [], []
    for record in source["records"]:
        _, feature = q_holdout.source_event(record)
        per_seed = scores(pairs, feature, device)
        labels.append(int(record["better_by_margin"]))
        rewards.append(float(record["realized_trial_net_m"]))
        score_rows.append(per_seed)
        records.append({
            "event_id": record["event_id"],
            "episode_id": record["episode_id"],
            "scene_id": record["scene_id"],
            "label_better_by_margin": bool(record["better_by_margin"]),
            "realized_reward_if_activated_m": float(record["realized_trial_net_m"]),
            "same_seed_fused_scores": {
                str(seed): score for seed, score in zip(SEEDS, per_seed)
            },
        })
    labels_array = np.asarray(labels, dtype=np.int64)
    rewards_array = np.asarray(rewards, dtype=np.float64)
    score_array = np.asarray(score_rows, dtype=np.float64)
    per_seed = {}
    passing = []
    for index, seed in enumerate(SEEDS):
        seed_metrics = metrics(labels_array, rewards_array, score_array[:, index])
        seed_gates = gates(seed_metrics)
        passed = all(seed_gates.values())
        if passed:
            passing.append(seed)
        per_seed[str(seed)] = {
            "status": "PAIR_PASS" if passed else "PAIR_FAIL",
            "metrics": seed_metrics,
            "gates": seed_gates,
        }
    passed = len(passing) >= 2
    value = {
        "schema_version": "revealnav-r2r-ree-q-policy-holdout-result/5.2",
        "status": "R2R_REE_Q_V5_2_POLICY_HOLDOUT_PASS" if passed
                  else "R2R_REE_Q_V5_2_POLICY_HOLDOUT_FAIL",
        "rows": len(records),
        "positive_rows": int(labels_array.sum()),
        "negative_rows": int((labels_array == 0).sum()),
        "passing_seeds": passing,
        "per_seed": per_seed,
        "records": records,
        "protocol_sha256": sha256_file(PROTOCOL),
        "task_metric_payload_read": False,
        "val_seen_used_for_threshold_or_selection": False,
        "val_unseen_or_test_read": False,
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": "conservative controller integration" if passed
                     else "method representation revision",
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"],
        "passing_seeds": passing,
        "per_seed": per_seed,
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
