#!/usr/bin/env python3
"""Development-only evaluation with a train-supported causal context window."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_r2r_q_v5_1_policy_holdout as q_holdout  # noqa: E402
import evaluate_r2r_ree_q_v5_2_policy_holdout as fusion  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


TRAIN_LABELS = ROOT / (
    "artifacts/phase1/rxr_train_expansion/branch_excursion_v5_1/"
    "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V5_1.json"
)
PREVIOUS = fusion.RESULT
OUT = ROOT / "artifacts/evaluation/mf2_r2r_ree_q_v5_3_context_aligned"
PROTOCOL = OUT / "R2R_REE_Q_V5_3_CONTEXT_PROTOCOL.json"
RESULT = OUT / "R2R_REE_Q_V5_3_CONTEXT_RESULT.json"


def context_contract() -> dict:
    manifest = json.loads(TRAIN_LABELS.read_text())
    steps = []
    root = TRAIN_LABELS.parent
    for row in manifest["records"]:
        label = json.loads((root / row["path"]).read_text())
        steps.append(int(label["online_feature_relative_step"]) + 1)
    if len(steps) != 1830:
        raise RuntimeError("expanded Q context population drift")
    cap = int(math.ceil(float(np.quantile(steps, 0.95))))
    if cap != 5:
        raise RuntimeError(f"expanded Q 95th-percentile context drift: {cap}")
    return {
        "training_events": len(steps),
        "minimum_steps": min(steps),
        "median_steps": float(np.median(steps)),
        "95th_percentile_steps": float(np.quantile(steps, 0.95)),
        "maximum_steps": max(steps),
        "context_cap": cap,
    }


def protocol_value() -> dict:
    previous = json.loads(PREVIOUS.read_text())
    contract = context_contract()
    if previous.get("status") != "R2R_REE_Q_V5_2_POLICY_HOLDOUT_FAIL":
        raise RuntimeError("V5.2 failure evidence is absent")
    return {
        "schema_version": "revealnav-r2r-ree-q-context-protocol/5.3",
        "status": "SEALED_BEFORE_CONTEXT_ALIGNED_DEVELOPMENT_INFERENCE",
        "revision": (
            "Feed REE and Q only the latest context_cap causal states and reset "
            "their local age coordinate to zero at the retained suffix start."
        ),
        "context_selection": {
            **contract,
            "selection_source": "95th percentile of the 1,830 Q training decisions",
            "R2R_labels_or_scores_used_to_select_cap": False,
        },
        "unchanged": {
            "same_seed_pairs": list(fusion.SEEDS),
            "wrong_commitment_weight": fusion.WRONG_COMMITMENT_WEIGHT,
            "activation": "candidate_score > 0; no calibrated threshold",
            "ensemble": False,
            "checkpoint_weights": True,
        },
        "evaluation_role": (
            "opened R2R-train policy proposal development diagnostic; not fresh holdout"
        ),
        "success_gate": (
            "at least two same-seed pairs pass the unchanged V5.2 per-seed gates"
        ),
        "sources": {
            str(TRAIN_LABELS.relative_to(ROOT)): sha256_file(TRAIN_LABELS),
            str(PREVIOUS.relative_to(ROOT)): sha256_file(PREVIOUS),
            "scripts/evaluate_r2r_ree_q_v5_2_policy_holdout.py": sha256_file(
                ROOT / "scripts/evaluate_r2r_ree_q_v5_2_policy_holdout.py"
            ),
            "scripts/evaluate_r2r_ree_q_v5_3_context_aligned.py": sha256_file(
                ROOT / "scripts/evaluate_r2r_ree_q_v5_3_context_aligned.py"
            ),
        },
        "val_seen_used_for_threshold_or_selection": False,
        "val_unseen_or_test_read": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.3 context protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "context_selection": value["context_selection"],
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def context_inputs(
    feature: Path, device: torch.device, cap: int,
) -> tuple[torch.Tensor, ...]:
    with np.load(feature, allow_pickle=False) as shard:
        instruction = torch.from_numpy(
            shard["instruction_embedding"].astype(np.float32)
        ).unsqueeze(0).to(device)
        history = torch.from_numpy(
            shard["history_embeddings"][-cap:].astype(np.float32)
        ).unsqueeze(0).to(device)
        candidates = torch.from_numpy(
            shard["candidate_embeddings"][-cap:].astype(np.float32)
        ).unsqueeze(0).to(device)
        mask = torch.from_numpy(
            shard["candidate_mask"][-cap:].astype(np.bool_)
        ).unsqueeze(0).to(device)
    if candidates.shape[2] != 2 or not bool(mask[0, -1].all()):
        raise RuntimeError("V5.3 context feature shape drift")
    decision = torch.tensor([history.shape[1] - 1], device=device)
    budgets = torch.tensor(
        [1.5, 2.0, 3.0, 4.0], device=device
    ).view(1, 1, 4).expand(1, history.shape[1], 4)
    return history, candidates, mask, instruction, decision, budgets


def scores(pairs, feature: Path, device: torch.device, cap: int) -> list[float]:
    history, candidates, mask, instruction, decision, budgets = context_inputs(
        feature, device, cap
    )
    values = []
    with torch.no_grad():
        for _, ree, q in pairs:
            ree_output = ree(history, candidates, mask, budgets, instruction)
            q_output = q(history, candidates, mask, instruction, decision)
            probabilities = torch.softmax(ree_output.target_logits[0, -1], dim=-1)
            penalty = fusion.WRONG_COMMITMENT_WEIGHT * (1.0 - probabilities)
            fused = torch.stack((q_output.commit_cost[0], q_output.excursion_cost[0]))
            candidate_cost = (fused + penalty.unsqueeze(0)).min(0).values
            values.append(float(candidate_cost[0] - candidate_cost[1]))
    return values


def evaluate(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("V5.3 context protocol must be sealed")
    cap = context_contract()["context_cap"]
    source = q_holdout.source_manifest()
    pairs = fusion.load_pairs(device)
    labels, rewards, score_rows = [], [], []
    for record in source["records"]:
        _, feature = q_holdout.source_event(record)
        labels.append(int(record["better_by_margin"]))
        rewards.append(float(record["realized_trial_net_m"]))
        score_rows.append(scores(pairs, feature, device, cap))
    labels_array = np.asarray(labels, dtype=np.int64)
    rewards_array = np.asarray(rewards, dtype=np.float64)
    score_array = np.asarray(score_rows, dtype=np.float64)
    per_seed = {}
    passing = []
    for index, seed in enumerate(fusion.SEEDS):
        seed_metrics = fusion.metrics(labels_array, rewards_array, score_array[:, index])
        seed_gates = fusion.gates(seed_metrics)
        if all(seed_gates.values()):
            passing.append(seed)
        per_seed[str(seed)] = {
            "status": "PAIR_PASS" if all(seed_gates.values()) else "PAIR_FAIL",
            "metrics": seed_metrics,
            "gates": seed_gates,
        }
    passed = len(passing) >= 2
    value = {
        "schema_version": "revealnav-r2r-ree-q-context-result/5.3",
        "status": "R2R_REE_Q_V5_3_CONTEXT_DEVELOPMENT_PASS" if passed
                  else "R2R_REE_Q_V5_3_CONTEXT_DEVELOPMENT_FAIL",
        "context_cap": cap,
        "passing_seeds": passing,
        "per_seed": per_seed,
        "protocol_sha256": sha256_file(PROTOCOL),
        "evaluation_role": "opened_development_diagnostic_only",
        "val_seen_used_for_threshold_or_selection": False,
        "val_unseen_or_test_read": False,
        "paper_result": False,
        "next_gate": "fresh frozen confirmation" if passed
                     else "retrain policy-aligned representation",
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"],
        "context_cap": cap,
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
