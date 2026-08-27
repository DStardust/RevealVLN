#!/usr/bin/env python3
"""Tie-aware adjudication of the V4 candidate-agnostic median selector."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r4 import (  # noqa: E402
    BranchExcursionQHead, collate_branch_excursion_examples,
)
import run_rxr_branch_excursion_q_v4 as training  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


FAILED = training.COMPARISON
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v4_1"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_Q_TIE_ADJUDICATION_PROTOCOL_V4_1.json"
RESULT = OUT / "RXR_BRANCH_EXCURSION_Q_TIE_ADJUDICATION_RESULT_V4_1.json"


def checkpoint_paths() -> dict[str, Path]:
    return {
        str(seed): training.OUT / f"seed_{seed}/branch_excursion_q.pt"
        for seed in training.SEEDS
    }


def protocol_value() -> dict:
    failed = json.loads(FAILED.read_text())
    false_gates = sorted(key for key, passed in failed["gates"].items() if not passed)
    if not (
        failed.get("status") == "BRANCH_EXCURSION_Q_ENGINEERING_GATE_FAIL"
        and false_gates == ["action_regret_beats_median_selector_in_two_seeds"]
        and failed.get("gold_payload_read") is False
    ):
        raise RuntimeError("tie adjudication precondition failed")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-q-tie-adjudication/4.1",
        "status": "SEALED_AFTER_V4_FAILURE_BEFORE_TIE_AWARE_ADJUDICATION",
        "identified_bug": (
            "The candidate-agnostic median selector assigns identical scores to "
            "every candidate. V4 used argmin, whose first-index tie break exploits "
            "the source convention target_branch_id=BR01 at candidate index 0."
        ),
        "correction": (
            "Evaluate a tied candidate-agnostic selector by its expected teacher "
            "cost over every minimum-score action, not by candidate array order."
        ),
        "checks": {
            "model_regret_reproduces_v4": True,
            "tie_aware_median_regret_computed": True,
            "model_beats_tie_aware_median_in_all_three_seeds": True,
            "checkpoint_candidate_permutation_equivariance": True,
            "original_v4_failure_preserved": True,
        },
        "sources": {
            str(FAILED.relative_to(ROOT)): sha256_file(FAILED),
            str(training.PROTOCOL.relative_to(ROOT)): sha256_file(training.PROTOCOL),
            **{
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in checkpoint_paths().values()
            },
        },
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed tie-adjudication protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def load_model(seed: int, device):
    path = checkpoint_paths()[str(seed)]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not (
        payload["seed"] == seed
        and payload["protocol_sha256"] == sha256_file(training.PROTOCOL)
        and payload["manifest_sha256"] == sha256_file(training.MANIFEST)
    ):
        raise RuntimeError("checkpoint provenance drift")
    model = BranchExcursionQHead(768, 96, 128.0).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval(), (
        payload["train_medians"]["commit"],
        payload["train_medians"]["excursion"],
    )


def forward(model, batch):
    return model(
        batch["history_embeddings"], batch["candidate_embeddings"],
        batch["candidate_mask"], batch["instruction_embedding"],
        batch["decision_index"],
    )


def evaluate(seed: int, device) -> dict:
    model, medians = load_model(seed, device)
    _, development = training.datasets()
    loader = DataLoader(
        development, batch_size=1, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    )
    model_regrets = []
    tied_median_regrets = []
    first_index_regrets = []
    permutation_max = 0.0
    with torch.no_grad():
        for cpu in loader:
            batch = training.move(cpu, device)
            output = forward(model, batch)
            valid = torch.isfinite(batch["commit_cost"][0])
            teacher = torch.cat((
                batch["commit_cost"][0, valid],
                batch["excursion_cost"][0, valid],
            ))
            predicted = torch.cat((
                output.commit_cost[0, valid], output.excursion_cost[0, valid]
            ))
            oracle = teacher.min()
            model_regrets.append(float(teacher[predicted.argmin()] - oracle))
            count = int(valid.sum())
            median_prediction = torch.cat((
                torch.full_like(teacher[:count], medians[0]),
                torch.full_like(teacher[:count], medians[1]),
            ))
            tied = torch.isclose(
                median_prediction, median_prediction.min(), atol=1e-12, rtol=0.0
            )
            tied_median_regrets.append(float(teacher[tied].mean() - oracle))
            first_index_regrets.append(float(
                teacher[median_prediction.argmin()] - oracle
            ))
            order = torch.arange(
                batch["candidate_embeddings"].shape[2] - 1, -1, -1,
                device=device,
            )
            permuted = dict(batch)
            permuted["candidate_embeddings"] = batch["candidate_embeddings"][:, :, order]
            permuted["candidate_mask"] = batch["candidate_mask"][:, :, order]
            permuted_output = forward(model, permuted)
            restored_commit = permuted_output.commit_cost[:, order]
            restored_excursion = permuted_output.excursion_cost[:, order]
            finite_commit = torch.isfinite(output.commit_cost)
            finite_excursion = torch.isfinite(output.excursion_cost)
            permutation_max = max(
                permutation_max,
                float((restored_commit[finite_commit] - output.commit_cost[finite_commit]).abs().max()),
                float((restored_excursion[finite_excursion] - output.excursion_cost[finite_excursion]).abs().max()),
            )
    return {
        "model_mean_action_regret": statistics.mean(model_regrets),
        "tie_aware_median_mean_action_regret": statistics.mean(tied_median_regrets),
        "first_index_median_mean_action_regret": statistics.mean(first_index_regrets),
        "candidate_permutation_max_abs": permutation_max,
        "events": len(model_regrets),
    }


def run(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("tie-adjudication protocol must be sealed")
    failed = json.loads(FAILED.read_text())
    per_seed = {str(seed): evaluate(seed, device) for seed in training.SEEDS}
    reproduction = [
        abs(
            per_seed[str(seed)]["model_mean_action_regret"]
            - failed["metrics"]["mean_action_regret"]["values"][index]
        )
        for index, seed in enumerate(training.SEEDS)
    ]
    gates = {
        "model_regret_reproduces_v4": max(reproduction) <= 1e-6,
        "tie_aware_median_regret_computed": all(
            row["tie_aware_median_mean_action_regret"]
            > row["first_index_median_mean_action_regret"]
            for row in per_seed.values()
        ),
        "model_beats_tie_aware_median_in_all_three_seeds": all(
            row["model_mean_action_regret"]
            < row["tie_aware_median_mean_action_regret"]
            for row in per_seed.values()
        ),
        "checkpoint_candidate_permutation_equivariance": max(
            row["candidate_permutation_max_abs"] for row in per_seed.values()
        ) <= 1e-6,
        "original_v4_failure_preserved": failed["status"]
        == "BRANCH_EXCURSION_Q_ENGINEERING_GATE_FAIL",
        "no_gold_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-q-tie-adjudication-result/4.1",
        "status": (
            "BRANCH_EXCURSION_Q_TIE_AWARE_ENGINEERING_PASS" if passed
            else "BRANCH_EXCURSION_Q_TIE_AWARE_ENGINEERING_FAIL"
        ),
        "per_seed": per_seed,
        "aggregate": {
            key: statistics.mean(row[key] for row in per_seed.values())
            for key in (
                "model_mean_action_regret",
                "tie_aware_median_mean_action_regret",
                "first_index_median_mean_action_regret",
                "candidate_permutation_max_abs",
            )
        },
        "gates": gates,
        "original_v4_status_preserved": failed["status"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": "unseen controller integration" if passed else "model diagnosis",
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "aggregate": value["aggregate"],
        "gates": gates,
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return seal() if args.seal else run(torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
