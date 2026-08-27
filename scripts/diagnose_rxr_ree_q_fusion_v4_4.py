#!/usr/bin/env python3
"""Test the already-frozen REE target signal with V4 Q on train-only dev."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r3 import RelationalRevealExpiryHeads  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    BranchExcursionQHead, collate_branch_excursion_examples,
)
import run_rxr_branch_excursion_q_v4 as v4  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


SEEDS = v4.SEEDS
Q_LOCK = ROOT / "locks/RXR_UNSEEN_CHECKPOINT_LOCK_V4_2.json"
REE_ROOT = ROOT / "artifacts/evaluation/mf2_expiry_r3_1"
REE_ACCEPTANCE = REE_ROOT / "RXR_EXPIRY_R3_COMPARISON.json"
OPP_PROTOCOL = (
    ROOT / "artifacts/evaluation/mf2_ecog_opp_development_v2"
    / "RXR_ECOG_OPP_DEVELOPMENT_PROTOCOL.json"
)
BASELINE = (
    ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v4_3"
    / "RXR_BRANCH_EXCURSION_Q_STABILITY_DIAGNOSIS_V4_3.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_ree_q_fusion_v4_4"
PROTOCOL = OUT / "RXR_REE_Q_FUSION_PROTOCOL_V4_4.json"
RESULT = OUT / "RXR_REE_Q_FUSION_DIAGNOSIS_V4_4.json"
WRONG_COMMITMENT_WEIGHT = 5.0


def checkpoint_cohort() -> list[dict]:
    q_rows = json.loads(Q_LOCK.read_text())["checkpoints"]
    opp = json.loads(OPP_PROTOCOL.read_text())
    rows = []
    for seed, q_row in zip(SEEDS, q_rows):
        if q_row["seed"] != seed:
            raise RuntimeError("Q checkpoint cohort drift")
        q_path = ROOT / q_row["path"]
        ree_path = REE_ROOT / f"augmented_seed_{seed}/relational_expiry_ree.pt"
        recorded = opp["fixed_checkpoints"].get(str(ree_path.relative_to(ROOT)))
        for path, expected in ((q_path, q_row["sha256"]), (ree_path, recorded)):
            if (
                expected is None or path.is_symlink() or not path.is_file()
                or ROOT not in path.resolve().parents
                or sha256_file(path) != expected
            ):
                raise RuntimeError(f"checkpoint provenance drift: {path}")
        rows.append({
            "seed": seed,
            "q_path": str(q_path.relative_to(ROOT)),
            "q_sha256": q_row["sha256"],
            "ree_path": str(ree_path.relative_to(ROOT)),
            "ree_sha256": recorded,
        })
    return rows


def protocol_value() -> dict:
    acceptance = json.loads(REE_ACCEPTANCE.read_text())
    baseline = json.loads(BASELINE.read_text())
    if not (
        acceptance.get("status") == "EXPIRY_R3_1_GATE_PASS"
        and baseline.get("status")
        == "TRAIN_ONLY_DEVELOPMENT_STABILITY_DIAGNOSIS_COMPLETE"
        and baseline.get("gold_payload_read") is False
        and baseline.get("unseen_payload_read") is False
    ):
        raise RuntimeError("REE/Q fusion precondition failed")
    return {
        "schema_version": "revealnav-mf2-ree-q-fusion-protocol/4.4",
        "status": "SEALED_BEFORE_TRAIN_ONLY_REE_Q_FUSION_DIAGNOSIS",
        "scope": "same 83-event V4 train-only internal development partition",
        "checkpoint_cohort": checkpoint_cohort(),
        "fixed_composition": (
            "score(a,b) = V4_Q(a,b) + 5.0 * (1 - REE_target_probability(b))"
        ),
        "wrong_commitment_weight": WRONG_COMMITMENT_WEIGHT,
        "weight_source": (
            "existing frozen LearnedOPPConfig.wrong_commitment_weight; "
            "not selected in this diagnosis"
        ),
        "diagnostic_selectors": ["raw_v4_q", "ree_target_probability", "fixed_ree_q"],
        "acceptance_gates": [
            "raw_v4_metrics_reproduced",
            "fixed_fusion_mean_regret_below_raw_v4",
            "fixed_fusion_exact_oracle_equivalence_not_below_raw_v4",
            "fixed_fusion_branch_agreement_above_raw_v4",
        ],
        "sources": {
            str(Q_LOCK.relative_to(ROOT)): sha256_file(Q_LOCK),
            str(REE_ACCEPTANCE.relative_to(ROOT)): sha256_file(REE_ACCEPTANCE),
            str(OPP_PROTOCOL.relative_to(ROOT)): sha256_file(OPP_PROTOCOL),
            str(BASELINE.relative_to(ROOT)): sha256_file(BASELINE),
            "revealnav_mf2r3/policy.py": sha256_file(ROOT / "revealnav_mf2r3/policy.py"),
        },
        "val_unseen_selection_allowed": False,
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed REE/Q fusion protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def load_models(row: dict, device: torch.device):
    q_payload = torch.load(
        ROOT / row["q_path"], map_location="cpu", weights_only=False
    )
    q_model = BranchExcursionQHead(768, 96, 128.0).to(device)
    q_model.load_state_dict(q_payload["model_state_dict"], strict=True)
    ree_payload = torch.load(
        ROOT / row["ree_path"], map_location="cpu", weights_only=False
    )
    ree_model = RelationalRevealExpiryHeads(768, 128, 4).to(device)
    ree_model.load_state_dict(ree_payload["model_state_dict"], strict=True)
    return q_model.eval(), ree_model.eval()


def selector_summary(indices, teachers) -> dict:
    regrets = [
        float(teacher[index] - teacher.min())
        for index, teacher in zip(indices, teachers)
    ]
    return {
        "mean_teacher_cost_regret": statistics.mean(regrets),
        "exact_oracle_equivalence": statistics.mean(
            regret <= 1e-9 for regret in regrets
        ),
    }


def agreement(decisions: list[list[tuple[bool, int]]]) -> dict:
    action, branch, joint = [], [], []
    for event in range(len(decisions[0])):
        values = [row[event] for row in decisions]
        same_action = len({value[0] for value in values}) == 1
        same_branch = len({value[1] for value in values}) == 1
        action.append(same_action)
        branch.append(same_branch)
        joint.append(same_action and same_branch)
    return {
        "all_three_macro_action_agreement": statistics.mean(action),
        "all_three_branch_index_agreement": statistics.mean(branch),
        "all_three_joint_action_branch_agreement": statistics.mean(joint),
    }


def run(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("REE/Q fusion protocol must be sealed")
    _, development = v4.datasets()
    loader = list(DataLoader(
        development, batch_size=1, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    ))
    teachers = []
    raw_decisions = []
    fused_decisions = []
    ree_branches = []
    raw_summaries = {}
    fused_summaries = {}
    ree_branch_accuracy = {}
    cohort = checkpoint_cohort()
    for row in cohort:
        q_model, ree_model = load_models(row, device)
        raw_indices = []
        fused_indices = []
        predicted_branches = []
        seed_teachers = []
        with torch.no_grad():
            for cpu in loader:
                batch = v4.move(cpu, device)
                valid = torch.isfinite(batch["commit_cost"][0])
                count = int(valid.sum())
                teacher = torch.cat((
                    batch["commit_cost"][0, valid],
                    batch["excursion_cost"][0, valid],
                ))
                q_output = v4.forward(q_model, batch)
                q_costs = torch.cat((
                    q_output.commit_cost[0, valid],
                    q_output.excursion_cost[0, valid],
                ))
                steps = batch["history_embeddings"].shape[1]
                budgets = torch.tensor(
                    [1.5, 2.0, 3.0, 4.0], device=device
                ).view(1, 1, 4).expand(1, steps, 4)
                ree_output = ree_model(
                    batch["history_embeddings"], batch["candidate_embeddings"],
                    batch["candidate_mask"], budgets,
                    batch["instruction_embedding"],
                )
                step = int(batch["decision_index"][0])
                probability = torch.softmax(
                    ree_output.target_logits[0, step, valid], dim=-1
                )
                penalty = WRONG_COMMITMENT_WEIGHT * (1.0 - probability)
                fused_costs = q_costs + torch.cat((penalty, penalty))
                raw_index = int(q_costs.argmin())
                fused_index = int(fused_costs.argmin())
                raw_indices.append(raw_index)
                fused_indices.append(fused_index)
                predicted_branches.append(int(probability.argmax()))
                seed_teachers.append(teacher.cpu())
        if not teachers:
            teachers = seed_teachers
        raw_decisions.append([
            (index >= len(teacher) // 2, index % (len(teacher) // 2))
            for index, teacher in zip(raw_indices, seed_teachers)
        ])
        fused_decisions.append([
            (index >= len(teacher) // 2, index % (len(teacher) // 2))
            for index, teacher in zip(fused_indices, seed_teachers)
        ])
        ree_branches.append(predicted_branches)
        raw_summaries[str(row["seed"])] = selector_summary(
            raw_indices, seed_teachers
        )
        fused_summaries[str(row["seed"])] = selector_summary(
            fused_indices, seed_teachers
        )
        ree_branch_accuracy[str(row["seed"])] = statistics.mean(
            bool(teacher[branch] <= teacher.min() + 1e-9)
            or bool(
                teacher[branch + len(teacher) // 2]
                <= teacher.min() + 1e-9
            )
            for teacher, branch in zip(seed_teachers, predicted_branches)
        )

    raw = {
        "per_seed": raw_summaries,
        "mean_teacher_cost_regret": statistics.mean(
            row["mean_teacher_cost_regret"] for row in raw_summaries.values()
        ),
        "mean_exact_oracle_equivalence": statistics.mean(
            row["exact_oracle_equivalence"] for row in raw_summaries.values()
        ),
        **agreement(raw_decisions),
    }
    fused = {
        "per_seed": fused_summaries,
        "mean_teacher_cost_regret": statistics.mean(
            row["mean_teacher_cost_regret"] for row in fused_summaries.values()
        ),
        "mean_exact_oracle_equivalence": statistics.mean(
            row["exact_oracle_equivalence"] for row in fused_summaries.values()
        ),
        **agreement(fused_decisions),
    }
    ree_agreement = statistics.mean(
        len({row[event] for row in ree_branches}) == 1
        for event in range(len(development))
    )
    baseline = json.loads(BASELINE.read_text())
    baseline_regret = statistics.mean(
        baseline["selectors"][str(seed)]["mean_teacher_cost_regret"]
        for seed in SEEDS
    )
    baseline_equivalence = statistics.mean(
        baseline["selectors"][str(seed)]["oracle_equivalence_rate"]["0.0"]
        for seed in SEEDS
    )
    gates = {
        "raw_v4_metrics_reproduced": (
            abs(raw["mean_teacher_cost_regret"] - baseline_regret) <= 1e-6
            and abs(raw["mean_exact_oracle_equivalence"] - baseline_equivalence)
            <= 1e-6
            and abs(
                raw["all_three_branch_index_agreement"]
                - baseline["all_three"]["branch_index_agreement"]
            ) <= 1e-6
        ),
        "fixed_fusion_mean_regret_below_raw_v4": (
            fused["mean_teacher_cost_regret"] < raw["mean_teacher_cost_regret"]
        ),
        "fixed_fusion_exact_oracle_equivalence_not_below_raw_v4": (
            fused["mean_exact_oracle_equivalence"]
            >= raw["mean_exact_oracle_equivalence"]
        ),
        "fixed_fusion_branch_agreement_above_raw_v4": (
            fused["all_three_branch_index_agreement"]
            > raw["all_three_branch_index_agreement"]
        ),
        "no_gold_or_unseen_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-ree-q-fusion-diagnosis/4.4",
        "status": (
            "REE_Q_FIXED_FUSION_ENGINEERING_PASS" if passed
            else "REE_Q_FIXED_FUSION_ENGINEERING_FAIL"
        ),
        "events": len(development),
        "raw_v4_q": raw,
        "ree_target_probability": {
            "per_seed_branch_oracle_accuracy": ree_branch_accuracy,
            "mean_branch_oracle_accuracy": statistics.mean(
                ree_branch_accuracy.values()
            ),
            "all_three_branch_agreement": ree_agreement,
        },
        "fixed_ree_q_fusion": fused,
        "gates": gates,
        "known_limitation": (
            "These 83 events were part of earlier REE training; this is an "
            "integration diagnosis, not confirmatory generalization evidence."
        ),
        "protocol_sha256": sha256_file(PROTOCOL),
        "gold_payload_read": False,
        "unseen_payload_read": False,
        "paper_result": False,
        "next_gate": (
            "freeze composition and test on fresh confirmatory events"
            if passed else "do not integrate; redesign branch representation"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps(value, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    return seal() if args.seal else run(torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
