#!/usr/bin/env python3
"""Train and gate the CR7 tie-aware branch-excursion Q objective."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r4 import (  # noqa: E402
    BranchExcursionQHead, collate_branch_excursion_examples,
)
from revealnav_mf2r4.stable_losses import StableBranchExcursionQLoss  # noqa: E402
import run_rxr_branch_excursion_q_v4 as v4  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


SEEDS = v4.SEEDS
BASELINE = (
    ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v4_3"
    / "RXR_BRANCH_EXCURSION_Q_STABILITY_DIAGNOSIS_V4_3.json"
)
DESIGN = ROOT / "artifacts/design/MF2_CORRECTNESS_REVISION_CR7_LISTWISE_Q.md"
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_q_cr7"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_Q_CR7_PROTOCOL.json"
COMPARISON = OUT / "RXR_BRANCH_EXCURSION_Q_CR7_COMPARISON.json"


def protocol_value() -> dict:
    baseline = json.loads(BASELINE.read_text())
    _, _, counts = v4.partitions()
    if not (
        baseline.get("status")
        == "TRAIN_ONLY_DEVELOPMENT_STABILITY_DIAGNOSIS_COMPLETE"
        and baseline.get("events") == counts["development_events"] == 83
        and baseline.get("gold_payload_read") is False
        and baseline.get("unseen_payload_read") is False
    ):
        raise RuntimeError("CR7 diagnosis precondition failed")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-q-cr7-protocol/1",
        "status": "SEALED_BEFORE_CR7_TRAINING",
        "seeds": list(SEEDS),
        "partition": "unchanged V4 train-only scene-disjoint partition",
        "counts": counts,
        "architecture": "unchanged BranchExcursionQHead(768, 96, 128.0)",
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "batch_size": 16,
        "epoch_limit": 25,
        "early_stopping_patience": 5,
        "objective": {
            "v4_commit_huber": 1.0,
            "v4_excursion_huber": 1.0,
            "v4_paired_gap_huber": 1.0,
            "v4_ranking_weight": 0.25,
            "v4_ranking_margin": 0.1,
            "tie_aware_listwise_weight": 1.0,
            "tie_target": "uniform over every action at minimum teacher cost",
        },
        "selection": "minimum internal-development CR7 native loss",
        "acceptance_gates": [
            "all_three_runs_complete",
            "mean_teacher_cost_regret_improves_over_v4",
            "mean_exact_oracle_equivalence_improves_over_v4",
            "all_three_branch_agreement_improves_over_v4",
            "all_outputs_finite",
        ],
        "sources": {
            str(BASELINE.relative_to(ROOT)): sha256_file(BASELINE),
            str(DESIGN.relative_to(ROOT)): sha256_file(DESIGN),
            str(v4.MANIFEST.relative_to(ROOT)): sha256_file(v4.MANIFEST),
            "revealnav_mf2r4/model.py": sha256_file(ROOT / "revealnav_mf2r4/model.py"),
            "revealnav_mf2r4/losses.py": sha256_file(ROOT / "revealnav_mf2r4/losses.py"),
            "revealnav_mf2r4/stable_losses.py": sha256_file(
                ROOT / "revealnav_mf2r4/stable_losses.py"
            ),
        },
        "val_unseen_selection_allowed": False,
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed CR7 protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def run(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("seed outside CR7 protocol")
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("CR7 protocol must be sealed")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    train_data, development_data = v4.datasets()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=16, shuffle=True, generator=generator,
        collate_fn=collate_branch_excursion_examples,
    )
    development_loader = DataLoader(
        development_data, batch_size=16, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    )
    model = BranchExcursionQHead(768, 96, 128.0).to(device)
    objective = StableBranchExcursionQLoss(1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss = None
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, 26):
        model.train()
        train_sum = 0.0
        train_count = 0
        for cpu in train_loader:
            batch = v4.move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            losses = objective(v4.forward(model, batch), batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite CR7 training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = batch["history_embeddings"].shape[0]
            train_sum += float(losses["total"].detach()) * size
            train_count += size
        model.eval()
        development_sum = 0.0
        development_count = 0
        with torch.no_grad():
            for cpu in development_loader:
                batch = v4.move(cpu, device)
                losses = objective(v4.forward(model, batch), batch)
                if not torch.isfinite(losses["total"]):
                    raise RuntimeError("non-finite CR7 development loss")
                size = batch["history_embeddings"].shape[0]
                development_sum += float(losses["total"]) * size
                development_count += size
        native = development_sum / development_count
        history.append({
            "epoch": epoch,
            "train_total": train_sum / train_count,
            "development_total": native,
        })
        if best_loss is None or native < best_loss:
            best_loss = native
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best_state, strict=True)
    medians = v4.train_medians(train_data)
    metrics = v4.evaluate(model, development_data, device, medians)
    checkpoint = run_dir / "branch_excursion_q_cr7.pt"
    torch.save({
        "schema_version": "revealnav-mf2-branch-excursion-q-checkpoint/cr7",
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(v4.MANIFEST),
    }, checkpoint)
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-q-run/cr7",
        "status": "BRANCH_EXCURSION_Q_CR7_RUN_COMPLETE",
        "seed": seed,
        "metrics": metrics,
        "history": history,
        "best_development_native_loss": best_loss,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "gold_payload_read": False,
        "unseen_payload_read": False,
        "paper_result": False,
    }
    atomic_json(run_dir / "result.json", value)
    print(json.dumps({
        "status": value["status"], "seed": seed, "metrics": metrics,
    }, indent=2))
    return 0


def load_trained_models(device: torch.device) -> list[torch.nn.Module]:
    models = []
    for seed in SEEDS:
        path = OUT / f"seed_{seed}/branch_excursion_q_cr7.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not (
            payload.get("seed") == seed
            and payload.get("protocol_sha256") == sha256_file(PROTOCOL)
            and payload.get("manifest_sha256") == sha256_file(v4.MANIFEST)
        ):
            raise RuntimeError("CR7 checkpoint provenance drift")
        model = BranchExcursionQHead(768, 96, 128.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.eval())
    return models


def stability(device: torch.device) -> dict:
    models = load_trained_models(device)
    _, development = v4.datasets()
    loader = DataLoader(
        development, batch_size=1, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    )
    decisions = [[] for _ in models]
    margins = [[] for _ in models]
    with torch.no_grad():
        for cpu in loader:
            batch = v4.move(cpu, device)
            valid = torch.isfinite(batch["commit_cost"][0])
            count = int(valid.sum())
            for index, model in enumerate(models):
                output = v4.forward(model, batch)
                prediction = torch.cat((
                    output.commit_cost[0, valid], output.excursion_cost[0, valid],
                ))
                selected = int(prediction.argmin())
                decisions[index].append((selected >= count, selected % count))
                top = torch.topk(prediction, 2, largest=False).values
                margins[index].append(float(top[1] - top[0]))
    action_agreement = []
    branch_agreement = []
    joint_agreement = []
    for event in range(len(development)):
        values = [row[event] for row in decisions]
        same_action = len({value[0] for value in values}) == 1
        same_branch = len({value[1] for value in values}) == 1
        action_agreement.append(same_action)
        branch_agreement.append(same_branch)
        joint_agreement.append(same_action and same_branch)
    return {
        "all_three_macro_action_agreement": statistics.mean(action_agreement),
        "all_three_branch_index_agreement": statistics.mean(branch_agreement),
        "all_three_joint_action_branch_agreement": statistics.mean(joint_agreement),
        "predicted_top1_top2_margin_median": {
            str(seed): statistics.median(values)
            for seed, values in zip(SEEDS, margins)
        },
    }


def aggregate(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("CR7 protocol drift")
    rows = [
        json.loads((OUT / f"seed_{seed}/result.json").read_text())
        for seed in SEEDS
    ]
    complete = all(
        row.get("status") == "BRANCH_EXCURSION_Q_CR7_RUN_COMPLETE"
        and row.get("seed") == seed
        and row.get("gold_payload_read") is False
        and row.get("unseen_payload_read") is False
        for row, seed in zip(rows, SEEDS)
    )
    current_stability = stability(device)
    baseline = json.loads(BASELINE.read_text())
    baseline_regret = statistics.mean(
        baseline["selectors"][str(seed)]["mean_teacher_cost_regret"]
        for seed in SEEDS
    )
    baseline_equivalence = statistics.mean(
        baseline["selectors"][str(seed)]["oracle_equivalence_rate"]["0.0"]
        for seed in SEEDS
    )
    current_regret = statistics.mean(
        row["metrics"]["mean_action_regret"] for row in rows
    )
    current_equivalence = statistics.mean(
        row["metrics"]["best_action_accuracy"] for row in rows
    )
    finite = all(
        np.isfinite(value)
        for row in rows
        for value in row["metrics"].values()
        if isinstance(value, (int, float))
    )
    gates = {
        "all_three_runs_complete": complete,
        "mean_teacher_cost_regret_improves_over_v4": current_regret < baseline_regret,
        "mean_exact_oracle_equivalence_improves_over_v4": (
            current_equivalence > baseline_equivalence
        ),
        "all_three_branch_agreement_improves_over_v4": (
            current_stability["all_three_branch_index_agreement"]
            > baseline["all_three"]["branch_index_agreement"]
        ),
        "all_outputs_finite": finite,
        "no_gold_or_unseen_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-q-comparison/cr7",
        "status": (
            "BRANCH_EXCURSION_Q_CR7_DEVELOPMENT_GATE_PASS" if passed
            else "BRANCH_EXCURSION_Q_CR7_DEVELOPMENT_GATE_FAIL"
        ),
        "baseline": {
            "mean_teacher_cost_regret": baseline_regret,
            "mean_exact_oracle_equivalence": baseline_equivalence,
            "all_three_branch_index_agreement": baseline["all_three"][
                "branch_index_agreement"
            ],
        },
        "cr7": {
            "mean_teacher_cost_regret": current_regret,
            "mean_exact_oracle_equivalence": current_equivalence,
            **current_stability,
        },
        "per_seed": {str(seed): row for seed, row in zip(SEEDS, rows)},
        "gates": gates,
        "protocol_sha256": sha256_file(PROTOCOL),
        "gold_payload_read": False,
        "unseen_payload_read": False,
        "paper_result": False,
        "next_gate": (
            "lock three CR7 checkpoints before fresh confirmatory evaluation"
            if passed else "preserve failure and return to causal diagnosis"
        ),
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({
        "status": value["status"], "baseline": value["baseline"],
        "cr7": value["cr7"], "gates": gates,
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.aggregate:
        return aggregate(torch.device(args.device))
    if args.seed is None:
        parser.error("--run requires --seed")
    return run(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
