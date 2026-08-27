#!/usr/bin/env python3
"""Train and gate the train-only branch-excursion action-cost head."""

from __future__ import annotations

import argparse
import hashlib
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
ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r4 import (  # noqa: E402
    BranchExcursionDataset, BranchExcursionQHead, BranchExcursionQLoss,
    collate_branch_excursion_examples,
)
from run_rxr_opp_q_adapter_r3_2 import (  # noqa: E402
    atomic_json, rank_auc, sha256_file,
)


SEEDS = (20260826, 20260827, 20260828)
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/branch_excursion_v4"
MANIFEST = BASE / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V4.json"
ACCEPTANCE = BASE / "RXR_BRANCH_EXCURSION_CORRECTNESS_ACCEPTANCE_V4_1.json"
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v4"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_Q_PROTOCOL_V4.json"
COMPARISON = OUT / "RXR_BRANCH_EXCURSION_Q_COMPARISON_V4.json"


def is_development_scene(scene_id: str) -> bool:
    return int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 6 == 1


def partitions() -> tuple[set[str], set[str], dict]:
    manifest = json.loads(MANIFEST.read_text())
    train, development = set(), set()
    train_scenes, development_scenes = set(), set()
    for row in manifest["records"]:
        if is_development_scene(row["scene_id"]):
            development.add(row["event_id"])
            development_scenes.add(row["scene_id"])
        else:
            train.add(row["event_id"])
            train_scenes.add(row["scene_id"])
    counts = {
        "train_events": len(train),
        "development_events": len(development),
        "train_scenes": len(train_scenes),
        "development_scenes": len(development_scenes),
    }
    if train_scenes & development_scenes or counts != {
        "train_events": 341, "development_events": 83,
        "train_scenes": 26, "development_scenes": 9,
    }:
        raise RuntimeError(f"branch-excursion partition drift: {counts}")
    return train, development, counts


def protocol_value() -> dict:
    acceptance = json.loads(ACCEPTANCE.read_text())
    train, development, counts = partitions()
    if not (
        acceptance.get("status")
        == "BRANCH_EXCURSION_LABEL_CORRECTNESS_ACCEPTANCE_PASS"
        and acceptance.get("training_authorized") is True
        and acceptance.get("gold_payload_read") is False
        and len(train | development) == 424
    ):
        raise RuntimeError("branch-excursion Q precondition failed")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-q-protocol/4",
        "status": "SEALED_BEFORE_BRANCH_EXCURSION_Q_TRAINING",
        "seeds": list(SEEDS),
        "partition": "sha256(scene_id) mod 6 == 1 development; otherwise train",
        "counts": counts,
        "architecture": {
            "feature_dim": 768, "hidden_dim": 96,
            "age_denominator": 128.0, "causal": True,
            "outputs": ["commit_cost", "checkpointed_excursion_cost"],
        },
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "batch_size": 16,
        "epoch_limit": 25,
        "early_stopping_patience": 5,
        "loss": {
            "commit_huber": 1.0, "excursion_huber": 1.0,
            "paired_gap_huber": 1.0,
            "within_event_action_ranking": 0.25, "margin": 0.1,
        },
        "selection": "minimum scene-disjoint development native loss",
        "success_gates": {
            "commit_mae_beats_train_median": True,
            "excursion_mae_beats_train_median": True,
            "action_regret_beats_median_selector_in_two_seeds": True,
            "best_action_accuracy_beats_random_in_two_seeds": True,
            "preservation_gain_auc_above_0_5_in_two_seeds": True,
            "all_three_runs_complete": True,
        },
        "sources": {
            str(MANIFEST.relative_to(ROOT)): sha256_file(MANIFEST),
            str(ACCEPTANCE.relative_to(ROOT)): sha256_file(ACCEPTANCE),
            "revealnav_mf2r4/model.py": sha256_file(ROOT / "revealnav_mf2r4/model.py"),
            "revealnav_mf2r4/losses.py": sha256_file(ROOT / "revealnav_mf2r4/losses.py"),
            "revealnav_mf2r4/data.py": sha256_file(ROOT / "revealnav_mf2r4/data.py"),
            "artifacts/design/MF2_BRANCH_EXCURSION_ACTION_Q_V4.md": sha256_file(
                ROOT / "artifacts/design/MF2_BRANCH_EXCURSION_ACTION_Q_V4.md"
            ),
        },
        "development_is_internal_train_partition": True,
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed branch-excursion Q protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "counts": value["counts"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def datasets():
    train_ids, development_ids, _ = partitions()
    return (
        BranchExcursionDataset(MANIFEST, train_ids),
        BranchExcursionDataset(MANIFEST, development_ids),
    )


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def forward(model, batch):
    return model(
        batch["history_embeddings"], batch["candidate_embeddings"],
        batch["candidate_mask"], batch["instruction_embedding"],
        batch["decision_index"],
    )


def train_medians(dataset) -> tuple[float, float]:
    commit, excursion = [], []
    for index in range(len(dataset)):
        row = dataset[index]
        commit.extend(row["commit_cost"].tolist())
        excursion.extend(row["excursion_cost"].tolist())
    return statistics.median(commit), statistics.median(excursion)


def evaluate(model, dataset, device, medians) -> dict:
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    )
    commit_truth, commit_prediction = [], []
    excursion_truth, excursion_prediction = [], []
    gain_truth, gain_prediction = [], []
    model_regret, median_regret, random_regret = [], [], []
    correct = 0
    random_accuracy = []
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = forward(model, batch)
            for index in range(batch["commit_cost"].shape[0]):
                valid = torch.isfinite(batch["commit_cost"][index])
                tc = batch["commit_cost"][index, valid]
                te = batch["excursion_cost"][index, valid]
                pc = output.commit_cost[index, valid]
                pe = output.excursion_cost[index, valid]
                commit_truth.extend(tc.cpu().tolist())
                commit_prediction.extend(pc.cpu().tolist())
                excursion_truth.extend(te.cpu().tolist())
                excursion_prediction.extend(pe.cpu().tolist())
                gain_truth.extend((tc - te).cpu().tolist())
                gain_prediction.extend((pc - pe).cpu().tolist())
                teacher = torch.cat((tc, te))
                predicted = torch.cat((pc, pe))
                oracle = float(teacher.min())
                selected = int(predicted.argmin())
                model_cost = float(teacher[selected])
                correct += int(model_cost <= oracle + 1e-6)
                model_regret.append(model_cost - oracle)
                median_prediction = torch.cat((
                    torch.full_like(tc, medians[0]),
                    torch.full_like(te, medians[1]),
                ))
                median_regret.append(float(teacher[int(median_prediction.argmin())]) - oracle)
                random_regret.append(float(teacher.mean()) - oracle)
                random_accuracy.append(float((teacher <= oracle + 1e-6).float().mean()))
    tc = np.asarray(commit_truth); pc = np.asarray(commit_prediction)
    te = np.asarray(excursion_truth); pe = np.asarray(excursion_prediction)
    gt = np.asarray(gain_truth); gp = np.asarray(gain_prediction)
    labels = (gt > 1e-6).astype(np.int64)
    events = len(model_regret)
    return {
        "commit_mae": float(np.mean(np.abs(tc - pc))),
        "commit_train_median_mae": float(np.mean(np.abs(tc - medians[0]))),
        "excursion_mae": float(np.mean(np.abs(te - pe))),
        "excursion_train_median_mae": float(np.mean(np.abs(te - medians[1]))),
        "preservation_gain_mae": float(np.mean(np.abs(gt - gp))),
        "preservation_gain_auc": rank_auc(labels, gp),
        "best_action_accuracy": correct / events,
        "best_action_random_accuracy": statistics.mean(random_accuracy),
        "mean_action_regret": statistics.mean(model_regret),
        "median_selector_mean_action_regret": statistics.mean(median_regret),
        "random_mean_action_regret": statistics.mean(random_regret),
        "development_events": events,
        "positive_preservation_branches": int(labels.sum()),
        "evaluated_branches": len(labels),
    }


def run(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("seed outside branch-excursion protocol")
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("branch-excursion Q protocol must be sealed")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    train_data, development_data = datasets()
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
    objective = BranchExcursionQLoss(0.25, 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss = None; best_state = None; stale = 0; history = []
    for epoch in range(1, 26):
        model.train(); train_sum = 0.0; train_count = 0
        for cpu in train_loader:
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            losses = objective(forward(model, batch), batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite branch-excursion loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = batch["history_embeddings"].shape[0]
            train_sum += float(losses["total"].detach()) * size
            train_count += size
        model.eval(); dev_sum = 0.0; dev_count = 0
        with torch.no_grad():
            for cpu in development_loader:
                batch = move(cpu, device)
                losses = objective(forward(model, batch), batch)
                size = batch["history_embeddings"].shape[0]
                dev_sum += float(losses["total"]) * size
                dev_count += size
        native = dev_sum / dev_count
        history.append({
            "epoch": epoch, "train_total": train_sum / train_count,
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
    medians = train_medians(train_data)
    metrics = evaluate(model, development_data, device, medians)
    checkpoint = run_dir / "branch_excursion_q.pt"
    torch.save({
        "schema_version": "revealnav-mf2-branch-excursion-q-checkpoint/4",
        "seed": seed, "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(MANIFEST),
        "train_medians": {"commit": medians[0], "excursion": medians[1]},
    }, checkpoint)
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-q-run/4",
        "status": "BRANCH_EXCURSION_Q_RUN_COMPLETE",
        "seed": seed, "metrics": metrics, "history": history,
        "train_events": len(train_data),
        "development_events": len(development_data),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "gold_payload_read": False, "paper_result": False,
    }
    atomic_json(run_dir / "result.json", value)
    print(json.dumps({"status": value["status"], "seed": seed, "metrics": metrics}, indent=2))
    return 0


def summary(values):
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values), "values": values,
    }


def aggregate() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("branch-excursion Q protocol drift")
    rows = [
        json.loads((OUT / f"seed_{seed}/result.json").read_text())
        for seed in SEEDS
    ]
    if any(row.get("status") != "BRANCH_EXCURSION_Q_RUN_COMPLETE" for row in rows):
        raise RuntimeError("incomplete branch-excursion Q runs")
    names = [
        key for key, value in rows[0]["metrics"].items()
        if isinstance(value, float)
    ]
    metrics = {
        name: summary([row["metrics"][name] for row in rows]) for name in names
    }
    gates = {
        "all_three_runs_complete": len(rows) == 3,
        "commit_mae_beats_train_median": (
            metrics["commit_mae"]["mean"]
            < metrics["commit_train_median_mae"]["mean"]
        ),
        "excursion_mae_beats_train_median": (
            metrics["excursion_mae"]["mean"]
            < metrics["excursion_train_median_mae"]["mean"]
        ),
        "action_regret_beats_median_selector_in_two_seeds": sum(
            row["metrics"]["mean_action_regret"]
            < row["metrics"]["median_selector_mean_action_regret"]
            for row in rows
        ) >= 2,
        "best_action_accuracy_beats_random_in_two_seeds": sum(
            row["metrics"]["best_action_accuracy"]
            > row["metrics"]["best_action_random_accuracy"]
            for row in rows
        ) >= 2,
        "preservation_gain_auc_above_0_5_in_two_seeds": sum(
            row["metrics"]["preservation_gain_auc"] > 0.5 for row in rows
        ) >= 2,
        "no_gold_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-q-comparison/4",
        "status": (
            "BRANCH_EXCURSION_Q_ENGINEERING_GATE_PASS" if passed
            else "BRANCH_EXCURSION_Q_ENGINEERING_GATE_FAIL"
        ),
        "metrics": metrics, "per_seed": {str(row["seed"]): row for row in rows},
        "gates": gates,
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(MANIFEST),
        "gold_payload_read": False, "paper_result": False,
        "next_gate": "unseen controller integration" if passed else "model diagnosis",
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({"status": value["status"], "gates": gates, "metrics": metrics}, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.aggregate:
        return aggregate()
    if args.seed is None:
        parser.error("--run requires --seed")
    return run(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
