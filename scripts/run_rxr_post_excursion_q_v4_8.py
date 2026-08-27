#!/usr/bin/env python3
"""Train and gate the state-conditioned post-excursion BACKTRACK cost head."""

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
ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r4 import (  # noqa: E402
    PostExcursionDataset, PostExcursionQHead, PostExcursionQLoss,
    collate_post_excursion_examples,
)
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
DATA = ROOT / "artifacts/phase1/rxr_train_expansion/post_excursion_v4_7"
MANIFEST = DATA / "RXR_POST_EXCURSION_FULL_MANIFEST_V4_7.json"
DATA_RESULT = DATA / "RXR_POST_EXCURSION_FULL_RESULT_V4_7.json"
OUT = ROOT / "artifacts/evaluation/mf2_post_excursion_q_v4_8"
PROTOCOL = OUT / "RXR_POST_EXCURSION_Q_PROTOCOL_V4_8.json"
COMPARISON = OUT / "RXR_POST_EXCURSION_Q_COMPARISON_V4_8.json"
DESIGN = ROOT / "artifacts/design/MF2_POST_EXCURSION_Q_TRAINING_V4_8.md"
MODEL_SOURCE = ROOT / "revealnav_mf2r4/post_excursion.py"
DATA_SOURCE = ROOT / "revealnav_mf2r4/post_excursion_data.py"
SCRIPT = ROOT / "scripts/run_rxr_post_excursion_q_v4_8.py"


def protocol_value() -> dict:
    result = json.loads(DATA_RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    train = PostExcursionDataset(MANIFEST, "train")
    development = PostExcursionDataset(MANIFEST, "development")
    if not (
        result.get("status") == "POST_EXCURSION_FULL_GATE_PASS"
        and result.get("training_authorized") is True
        and manifest.get("metadata", {}).get("training_authorized") is True
        and len(train) == 537 and len(development) == 118
    ):
        raise RuntimeError("post-excursion Q precondition failed")
    return {
        "schema_version": "revealnav-mf2-post-excursion-q-protocol/4.8",
        "status": "SEALED_BEFORE_POST_EXCURSION_Q_TRAINING",
        "seeds": list(SEEDS),
        "counts": {"train_examples": len(train),
                   "development_examples": len(development)},
        "partition": "V4.7 scene-disjoint internal train/development partition",
        "architecture": {
            "feature_dim": 768, "hidden_dim": 96,
            "elapsed_denominator": 5.0, "causal": True,
            "outputs": ["continue_cost", "backtrack_cost"],
            "trainable_parameters": "post-excursion head only",
        },
        "optimizer": {"name": "AdamW", "lr": 0.0003,
                      "weight_decay": 0.0001},
        "batch_size": 32, "epoch_limit": 30,
        "early_stopping_patience": 6,
        "loss": {
            "continue_huber": 1.0, "backtrack_huber": 1.0,
            "paired_gap_huber": 1.0,
            "strict_action_ranking": 0.25, "margin": 0.1,
            "ties_excluded_from_ranking": True,
        },
        "selection": "minimum scene-disjoint internal-development native loss",
        "success_gates": {
            "all_three_runs_complete": True,
            "continue_mae_beats_train_median_in_two_seeds": True,
            "backtrack_mae_beats_train_median_in_two_seeds": True,
            "strict_accuracy_beats_median_selector_in_two_seeds": True,
            "mean_action_regret_beats_median_selector_in_two_seeds": True,
        },
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (MANIFEST, DATA_RESULT, DESIGN, MODEL_SOURCE, DATA_SOURCE, SCRIPT)
        },
        "gold_access_allowed": False,
        "evaluation_split_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed post-excursion Q protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "counts": value["counts"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def forward(model, batch):
    return model(
        batch["history_embeddings"], batch["history_length"],
        batch["instruction_embedding"], batch["selected_branch_embedding"],
        batch["checkpoint_embedding"], batch["post_candidate_embedding"],
        batch["normalized_excursion_elapsed"],
    )


def train_medians(dataset) -> tuple[float, float]:
    continue_cost, backtrack_cost = [], []
    for index in range(len(dataset)):
        example = dataset[index]
        continue_cost.append(float(example["continue_cost"]))
        backtrack_cost.append(float(example["backtrack_cost"]))
    return statistics.median(continue_cost), statistics.median(backtrack_cost)


def evaluate(model, dataset, device, medians) -> dict:
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=collate_post_excursion_examples,
    )
    truth_continue, truth_backtrack = [], []
    predicted_continue, predicted_backtrack = [], []
    model_regret, median_regret = [], []
    strict_model_correct = 0
    strict_median_correct = 0
    strict_count = 0
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = forward(model, batch)
            truth = torch.stack((batch["continue_cost"], batch["backtrack_cost"]), -1)
            predicted = torch.stack((output.continue_cost, output.backtrack_cost), -1)
            median_prediction = torch.tensor(
                medians, device=device, dtype=predicted.dtype
            ).view(1, 2).expand_as(predicted)
            oracle = truth.min(-1).values
            model_choice = predicted.argmin(-1)
            median_choice = median_prediction.argmin(-1)
            model_cost = truth.gather(-1, model_choice.unsqueeze(-1)).squeeze(-1)
            median_cost = truth.gather(-1, median_choice.unsqueeze(-1)).squeeze(-1)
            strict = (truth[:, 0] - truth[:, 1]).abs() > 1e-6
            teacher = truth.argmin(-1)
            strict_model_correct += int(((model_choice == teacher) & strict).sum())
            strict_median_correct += int(((median_choice == teacher) & strict).sum())
            strict_count += int(strict.sum())
            model_regret.extend((model_cost - oracle).cpu().tolist())
            median_regret.extend((median_cost - oracle).cpu().tolist())
            truth_continue.extend(truth[:, 0].cpu().tolist())
            truth_backtrack.extend(truth[:, 1].cpu().tolist())
            predicted_continue.extend(predicted[:, 0].cpu().tolist())
            predicted_backtrack.extend(predicted[:, 1].cpu().tolist())
    tc = np.asarray(truth_continue); tb = np.asarray(truth_backtrack)
    pc = np.asarray(predicted_continue); pb = np.asarray(predicted_backtrack)
    return {
        "continue_mae": float(np.abs(tc - pc).mean()),
        "continue_train_median_mae": float(np.abs(tc - medians[0]).mean()),
        "backtrack_mae": float(np.abs(tb - pb).mean()),
        "backtrack_train_median_mae": float(np.abs(tb - medians[1]).mean()),
        "cost_gap_mae": float(np.abs((tb - tc) - (pb - pc)).mean()),
        "strict_action_accuracy": strict_model_correct / strict_count,
        "median_selector_strict_action_accuracy": strict_median_correct / strict_count,
        "mean_action_regret": statistics.mean(model_regret),
        "median_selector_mean_action_regret": statistics.mean(median_regret),
        "strict_examples": strict_count,
        "tie_examples": len(dataset) - strict_count,
        "development_examples": len(dataset),
    }


def run(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("seed outside post-excursion Q protocol")
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("post-excursion Q protocol drift")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    train = PostExcursionDataset(MANIFEST, "train")
    development = PostExcursionDataset(MANIFEST, "development")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=32, shuffle=True, generator=generator,
        collate_fn=collate_post_excursion_examples,
    )
    development_loader = DataLoader(
        development, batch_size=32, shuffle=False,
        collate_fn=collate_post_excursion_examples,
    )
    model = PostExcursionQHead(768, 96, 5.0).to(device)
    objective = PostExcursionQLoss(0.25, 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss = None; best_state = None; stale = 0; history = []
    for epoch in range(1, 31):
        model.train(); train_sum = 0.0; train_count = 0
        for cpu in train_loader:
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            losses = objective(forward(model, batch), batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite post-excursion Q loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = batch["continue_cost"].shape[0]
            train_sum += float(losses["total"].detach()) * size
            train_count += size
        model.eval(); development_sum = 0.0; development_count = 0
        with torch.no_grad():
            for cpu in development_loader:
                batch = move(cpu, device)
                losses = objective(forward(model, batch), batch)
                size = batch["continue_cost"].shape[0]
                development_sum += float(losses["total"]) * size
                development_count += size
        native = development_sum / development_count
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
        if stale >= 6:
            break
    if best_state is None:
        raise RuntimeError("post-excursion Q never produced a checkpoint")
    model.load_state_dict(best_state, strict=True)
    medians = train_medians(train)
    metrics = evaluate(model, development, device, medians)
    checkpoint = run_dir / "post_excursion_q.pt"
    torch.save({
        "schema_version": "revealnav-mf2-post-excursion-q-checkpoint/4.8",
        "seed": seed, "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(MANIFEST),
        "train_medians": {"continue": medians[0], "backtrack": medians[1]},
    }, checkpoint)
    value = {
        "schema_version": "revealnav-mf2-post-excursion-q-run/4.8",
        "status": "POST_EXCURSION_Q_RUN_COMPLETE",
        "seed": seed, "metrics": metrics, "history": history,
        "train_examples": len(train), "development_examples": len(development),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "gold_payload_read": False, "paper_result": False,
    }
    atomic_json(run_dir / "result.json", value)
    print(json.dumps({"status": value["status"], "seed": seed,
                      "metrics": metrics}, indent=2))
    return 0


def summary(values):
    return {"mean": statistics.mean(values),
            "population_std": statistics.pstdev(values), "values": values}


def aggregate() -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("post-excursion Q protocol drift")
    rows = [json.loads((OUT / f"seed_{seed}/result.json").read_text())
            for seed in SEEDS]
    metrics = [row["metrics"] for row in rows]
    wins = {
        "continue_mae": sum(row["continue_mae"] < row[
            "continue_train_median_mae"] for row in metrics),
        "backtrack_mae": sum(row["backtrack_mae"] < row[
            "backtrack_train_median_mae"] for row in metrics),
        "strict_accuracy": sum(row["strict_action_accuracy"] > row[
            "median_selector_strict_action_accuracy"] for row in metrics),
        "action_regret": sum(row["mean_action_regret"] < row[
            "median_selector_mean_action_regret"] for row in metrics),
    }
    gates = {
        "all_three_runs_complete": len(rows) == 3,
        "continue_mae_beats_train_median_in_two_seeds": wins["continue_mae"] >= 2,
        "backtrack_mae_beats_train_median_in_two_seeds": wins["backtrack_mae"] >= 2,
        "strict_accuracy_beats_median_selector_in_two_seeds": wins[
            "strict_accuracy"] >= 2,
        "mean_action_regret_beats_median_selector_in_two_seeds": wins[
            "action_regret"] >= 2,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-post-excursion-q-comparison/4.8",
        "status": (
            "POST_EXCURSION_Q_GATE_PASS" if passed
            else "POST_EXCURSION_Q_GATE_FAIL"
        ),
        "seeds": list(SEEDS), "wins_over_train_median": wins,
        "summary": {
            key: summary([row[key] for row in metrics])
            for key in (
                "continue_mae", "backtrack_mae", "cost_gap_mae",
                "strict_action_accuracy", "mean_action_regret",
            )
        },
        "per_seed": {str(row["seed"]): row["metrics"] for row in rows},
        "gates": gates,
        "checkpoints": {str(row["seed"]): row["checkpoint"] for row in rows},
        "protocol_sha256": sha256_file(PROTOCOL),
        "integration_authorized": passed,
        "gold_payload_read": False, "paper_result": False,
        "next_gate": (
            "controller integration with locked REE/Q and return executor"
            if passed else "diagnose post-excursion head without evaluation tuning"
        ),
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({"status": value["status"], "wins": wins,
                      "summary": value["summary"], "gates": gates}, indent=2))
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
        return aggregate()
    if args.seed is None:
        raise SystemExit("--run requires --seed")
    return run(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
