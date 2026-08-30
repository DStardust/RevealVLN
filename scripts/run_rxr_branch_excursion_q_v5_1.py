#!/usr/bin/env python3
"""Train and gate expanded-data BranchExcursion Q heads."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r5 import (  # noqa: E402
    BranchExcursionDataset, BranchExcursionQHead, BranchExcursionQLoss,
    collate_branch_excursion_examples,
)
from run_rxr_opp_q_adapter_r3_2 import atomic_json, rank_auc, sha256_file  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
VARIANTS = ("natural", "source_balanced")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/branch_excursion_v5_1"
MANIFEST = BASE / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V5_1.json"
LABEL_GATE = BASE / "RXR_BRANCH_EXCURSION_LABEL_GATE_V5_1.json"
REVISION = ROOT / "artifacts/design/MF2_BRANCH_EXCURSION_Q_DATA_REVISION_V5.md"
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v5_1"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_Q_PROTOCOL_V5_1.json"
PROGRESS = OUT / "RXR_BRANCH_EXCURSION_Q_PROGRESS_V5_1.json"
COMPARISON = OUT / "RXR_BRANCH_EXCURSION_Q_COMPARISON_V5_1.json"


def is_development_scene(scene_id: str) -> bool:
    return int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 6 == 1


def partitions() -> tuple[set[str], set[str], dict]:
    manifest = json.loads(MANIFEST.read_text())
    train, development = set(), set()
    train_scenes, development_scenes = set(), set()
    source_counts = Counter()
    for row in manifest["records"]:
        development_row = is_development_scene(row["scene_id"])
        selected = development if development_row else train
        scenes = development_scenes if development_row else train_scenes
        split = "development" if development_row else "train"
        selected.add(row["event_id"])
        scenes.add(row["scene_id"])
        source_counts[(split, row["label_source"])] += 1
    counts = {
        "train_events": len(train),
        "development_events": len(development),
        "train_scenes": len(train_scenes),
        "development_scenes": len(development_scenes),
        "source_counts": {f"{split}:{source}": count
                          for (split, source), count in sorted(source_counts.items())},
    }
    if (
        train_scenes & development_scenes
        or counts["train_events"] != 1498
        or counts["development_events"] != 332
        or counts["train_scenes"] != 26
        or counts["development_scenes"] != 9
    ):
        raise RuntimeError(f"expanded branch-excursion partition drift: {counts}")
    return train, development, counts


def protocol_value() -> dict:
    gate = json.loads(LABEL_GATE.read_text())
    train, development, counts = partitions()
    if not (
        gate.get("status") == "BRANCH_EXCURSION_EXPANDED_LABEL_GATE_PASS"
        and gate.get("training_authorized") is True
        and gate["manifest"]["sha256"] == sha256_file(MANIFEST)
        and len(train | development) == 1830
    ):
        raise RuntimeError("expanded Q training precondition failed")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-q-protocol/5.1",
        "status": "SEALED_BEFORE_EXPANDED_Q_TRAINING",
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "partition": "sha256(scene_id) mod 6 == 1 development; otherwise train",
        "counts": counts,
        "architecture": {
            "feature_dim": 768,
            "hidden_dim": 96,
            "age_denominator": 128.0,
            "candidate_set_operator": "masked_mean",
            "causal": True,
            "outputs": ["commit_cost", "checkpointed_excursion_cost"],
        },
        "variants_definition": {
            "natural": "one shuffled pass over the observed source mixture per epoch",
            "source_balanced": (
                "weighted sampling with equal expected event mass per label source"
            ),
        },
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "batch_size": 32,
        "epoch_limit": 35,
        "early_stopping_patience": 7,
        "loss": {
            "commit_huber": 1.0,
            "excursion_huber": 1.0,
            "paired_gap_huber": 1.0,
            "within_event_action_ranking": 0.25,
            "margin": 0.1,
        },
        "checkpoint_selection": "minimum full scene-disjoint development loss",
        "evaluation": {
            "constant_baseline": "tie-aware train commit/excursion medians",
            "primary_human_stratum_reported": True,
            "candidate_permutation_tolerance": 1e-5,
        },
        "offline_success_gates": {
            "overall_action_regret_beats_tie_aware_constant_in_two_seeds": True,
            "human_action_regret_beats_tie_aware_constant_in_two_seeds": True,
            "best_action_accuracy_beats_random_in_two_seeds": True,
            "human_preservation_gain_auc_above_0_5_in_two_seeds": True,
            "candidate_permutation_error_at_most_1e_5": True,
        },
        "remaining_gate": "policy-induced proposal holdout before controller integration",
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                MANIFEST, LABEL_GATE, REVISION,
                ROOT / "revealnav_mf2r5/data.py",
                ROOT / "revealnav_mf2r5/__init__.py",
                ROOT / "revealnav_mf2r4/model.py",
                ROOT / "revealnav_mf2r4/losses.py",
                ROOT / "scripts/run_rxr_branch_excursion_q_v5_1.py",
            )
        },
        "development_is_internal_train_partition": True,
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed expanded Q protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "counts": value["counts"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def datasets() -> tuple[BranchExcursionDataset, BranchExcursionDataset]:
    train_ids, development_ids, _ = partitions()
    return (
        BranchExcursionDataset(MANIFEST, train_ids),
        BranchExcursionDataset(MANIFEST, development_ids),
    )


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def forward(model: BranchExcursionQHead, batch: dict[str, torch.Tensor]):
    return model(
        batch["history_embeddings"], batch["candidate_embeddings"],
        batch["candidate_mask"], batch["instruction_embedding"],
        batch["decision_index"],
    )


def train_medians(dataset: BranchExcursionDataset) -> tuple[float, float]:
    commit, excursion = [], []
    for row in dataset.examples:
        commit.extend(row["commit_cost"].tolist())
        excursion.extend(row["excursion_cost"].tolist())
    return statistics.median(commit), statistics.median(excursion)


def tie_aware_cost(truth: torch.Tensor, prediction: torch.Tensor) -> tuple[float, float]:
    minimum = prediction.min()
    selected = torch.isclose(prediction, minimum, atol=1e-8, rtol=1e-7)
    costs = truth[selected]
    oracle = truth.min()
    return float(costs.mean()), float((costs <= oracle + 1e-6).float().mean())


def evaluate(
    model: BranchExcursionQHead,
    dataset: BranchExcursionDataset,
    device: torch.device,
    medians: tuple[float, float],
) -> dict:
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    )
    commit_truth, commit_prediction = [], []
    excursion_truth, excursion_prediction = [], []
    gain_truth, gain_prediction = [], []
    model_regret, constant_regret, random_regret = [], [], []
    model_accuracy, random_accuracy = [], []
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
                selected_cost, selected_accuracy = tie_aware_cost(teacher, predicted)
                model_regret.append(selected_cost - oracle)
                model_accuracy.append(selected_accuracy)
                constant = torch.cat((
                    torch.full_like(tc, medians[0]),
                    torch.full_like(te, medians[1]),
                ))
                constant_cost, _ = tie_aware_cost(teacher, constant)
                constant_regret.append(constant_cost - oracle)
                random_regret.append(float(teacher.mean()) - oracle)
                random_accuracy.append(float((teacher <= oracle + 1e-6).float().mean()))
    tc = np.asarray(commit_truth); pc = np.asarray(commit_prediction)
    te = np.asarray(excursion_truth); pe = np.asarray(excursion_prediction)
    gt = np.asarray(gain_truth); gp = np.asarray(gain_prediction)
    labels = (gt > 1e-6).astype(np.int64)
    return {
        "commit_mae": float(np.mean(np.abs(tc - pc))),
        "commit_train_median_mae": float(np.mean(np.abs(tc - medians[0]))),
        "excursion_mae": float(np.mean(np.abs(te - pe))),
        "excursion_train_median_mae": float(np.mean(np.abs(te - medians[1]))),
        "preservation_gain_mae": float(np.mean(np.abs(gt - gp))),
        "preservation_gain_auc": rank_auc(labels, gp),
        "best_action_accuracy": statistics.mean(model_accuracy),
        "best_action_random_accuracy": statistics.mean(random_accuracy),
        "mean_action_regret": statistics.mean(model_regret),
        "tie_aware_constant_mean_action_regret": statistics.mean(constant_regret),
        "random_mean_action_regret": statistics.mean(random_regret),
        "development_events": len(model_regret),
        "positive_preservation_branches": int(labels.sum()),
        "evaluated_branches": len(labels),
    }


def permutation_error(
    model: BranchExcursionQHead, dataset: BranchExcursionDataset,
    device: torch.device,
) -> float:
    maximum = 0.0
    model.eval()
    with torch.no_grad():
        for example in dataset.examples:
            batch = move(collate_branch_excursion_examples([example]), device)
            original = forward(model, batch)
            permuted = dict(batch)
            permuted["candidate_embeddings"] = torch.flip(
                batch["candidate_embeddings"], dims=(2,)
            )
            permuted["candidate_mask"] = torch.flip(batch["candidate_mask"], dims=(2,))
            changed = forward(model, permuted)
            maximum = max(
                maximum,
                float((original.commit_cost - torch.flip(changed.commit_cost, (1,))).abs().max()),
                float((original.excursion_cost - torch.flip(changed.excursion_cost, (1,))).abs().max()),
            )
    return maximum


def train_loader(
    dataset: BranchExcursionDataset, variant: str, seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    if variant == "natural":
        return DataLoader(
            dataset, batch_size=32, shuffle=True, generator=generator,
            collate_fn=collate_branch_excursion_examples,
        )
    counts = Counter(dataset.sources)
    weights = torch.tensor([1.0 / counts[source] for source in dataset.sources])
    sampler = WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True, generator=generator,
    )
    return DataLoader(
        dataset, batch_size=32, sampler=sampler,
        collate_fn=collate_branch_excursion_examples,
    )


def run(variant: str, seed: int, device: torch.device) -> int:
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("run is outside sealed expanded Q protocol")
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("expanded Q protocol must be sealed")
    run_dir = OUT / f"{variant}_seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    train_data, development_data = datasets()
    loader = train_loader(train_data, variant, seed)
    dev_loader = DataLoader(
        development_data, batch_size=32, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    )
    model = BranchExcursionQHead(768, 96, 128.0).to(device)
    objective = BranchExcursionQLoss(0.25, 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss = None; best_state = None; stale = 0; history = []
    for epoch in range(1, 36):
        model.train(); train_sum = 0.0; train_count = 0
        for cpu in loader:
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            losses = objective(forward(model, batch), batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite expanded Q loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = batch["history_embeddings"].shape[0]
            train_sum += float(losses["total"].detach()) * size
            train_count += size
        model.eval(); dev_sum = 0.0; dev_count = 0
        with torch.no_grad():
            for cpu in dev_loader:
                batch = move(cpu, device)
                losses = objective(forward(model, batch), batch)
                size = batch["history_embeddings"].shape[0]
                dev_sum += float(losses["total"]) * size
                dev_count += size
        native = dev_sum / dev_count
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
        if stale >= 7:
            break
    model.load_state_dict(best_state, strict=True)
    medians = train_medians(train_data)
    all_metrics = evaluate(model, development_data, device, medians)
    human_ids = {
        row["event_id"] for row in development_data.records
        if row["label_source"] == "primary_human_audited"
    }
    human_metrics = evaluate(
        model, BranchExcursionDataset(MANIFEST, human_ids), device, medians
    )
    maximum_permutation_error = permutation_error(model, development_data, device)
    checkpoint = run_dir / "branch_excursion_q.pt"
    torch.save({
        "schema_version": "revealnav-mf2-branch-excursion-q-checkpoint/5.1",
        "variant": variant,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(MANIFEST),
        "train_medians": {"commit": medians[0], "excursion": medians[1]},
    }, checkpoint)
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-q-run/5.1",
        "status": "BRANCH_EXCURSION_Q_RUN_COMPLETE",
        "variant": variant,
        "seed": seed,
        "metrics": {"all": all_metrics, "primary_human": human_metrics},
        "maximum_candidate_permutation_error": maximum_permutation_error,
        "history": history,
        "train_events": len(train_data),
        "development_events": len(development_data),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "gold_payload_read": False,
        "paper_result": False,
    }
    atomic_json(run_dir / "result.json", value)
    print(json.dumps({
        "status": value["status"], "variant": variant, "seed": seed,
        "metrics": value["metrics"],
        "maximum_candidate_permutation_error": maximum_permutation_error,
    }, indent=2))
    return 0


def variant_gates(rows: list[dict]) -> dict:
    return {
        "all_three_runs_complete": len(rows) == 3,
        "overall_action_regret_beats_tie_aware_constant_in_two_seeds": sum(
            row["metrics"]["all"]["mean_action_regret"]
            < row["metrics"]["all"]["tie_aware_constant_mean_action_regret"]
            for row in rows
        ) >= 2,
        "human_action_regret_beats_tie_aware_constant_in_two_seeds": sum(
            row["metrics"]["primary_human"]["mean_action_regret"]
            < row["metrics"]["primary_human"]["tie_aware_constant_mean_action_regret"]
            for row in rows
        ) >= 2,
        "best_action_accuracy_beats_random_in_two_seeds": sum(
            row["metrics"]["all"]["best_action_accuracy"]
            > row["metrics"]["all"]["best_action_random_accuracy"]
            for row in rows
        ) >= 2,
        "human_preservation_gain_auc_above_0_5_in_two_seeds": sum(
            row["metrics"]["primary_human"]["preservation_gain_auc"] > 0.5
            for row in rows
        ) >= 2,
        "candidate_permutation_error_at_most_1e_5": all(
            row["maximum_candidate_permutation_error"] <= 1e-5 for row in rows
        ),
    }


def aggregate() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("expanded Q protocol drift")
    per_variant = {}
    for variant in VARIANTS:
        rows = [
            json.loads((OUT / f"{variant}_seed_{seed}/result.json").read_text())
            for seed in SEEDS
        ]
        gates = variant_gates(rows)
        per_variant[variant] = {
            "status": "OFFLINE_GATE_PASS" if all(gates.values()) else "OFFLINE_GATE_FAIL",
            "gates": gates,
            "runs": {str(row["seed"]): row for row in rows},
        }
    passing = [name for name, value in per_variant.items()
               if value["status"] == "OFFLINE_GATE_PASS"]
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-q-comparison/5.1",
        "status": "BRANCH_EXCURSION_Q_OFFLINE_GATE_PASS" if passing
                  else "BRANCH_EXCURSION_Q_OFFLINE_GATE_FAIL",
        "passing_variants": passing,
        "variants": per_variant,
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(MANIFEST),
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": "policy-induced proposal holdout" if passing else "model diagnosis",
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({
        "status": value["status"], "passing_variants": passing,
        "gates": {name: row["gates"] for name, row in per_variant.items()},
    }, indent=2))
    return 0 if passing else 1


def write_progress(completed: int, failures: list[str]) -> None:
    atomic_json(PROGRESS, {
        "schema_version": "revealnav-mf2-branch-excursion-q-progress/5.1",
        "status": "RUNNING" if completed < 6 else (
            "COMPLETE" if not failures else "FAILED"
        ),
        "completed": completed,
        "total": 6,
        "failed": len(failures),
        "failed_runs": failures,
    })


def run_all(gpus: tuple[int, ...]) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("expanded Q protocol must be sealed")
    if not gpus:
        raise ValueError("at least one GPU is required")
    jobs = [(variant, seed) for variant in VARIANTS for seed in SEEDS]
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(index: int, variant: str, seed: int) -> tuple[str, int]:
        name = f"{variant}_seed_{seed}"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpus[index % len(gpus)])
        process = subprocess.run(
            [
                sys.executable, __file__, "--run", "--variant", variant,
                "--seed", str(seed), "--device", "cuda:0",
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (logs / f"{name}.log").write_text(process.stdout)
        return name, process.returncode

    failures = []
    completed = 0
    write_progress(completed, failures)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(launch, index, variant, seed)
            for index, (variant, seed) in enumerate(jobs)
        ]
        for future in concurrent.futures.as_completed(futures):
            name, returncode = future.result()
            completed += 1
            if returncode:
                failures.append(name)
            write_progress(completed, failures)
            print(f"TRAIN_PROGRESS {completed}/6 failures={failures}", flush=True)
    if failures:
        return 1
    return aggregate()


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--all", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.aggregate:
        return aggregate()
    if args.all:
        return run_all(tuple(int(value) for value in args.gpus.split(",") if value))
    if args.variant is None or args.seed is None:
        parser.error("--run requires --variant and --seed")
    return run(args.variant, args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
