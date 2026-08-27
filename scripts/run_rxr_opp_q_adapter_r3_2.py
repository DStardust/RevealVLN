#!/usr/bin/env python3
"""Train and gate the contract-complete paired-Q adapter for OPP."""

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

from revealnav_mf2r3 import (  # noqa: E402
    PairedQAdapterLoss,
    RelationalRevealExpiryHeads,
    RevealExpiryQFeatureDataset,
    collate_reveal_expiry_q_examples,
)
from run_rxr_representation_comparison_v2 import move_batch  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIM = 128
MANIFEST = ROOT / (
    "artifacts/phase1/rxr_train_expansion/expiry_r3_qpair/"
    "RXR_EXPIRY_R3_Q_FEATURE_MANIFEST.json"
)
R3_1 = ROOT / "artifacts/evaluation/mf2_expiry_r3_1"
R3_1_COMPARISON = R3_1 / "RXR_EXPIRY_R3_COMPARISON.json"
REVISION = ROOT / "artifacts/design/MF2_PAIRED_Q_OBJECTIVE_CORRECTION_R3_2.md"
OUT = ROOT / "artifacts/evaluation/mf2_opp_q_r3_2"
PROTOCOL = OUT / "RXR_OPP_Q_R3_2_PROTOCOL.json"
COMPARISON = OUT / "RXR_OPP_Q_R3_2_COMPARISON.json"
Q_PREFIXES = ("cost_head.", "no_checkpoint_delta_head.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def build_protocol() -> dict:
    comparison = json.loads(R3_1_COMPARISON.read_text())
    sources = {
        str(MANIFEST.relative_to(ROOT)): sha256_file(MANIFEST),
        str(R3_1_COMPARISON.relative_to(ROOT)): sha256_file(R3_1_COMPARISON),
        str(REVISION.relative_to(ROOT)): sha256_file(REVISION),
    }
    checkpoints = {}
    if not (
        comparison.get("status") == "EXPIRY_R3_1_GATE_PASS"
        and comparison.get("selected_condition") == "augmented"
        and comparison.get("gold_payload_read") is False
    ):
        raise RuntimeError("R3.2 precondition failed")
    for seed in SEEDS:
        path = R3_1 / f"augmented_seed_{seed}/relational_expiry_ree.pt"
        checkpoints[str(path.relative_to(ROOT))] = sha256_file(path)
    return {
        "schema_version": "revealnav-mf2-opp-q-protocol/3.2",
        "status": "SEALED_BEFORE_PAIRED_Q_R3_2_TRAINING",
        "seeds": list(SEEDS),
        "condition": "augmented",
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "epoch_limit": 20,
        "early_stopping_patience": 4,
        "loss": {
            "q_with_huber": 1.0,
            "q_without_huber": 1.0,
            "within_prefix_margin_ranking": 0.25,
            "margin": 0.1,
        },
        "trainable_prefixes": list(Q_PREFIXES),
        "selection": "minimum development paired-Q native loss",
        "success_gates": {
            "mean_q_with_mae_beats_train_median": True,
            "mean_q_without_mae_beats_train_median": True,
            "best_option_accuracy_above_random_in_two_seeds": True,
            "opv_mae_beats_zero_in_two_seeds": True,
            "opv_auc_above_0_5_in_two_seeds": True,
            "non_q_state_bit_exact": True,
            "q_order_invariant": "Q_with <= Q_without",
        },
        "sources": sources,
        "input_checkpoints": checkpoints,
        "gold_access_allowed": False,
        "additional_hyperparameter_search_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = build_protocol()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed R3.2 protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "seeds": value["seeds"],
        "sha256": sha256_file(PROTOCOL),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
    }, indent=2))
    return 0


def loaders(seed: int):
    train = RevealExpiryQFeatureDataset(MANIFEST, "train")
    development = RevealExpiryQFeatureDataset(MANIFEST, "development")
    generator = torch.Generator().manual_seed(seed)
    return train, development, DataLoader(
        train, batch_size=8, shuffle=True, generator=generator,
        collate_fn=collate_reveal_expiry_q_examples,
    ), DataLoader(
        development, batch_size=16, shuffle=False,
        collate_fn=collate_reveal_expiry_q_examples,
    )


def forward(model, batch, budgets):
    batch_size, steps = batch["history_embeddings"].shape[:2]
    return model(
        batch["history_embeddings"], batch["candidate_embeddings"],
        batch["candidate_mask"],
        budgets.view(1, 1, -1).expand(batch_size, steps, -1),
        batch["instruction_embedding"],
    )


def non_q_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(Q_PREFIXES):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def scalar_train_medians(dataset) -> tuple[float, float]:
    with_values, without_values = [], []
    for row in dataset:
        mask = row["candidate_mask"]
        with_values.extend(row["option_cost"][mask].tolist())
        without_values.extend(
            row["option_cost_without_checkpoint"][mask].tolist()
        )
    return float(statistics.median(with_values)), float(
        statistics.median(without_values)
    )


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    positive = labels == 1
    p = int(positive.sum())
    n = len(labels) - p
    if not p or not n:
        raise RuntimeError("Q AUROC requires both classes")
    return float((ranks[positive].sum() - p * (p + 1) / 2) / (p * n))


def evaluate(model, dataset, device, medians) -> dict:
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_reveal_expiry_q_examples,
    )
    yw, pw, yn, pn, topv, popv = [], [], [], [], [], []
    correct, random_sum, ranked_steps = 0, 0.0, 0
    budgets = torch.tensor([1.5, 2.0, 3.0, 4.0], device=device)
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = move_batch(cpu, device)
            output = forward(model, batch, budgets)
            steps = int(batch["step_mask"].sum())
            for step in range(steps):
                mask = batch["candidate_mask"][0, step]
                if not bool(mask.any()):
                    continue
                truth_with = batch["option_cost"][0, step, mask]
                truth_without = batch[
                    "option_cost_without_checkpoint"
                ][0, step, mask]
                pred_with = output.option_cost[0, step, mask]
                pred_without = output.option_cost_without_checkpoint[0, step, mask]
                yw.extend(truth_with.cpu().tolist())
                pw.extend(pred_with.cpu().tolist())
                yn.extend(truth_without.cpu().tolist())
                pn.extend(pred_without.cpu().tolist())
                truth_delta = truth_without - truth_with
                pred_delta = pred_without - pred_with
                topv.append(float(truth_delta.max().cpu()))
                popv.append(float(pred_delta.max().cpu()))
                if len(truth_with) >= 2 and float(
                    truth_with.max() - truth_with.min()
                ) > 1e-6:
                    correct += int(
                        int(torch.argmin(truth_with)) == int(torch.argmin(pred_with))
                    )
                    random_sum += 1.0 / len(truth_with)
                    ranked_steps += 1
    yw = np.asarray(yw); pw = np.asarray(pw)
    yn = np.asarray(yn); pn = np.asarray(pn)
    topv = np.asarray(topv); popv = np.asarray(popv)
    labels = (topv > 1e-6).astype(np.int64)
    return {
        "q_with_mae": float(np.mean(np.abs(yw - pw))),
        "q_with_train_median_baseline_mae": float(
            np.mean(np.abs(yw - medians[0]))
        ),
        "q_without_mae": float(np.mean(np.abs(yn - pn))),
        "q_without_train_median_baseline_mae": float(
            np.mean(np.abs(yn - medians[1]))
        ),
        "opv_mae": float(np.mean(np.abs(topv - popv))),
        "opv_zero_baseline_mae": float(np.mean(np.abs(topv))),
        "opv_auc": rank_auc(labels, popv),
        "best_option_accuracy": correct / ranked_steps,
        "best_option_random_accuracy": random_sum / ranked_steps,
        "ranked_steps": ranked_steps,
        "opv_positive_steps": int(labels.sum()),
        "evaluated_steps": len(topv),
        "q_order_violations": int(np.sum(pw > pn + 1e-6)),
    }


def run(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("seed outside R3.2 protocol")
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != build_protocol():
        raise RuntimeError("R3.2 protocol must be sealed without drift")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    train, development, train_loader, development_loader = loaders(seed)
    source = R3_1 / f"augmented_seed_{seed}/relational_expiry_ree.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model = RelationalRevealExpiryHeads(768, HIDDEN_DIM, 4)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(Q_PREFIXES))
    model.to(device)
    state_before = non_q_sha256(model)
    objective = PairedQAdapterLoss(0.25, 0.1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-4,
    )
    budgets = torch.tensor([1.5, 2.0, 3.0, 4.0], device=device)
    history, best_loss, best_state, stale = [], None, None, 0
    for epoch in range(1, 21):
        model.train(); train_sum = 0.0; train_count = 0
        for cpu in train_loader:
            batch = move_batch(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            losses = objective(forward(model, batch, budgets), batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite R3.2 train loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            size = int(batch["step_mask"].sum())
            train_sum += float(losses["total"].detach()) * size
            train_count += size
        model.eval(); dev_sum = 0.0; dev_count = 0
        with torch.no_grad():
            for cpu in development_loader:
                batch = move_batch(cpu, device)
                losses = objective(forward(model, batch, budgets), batch)
                size = int(batch["step_mask"].sum())
                dev_sum += float(losses["total"]) * size
                dev_count += size
        native = dev_sum / dev_count
        history.append({"epoch": epoch, "train_total": train_sum / train_count,
                        "development_total": native})
        if best_loss is None or native < best_loss:
            best_loss = native
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state, strict=True)
    state_after = non_q_sha256(model)
    if state_after != state_before:
        raise RuntimeError("non-Q model state changed in R3.2")
    medians = scalar_train_medians(train)
    metrics = evaluate(model, development, device, medians)
    checkpoint = run_dir / "paired_q_opp.pt"
    torch.save({
        "schema_version": "revealnav-mf2-opp-q-checkpoint/3.2",
        "seed": seed, "condition": "augmented",
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "manifest_sha256": sha256_file(MANIFEST),
        "source_r3_1_sha256": sha256_file(source),
        "train_medians": {"q_with": medians[0], "q_without": medians[1]},
        "non_q_state_sha256": state_after,
    }, checkpoint)
    result = {
        "schema_version": "revealnav-mf2-opp-q-run/3.2",
        "status": "OPP_Q_R3_2_RUN_COMPLETE",
        "seed": seed, "metrics": metrics, "history": history,
        "train_events": len(train), "development_events": len(development),
        "non_q_state_sha256_before": state_before,
        "non_q_state_sha256_after": state_after,
        "non_q_state_bit_exact": state_before == state_after,
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)),
                       "bytes": checkpoint.stat().st_size,
                       "sha256": sha256_file(checkpoint)},
        "gold_payload_read": False, "paper_result": False,
    }
    atomic_json(run_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "seed": seed,
                      "metrics": metrics}, indent=2))
    return 0


def summary(values):
    return {"mean": statistics.mean(values),
            "population_std": statistics.pstdev(values), "values": values}


def aggregate() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != build_protocol():
        raise RuntimeError("R3.2 protocol drift")
    rows = [json.loads((OUT / f"seed_{seed}/result.json").read_text())
            for seed in SEEDS]
    if any(row.get("status") != "OPP_Q_R3_2_RUN_COMPLETE" for row in rows):
        raise RuntimeError("incomplete R3.2 runs")
    names = tuple(rows[0]["metrics"])
    metrics = {name: summary([row["metrics"][name] for row in rows])
               for name in names if isinstance(rows[0]["metrics"][name], float)}
    gates = {
        "mean_q_with_mae_beats_train_median": (
            metrics["q_with_mae"]["mean"]
            < metrics["q_with_train_median_baseline_mae"]["mean"]
        ),
        "mean_q_without_mae_beats_train_median": (
            metrics["q_without_mae"]["mean"]
            < metrics["q_without_train_median_baseline_mae"]["mean"]
        ),
        "best_option_accuracy_above_random_in_two_seeds": sum(
            row["metrics"]["best_option_accuracy"]
            > row["metrics"]["best_option_random_accuracy"]
            for row in rows
        ) >= 2,
        "opv_mae_beats_zero_in_two_seeds": sum(
            row["metrics"]["opv_mae"]
            < row["metrics"]["opv_zero_baseline_mae"] for row in rows
        ) >= 2,
        "opv_auc_above_0_5_in_two_seeds": sum(
            row["metrics"]["opv_auc"] > 0.5 for row in rows
        ) >= 2,
        "non_q_state_bit_exact": all(row["non_q_state_bit_exact"] for row in rows),
        "q_order_invariant": all(
            row["metrics"]["q_order_violations"] == 0 for row in rows
        ),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-opp-q-comparison/3.2",
        "status": "OPP_Q_R3_2_GATE_PASS" if passed else "OPP_Q_R3_2_GATE_FAIL",
        "results": metrics, "gates": gates,
        "selected_seeds": list(SEEDS) if passed else [],
        "sources": {"protocol_sha256": sha256_file(PROTOCOL),
                    "manifest_sha256": sha256_file(MANIFEST)},
        "gold_payload_read": False, "paper_result": False,
        "next_step": "learned ECOG/OPP development evaluation" if passed else
                     "paired-Q diagnosis without Gold access",
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({"status": value["status"], "gates": gates,
                      "results": metrics}, indent=2))
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
        parser.error("--seed is required with --run")
    return run(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
