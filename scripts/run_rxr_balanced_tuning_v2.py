#!/usr/bin/env python3
"""Run the frozen development-only balanced Full REE/baseline tuning grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from revealnav_mf2 import (  # noqa: E402
    RevealFeatureDataset,
    RevealOptionHeads,
    RevealOptionLoss,
    RevealOptionLossConfig,
    collate_reveal_examples,
)
from run_rxr_representation_comparison_v2 import (  # noqa: E402
    DirectBaseline,
    classification_metrics,
    collect_probabilities,
    move_batch,
    uad_labels,
)
from train_revealnav_mf2_heads import forward_loss  # noqa: E402


V2 = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
MANIFEST = V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2_AUTHORIZED.json"
AUTHORIZATION = V2 / "RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"
PRIOR = ROOT / (
    "artifacts/evaluation/mf2_representation_v2/"
    "RXR_MULTIBRANCH_REPRESENTATION_COMPARISON_V2.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_balanced_tuning_v2"
PROTOCOL = OUT / "RXR_BALANCED_TUNING_PROTOCOL_V2.json"
SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIMS = (128, 256)
STATE_KEYS = (
    "target_in_set", "separation", "evidence_complete", "reveal_hazard"
)


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


def training_weights(dataset: RevealFeatureDataset):
    binary_counts = {key: [0, 0] for key in STATE_KEYS}
    uad_counts = np.zeros(3, dtype=np.int64)
    for example in dataset:
        for key in STATE_KEYS:
            values = example[key].numpy()
            valid = values >= 0
            binary_counts[key][1] += int((values[valid] >= 0.5).sum())
            binary_counts[key][0] += int((values[valid] < 0.5).sum())
        batch = {key: value.unsqueeze(0) for key, value in example.items()}
        labels = uad_labels(batch).numpy().reshape(-1)
        uad_counts += np.bincount(labels, minlength=3)
    if any(min(counts) < 1 for counts in binary_counts.values()):
        raise RuntimeError("one or more state outputs lack both train classes")
    if np.any(uad_counts == 0):
        raise RuntimeError("one or more U/A/D classes are absent from train")
    state = tuple(
        binary_counts[key][0] / binary_counts[key][1] for key in STATE_KEYS
    )
    direct = uad_counts.sum() / (3.0 * uad_counts.astype(np.float64))
    return state, direct, binary_counts, uad_counts


def loaders(train_set, development_set, seed):
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train_set, batch_size=8, shuffle=True, generator=generator,
            collate_fn=collate_reveal_examples,
        ),
        DataLoader(
            development_set, batch_size=16, shuffle=False,
            collate_fn=collate_reveal_examples,
        ),
    )


def selection_key(metrics: dict, native_loss: float):
    return (
        metrics["macro_f1"],
        -metrics["false_ready_rate"],
        -native_loss,
    )


def train_full(
    seed, hidden_dim, train_set, development_set, state_weights, device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_loader, development_loader = loaders(
        train_set, development_set, seed
    )
    model = RevealOptionHeads(
        feature_dim=768, hidden_dim=hidden_dim, budget_count=4
    ).to(device)
    objective = RevealOptionLoss(RevealOptionLossConfig(
        state_pos_weights=tuple(float(value) for value in state_weights)
    ))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    budgets = torch.tensor([1.5, 2.0, 3.0, 4.0], device=device)
    checkpoint_metadata = {"normalized_budgets": [1.5, 2.0, 3.0, 4.0]}
    history = []
    best_key = None
    best_state = None
    best_metrics = None
    stale_epochs = 0
    for epoch in range(1, 21):
        model.train()
        train_total, train_count = 0.0, 0
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = forward_loss(model, objective, batch, budgets)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite Full REE loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch["step_mask"].sum())
            train_total += float(losses["total"].detach()) * size
            train_count += size
        model.eval()
        development_total, development_count = 0.0, 0
        with torch.no_grad():
            for cpu_batch in development_loader:
                batch = move_batch(cpu_batch, device)
                losses = forward_loss(model, objective, batch, budgets)
                size = int(batch["step_mask"].sum())
                development_total += float(losses["total"]) * size
                development_count += size
        native_loss = development_total / development_count
        labels, probabilities = collect_probabilities(
            "full_ree", model, development_loader, device,
            full_checkpoint=checkpoint_metadata,
        )
        metrics = classification_metrics(labels, probabilities)
        key = selection_key(metrics, native_loss)
        history.append({
            "epoch": epoch,
            "train_total": train_total / train_count,
            "development_total": native_loss,
            "development_macro_f1": metrics["macro_f1"],
            "development_false_ready_rate": metrics["false_ready_rate"],
            "development_missed_ready_rate": metrics["missed_ready_rate"],
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 4:
            break
    model.load_state_dict(best_state, strict=True)
    return model, best_metrics, history


def train_history(
    seed, hidden_dim, train_set, development_set, class_weights, device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_loader, development_loader = loaders(
        train_set, development_set, seed
    )
    model = DirectBaseline(
        history_aware=True, output_dim=3, hidden_dim=hidden_dim
    ).to(device)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    history = []
    best_key = None
    best_state = None
    best_metrics = None
    stale_epochs = 0
    for epoch in range(1, 21):
        model.train()
        train_total, train_count = 0.0, 0
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            mask = batch["step_mask"]
            loss = torch.nn.functional.cross_entropy(
                model(batch)[mask], uad_labels(batch)[mask], weight=weights
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite history baseline loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(mask.sum())
            train_total += float(loss.detach()) * size
            train_count += size
        model.eval()
        development_total, development_count = 0.0, 0
        with torch.no_grad():
            for cpu_batch in development_loader:
                batch = move_batch(cpu_batch, device)
                mask = batch["step_mask"]
                loss = torch.nn.functional.cross_entropy(
                    model(batch)[mask], uad_labels(batch)[mask], weight=weights
                )
                size = int(mask.sum())
                development_total += float(loss) * size
                development_count += size
        native_loss = development_total / development_count
        labels, probabilities = collect_probabilities(
            "history_direct_uad", model, development_loader, device
        )
        metrics = classification_metrics(labels, probabilities)
        key = selection_key(metrics, native_loss)
        history.append({
            "epoch": epoch,
            "train_total": train_total / train_count,
            "development_total": native_loss,
            "development_macro_f1": metrics["macro_f1"],
            "development_false_ready_rate": metrics["false_ready_rate"],
            "development_missed_ready_rate": metrics["missed_ready_rate"],
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 4:
            break
    model.load_state_dict(best_state, strict=True)
    return model, best_metrics, history


def run(seed: int, hidden_dim: int, device: torch.device) -> int:
    if seed not in SEEDS or hidden_dim not in HIDDEN_DIMS:
        raise ValueError("run is outside the frozen tuning grid")
    protocol = json.loads(PROTOCOL.read_text())
    authorization = json.loads(AUTHORIZATION.read_text())
    prior = json.loads(PRIOR.read_text())
    if not (
        protocol.get("status")
        == "DIAGNOSTIC_INFORMED_PROTOCOL_FROZEN_BEFORE_TUNING_RUNS"
        and authorization.get("status") == "TRAINING_AUTHORIZATION_PASS"
        and prior.get("status") == "THREE_SEED_DEVELOPMENT_COMPARISON_COMPLETE"
        and sha256_file(MANIFEST)
        == authorization["training_manifest"]["sha256"]
    ):
        raise RuntimeError("tuning precondition failed")
    torch.use_deterministic_algorithms(True)
    train_set = RevealFeatureDataset(MANIFEST, "train")
    development_set = RevealFeatureDataset(MANIFEST, "development")
    state_weights, class_weights, binary_counts, uad_counts = training_weights(
        train_set
    )
    run_dir = OUT / f"h{hidden_dim}_seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    full, full_metrics, full_history = train_full(
        seed, hidden_dim, train_set, development_set, state_weights, device
    )
    history, history_metrics, history_trace = train_history(
        seed, hidden_dim, train_set, development_set, class_weights, device
    )
    checkpoints = {}
    for name, model, trace in (
        ("balanced_full_ree", full, full_history),
        ("balanced_history_direct_uad", history, history_trace),
    ):
        path = run_dir / f"{name}.pt"
        torch.save({
            "schema_version": "revealnav-mf2-balanced-tuning-checkpoint/2",
            "model_name": name,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "model_state_dict": model.state_dict(),
            "manifest_sha256": sha256_file(MANIFEST),
            "protocol_sha256": sha256_file(PROTOCOL),
            "best_epoch": max(
                trace,
                key=lambda row: (
                    row["development_macro_f1"],
                    -row["development_false_ready_rate"],
                    -row["development_total"],
                ),
            )["epoch"],
        }, path)
        checkpoints[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    result = {
        "schema_version": "revealnav-mf2-balanced-tuning-run/2",
        "status": "BALANCED_TUNING_RUN_COMPLETE",
        "seed": seed,
        "hidden_dim": hidden_dim,
        "sources": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "authorization_sha256": sha256_file(AUTHORIZATION),
            "manifest_sha256": sha256_file(MANIFEST),
            "prior_diagnostic_sha256": sha256_file(PRIOR),
        },
        "train_counts": {
            "binary": binary_counts,
            "uad": dict(zip(("U", "A", "D"), uad_counts.tolist())),
        },
        "weights": {
            "full_state_pos_weights": dict(zip(STATE_KEYS, state_weights)),
            "history_uad_class_weights": dict(
                zip(("U", "A", "D"), class_weights.tolist())
            ),
        },
        "results": {
            "balanced_full_ree": full_metrics,
            "balanced_history_direct_uad": history_metrics,
        },
        "training_history": {
            "balanced_full_ree": full_history,
            "balanced_history_direct_uad": history_trace,
        },
        "checkpoints": checkpoints,
        "gold_read": False,
        "paper_result": False,
    }
    path = run_dir / "result.json"
    atomic_json(path, result)
    print(json.dumps({
        "status": result["status"],
        "seed": seed,
        "hidden_dim": hidden_dim,
        "full_macro_f1": full_metrics["macro_f1"],
        "history_macro_f1": history_metrics["macro_f1"],
        "full_false_ready": full_metrics["false_ready_rate"],
        "history_false_ready": history_metrics["false_ready_rate"],
        "output": str(path.relative_to(ROOT)),
    }, indent=2))
    return 0


def aggregate() -> int:
    rows = []
    for hidden_dim in HIDDEN_DIMS:
        for seed in SEEDS:
            path = OUT / f"h{hidden_dim}_seed_{seed}/result.json"
            value = json.loads(path.read_text())
            if value.get("status") != "BALANCED_TUNING_RUN_COMPLETE":
                raise RuntimeError(f"incomplete run: h{hidden_dim} seed {seed}")
            rows.append(value)
    models = ("balanced_full_ree", "balanced_history_direct_uad")
    metric_names = (
        "accuracy", "macro_f1", "nll", "brier", "ece_10bin",
        "false_ready_rate", "missed_ready_rate",
    )
    grid = {}
    for hidden_dim in HIDDEN_DIMS:
        selected = [row for row in rows if row["hidden_dim"] == hidden_dim]
        grid[str(hidden_dim)] = {}
        for model in models:
            grid[str(hidden_dim)][model] = {}
            for metric in metric_names:
                values = [row["results"][model][metric] for row in selected]
                grid[str(hidden_dim)][model][metric] = {
                    "mean": statistics.mean(values),
                    "population_std": statistics.pstdev(values),
                    "values": values,
                }
    best_full_hidden = max(
        HIDDEN_DIMS,
        key=lambda hidden: grid[str(hidden)]["balanced_full_ree"]["macro_f1"]["mean"],
    )
    best_history_hidden = max(
        HIDDEN_DIMS,
        key=lambda hidden: grid[str(hidden)]["balanced_history_direct_uad"]["macro_f1"]["mean"],
    )
    full_metrics = grid[str(best_full_hidden)]["balanced_full_ree"]
    history_metrics = grid[str(best_history_hidden)][
        "balanced_history_direct_uad"
    ]
    result = {
        "schema_version": "revealnav-mf2-balanced-tuning-aggregate/2",
        "status": "DEVELOPMENT_TUNING_COMPLETE_GOLD_UNTOUCHED",
        "scope": (
            "diagnostic-informed development tuning; not an unbiased test or "
            "paper result"
        ),
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": sha256_file(PROTOCOL),
        },
        "grid": grid,
        "selected": {
            "balanced_full_ree_hidden_dim": best_full_hidden,
            "balanced_history_direct_uad_hidden_dim": best_history_hidden,
            "full_minus_history_macro_f1": (
                full_metrics["macro_f1"]["mean"]
                - history_metrics["macro_f1"]["mean"]
            ),
            "full_minus_history_false_ready_rate": (
                full_metrics["false_ready_rate"]["mean"]
                - history_metrics["false_ready_rate"]["mean"]
            ),
        },
        "run_sources": {
            f"h{row['hidden_dim']}_seed_{row['seed']}": {
                "path": str((OUT / f"h{row['hidden_dim']}_seed_{row['seed']}/result.json").relative_to(ROOT)),
                "sha256": sha256_file(
                    OUT / f"h{row['hidden_dim']}_seed_{row['seed']}/result.json"
                ),
            }
            for row in rows
        },
        "gold_read": False,
        "paper_result": False,
    }
    path = OUT / "RXR_BALANCED_TUNING_AGGREGATE_V2.json"
    atomic_json(path, result)
    print(json.dumps({
        "status": result["status"],
        "selected": result["selected"],
        "full": full_metrics,
        "history": history_metrics,
        "output": str(path.relative_to(ROOT)),
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--aggregate", action="store_true")
    group.add_argument("--seed", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate()
    if args.hidden_dim is None:
        parser.error("--hidden-dim is required with --seed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return run(args.seed, args.hidden_dim, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
