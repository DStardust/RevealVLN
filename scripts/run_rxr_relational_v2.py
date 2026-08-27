#!/usr/bin/env python3
"""Train and aggregate the frozen primary-only relational revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2 import RevealFeatureDataset, RevealOptionLossConfig  # noqa: E402
from revealnav_mf2r2 import (  # noqa: E402
    BalancedStructuredUADLoss,
    RelationalRevealOptionHeads,
)
from run_rxr_balanced_tuning_v2 import (  # noqa: E402
    STATE_KEYS,
    loaders,
    selection_key,
    training_weights,
)
from run_rxr_representation_comparison_v2 import (  # noqa: E402
    classification_metrics,
    collect_probabilities,
    move_batch,
)
from train_revealnav_mf2_heads import forward_loss  # noqa: E402


V2 = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
MANIFEST = V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2_AUTHORIZED.json"
AUTHORIZATION = V2 / "RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"
BASELINE = ROOT / "artifacts/evaluation/mf2_balanced_tuning_v2"
BASELINE_AGGREGATE = BASELINE / "RXR_BALANCED_TUNING_AGGREGATE_V2.json"
DIAGNOSTIC = ROOT / (
    "artifacts/evaluation/mf2_structured_uad_v1/"
    "RXR_STRUCTURED_UAD_COMPARISON_V1.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_relational_v2"
PROTOCOL = OUT / "RXR_RELATIONAL_PROTOCOL_V2.json"
AGGREGATE = OUT / "RXR_RELATIONAL_COMPARISON_V2.json"
SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIM = 128
RUN_STATUS = "RELATIONAL_PRIMARY_RUN_COMPLETE"
USES_SECONDARY_EXPANSION = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def preconditions() -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    authorization = json.loads(AUTHORIZATION.read_text())
    baseline = json.loads(BASELINE_AGGREGATE.read_text())
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    if not (
        protocol.get("status") == "FROZEN_BEFORE_RELATIONAL_RUNS"
        and protocol.get("seeds") == list(SEEDS)
        and protocol.get("hidden_dim") == HIDDEN_DIM
        and protocol.get("uses_secondary_expansion") is False
        and protocol.get("gold_access_allowed") is False
        and diagnostic.get("status") == "RELATIONAL_REVISION_REQUIRED"
        and authorization.get("status") == "TRAINING_AUTHORIZATION_PASS"
        and authorization["training_manifest"]["sha256"] == sha256_file(MANIFEST)
        and baseline.get("status")
        == "DEVELOPMENT_TUNING_COMPLETE_GOLD_UNTOUCHED"
    ):
        raise RuntimeError("relational revision precondition failed")
    return protocol


def train(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    protocol = preconditions()
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    train_set = RevealFeatureDataset(MANIFEST, "train")
    development_set = RevealFeatureDataset(MANIFEST, "development")
    state_weights, class_weights, binary_counts, uad_counts = training_weights(
        train_set
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    train_loader, development_loader = loaders(
        train_set, development_set, seed
    )
    model = RelationalRevealOptionHeads(768, HIDDEN_DIM, 4).to(device)
    objective = BalancedStructuredUADLoss(
        RevealOptionLossConfig(
            state_pos_weights=tuple(float(value) for value in state_weights)
        ),
        tuple(float(value) for value in class_weights),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    budgets = torch.tensor([1.5, 2.0, 3.0, 4.0], device=device)
    history = []
    best_key = None
    best_state = None
    best_metrics = None
    stale_epochs = 0
    for epoch in range(1, 21):
        model.train()
        train_total = 0.0
        train_uad = 0.0
        train_count = 0
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = forward_loss(model, objective, batch, budgets)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite relational loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch["step_mask"].sum())
            train_total += float(losses["total"].detach()) * size
            train_uad += float(losses["balanced_uad"].detach()) * size
            train_count += size
        model.eval()
        development_total = 0.0
        development_uad = 0.0
        development_count = 0
        with torch.no_grad():
            for cpu_batch in development_loader:
                batch = move_batch(cpu_batch, device)
                losses = forward_loss(model, objective, batch, budgets)
                size = int(batch["step_mask"].sum())
                development_total += float(losses["total"]) * size
                development_uad += float(losses["balanced_uad"]) * size
                development_count += size
        native_loss = development_total / development_count
        labels, probabilities = collect_probabilities(
            "full_ree", model, development_loader, device,
            full_checkpoint={"normalized_budgets": [1.5, 2.0, 3.0, 4.0]},
        )
        metrics = classification_metrics(labels, probabilities)
        key = selection_key(metrics, native_loss)
        history.append({
            "epoch": epoch,
            "train_total": train_total / train_count,
            "train_balanced_uad": train_uad / train_count,
            "development_total": native_loss,
            "development_balanced_uad": development_uad / development_count,
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
    checkpoint = run_dir / "relational_full_ree.pt"
    best = max(
        history,
        key=lambda row: (
            row["development_macro_f1"],
            -row["development_false_ready_rate"],
            -row["development_total"],
        ),
    )
    torch.save({
        "schema_version": "revealnav-mf2-relational-checkpoint/2",
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "hidden_dim": HIDDEN_DIM,
        "class_weights": class_weights.tolist(),
        "best_epoch": best["epoch"],
        "manifest_sha256": sha256_file(MANIFEST),
        "protocol_sha256": sha256_file(PROTOCOL),
    }, checkpoint)
    result = {
        "schema_version": "revealnav-mf2-relational-run/2",
        "status": RUN_STATUS,
        "seed": seed,
        "hidden_dim": HIDDEN_DIM,
        "results": {"relational_full_ree": best_metrics},
        "training_history": history,
        "train_counts": {
            "binary": binary_counts,
            "uad": dict(zip(("U", "A", "D"), uad_counts.tolist())),
        },
        "weights": {
            "state": dict(zip(STATE_KEYS, state_weights)),
            "uad": dict(zip(("U", "A", "D"), class_weights.tolist())),
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "sources": {
            "manifest_sha256": sha256_file(MANIFEST),
            "protocol_sha256": sha256_file(PROTOCOL),
            "diagnostic_sha256": sha256_file(DIAGNOSTIC),
        },
        "uses_secondary_expansion": USES_SECONDARY_EXPANSION,
        "gold_payload_read": False,
        "paper_result": False,
    }
    atomic_json(run_dir / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "seed": seed,
        "macro_f1": best_metrics["macro_f1"],
        "false_ready_rate": best_metrics["false_ready_rate"],
        "missed_ready_rate": best_metrics["missed_ready_rate"],
    }, indent=2))
    return 0


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "values": values,
    }


def aggregate() -> int:
    preconditions()
    rows = {"relational_full_ree": [], "frozen_full_ree": [],
            "frozen_history_direct_uad": []}
    for seed in SEEDS:
        revised = json.loads((OUT / f"seed_{seed}/result.json").read_text())
        baseline = json.loads(
            (BASELINE / f"h128_seed_{seed}/result.json").read_text()
        )
        if revised.get("status") != "RELATIONAL_PRIMARY_RUN_COMPLETE":
            raise RuntimeError(f"incomplete relational run: {seed}")
        rows["relational_full_ree"].append(
            revised["results"]["relational_full_ree"]
        )
        rows["frozen_full_ree"].append(
            baseline["results"]["balanced_full_ree"]
        )
        rows["frozen_history_direct_uad"].append(
            baseline["results"]["balanced_history_direct_uad"]
        )
    metrics = (
        "accuracy", "macro_f1", "nll", "brier", "ece_10bin",
        "false_ready_rate", "missed_ready_rate",
    )
    results = {
        model: {
            metric: summary([row[metric] for row in model_rows])
            for metric in metrics
        }
        for model, model_rows in rows.items()
    }
    relational_macro = results["relational_full_ree"]["macro_f1"]
    full_macro = results["frozen_full_ree"]["macro_f1"]
    history_macro = results["frozen_history_direct_uad"]["macro_f1"]
    relational_false = results["relational_full_ree"]["false_ready_rate"]
    history_false = results["frozen_history_direct_uad"]["false_ready_rate"]
    gates = {
        "macro_f1_mean_at_least_history": (
            relational_macro["mean"] >= history_macro["mean"]
        ),
        "macro_f1_improves_over_frozen_full_in_at_least_two_seeds": sum(
            revised > frozen for revised, frozen in zip(
                relational_macro["values"], full_macro["values"]
            )
        ) >= 2,
        "mean_false_ready_no_higher_than_history": (
            relational_false["mean"] <= history_false["mean"]
        ),
    }
    solved = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-relational-comparison/2",
        "status": (
            "RELATIONAL_GATE_PASS"
            if solved else "TEMPORAL_REVISION_REQUIRED"
        ),
        "scope": "primary-only development engineering comparison",
        "results": results,
        "deltas": {
            "relational_minus_frozen_full_macro_f1": (
                relational_macro["mean"] - full_macro["mean"]
            ),
            "relational_minus_history_macro_f1": (
                relational_macro["mean"] - history_macro["mean"]
            ),
            "relational_minus_frozen_full_false_ready": (
                relational_false["mean"]
                - results["frozen_full_ree"]["false_ready_rate"]["mean"]
            ),
        },
        "predeclared_success_gates": gates,
        "next_step": (
            "apply relational revision to augmented training"
            if solved else "enter versioned temporal consistency revision"
        ),
        "sources": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "manifest_sha256": sha256_file(MANIFEST),
            "diagnostic_sha256": sha256_file(DIAGNOSTIC),
        },
        "uses_secondary_expansion": False,
        "gold_payload_read": False,
        "paper_result": False,
    }
    atomic_json(AGGREGATE, value)
    print(json.dumps({
        "status": value["status"], "deltas": value["deltas"],
        "gates": gates, "next_step": value["next_step"],
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return train(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
