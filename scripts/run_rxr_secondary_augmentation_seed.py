#!/usr/bin/env python3
"""Train one fixed-seed h128 augmented-data ablation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from revealnav_mf2 import RevealFeatureDataset  # noqa: E402
from run_rxr_balanced_tuning_v2 import (  # noqa: E402
    STATE_KEYS,
    train_full,
    train_history,
    training_weights,
)


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
MANIFEST = BASE / "RXR_SECONDARY_AUGMENTED_FEATURE_MANIFEST_V1.json"
AUTH = BASE / "RXR_SECONDARY_AUGMENTED_TRAINING_AUTHORIZATION_V1.json"
PRIMARY_PROTOCOL = ROOT / (
    "artifacts/evaluation/mf2_balanced_tuning_v2/"
    "RXR_BALANCED_TUNING_PROTOCOL_V2.json"
)
PRIMARY_AGGREGATE = ROOT / (
    "artifacts/evaluation/mf2_balanced_tuning_v2/"
    "RXR_BALANCED_TUNING_AGGREGATE_V2.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_secondary_augmentation_v1"
SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIM = 128


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


def best_epoch(trace: list[dict]) -> int:
    return max(
        trace,
        key=lambda row: (
            row["development_macro_f1"],
            -row["development_false_ready_rate"],
            -row["development_total"],
        ),
    )["epoch"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    authorization = json.loads(AUTH.read_text())
    protocol = json.loads(PRIMARY_PROTOCOL.read_text())
    primary_aggregate = json.loads(PRIMARY_AGGREGATE.read_text())
    if not (
        authorization.get("status")
        == "AUGMENTED_DEVELOPMENT_EXPERIMENT_AUTHORIZED"
        and authorization.get("training_authorized") is True
        and authorization["manifest"]["sha256"] == sha256_file(MANIFEST)
        and protocol.get("status")
        == "DIAGNOSTIC_INFORMED_PROTOCOL_FROZEN_BEFORE_TUNING_RUNS"
        and tuple(protocol["seeds"]) == SEEDS
        and protocol["gold_access_allowed"] is False
        and primary_aggregate.get("status")
        == "DEVELOPMENT_TUNING_COMPLETE_GOLD_UNTOUCHED"
        and primary_aggregate["selected"]["balanced_full_ree_hidden_dim"]
        == HIDDEN_DIM
        and primary_aggregate["selected"][
            "balanced_history_direct_uad_hidden_dim"
        ] == HIDDEN_DIM
    ):
        raise RuntimeError("fixed augmentation protocol precondition failed")

    train_set = RevealFeatureDataset(MANIFEST, "train")
    development_set = RevealFeatureDataset(MANIFEST, "development")
    state_weights, class_weights, binary_counts, uad_counts = training_weights(
        train_set
    )
    run_dir = OUT / f"seed_{args.seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)

    full, full_metrics, full_trace = train_full(
        args.seed, HIDDEN_DIM, train_set, development_set,
        state_weights, device,
    )
    history, history_metrics, history_trace = train_history(
        args.seed, HIDDEN_DIM, train_set, development_set,
        class_weights, device,
    )
    checkpoints = {}
    for name, model, trace in (
        ("balanced_full_ree", full, full_trace),
        ("balanced_history_direct_uad", history, history_trace),
    ):
        path = run_dir / f"{name}.pt"
        torch.save({
            "schema_version": "revealnav-mf2-secondary-augmentation-checkpoint/1",
            "condition": "primary_plus_automatic_secondary",
            "model_name": name,
            "seed": args.seed,
            "hidden_dim": HIDDEN_DIM,
            "model_state_dict": model.state_dict(),
            "manifest_sha256": sha256_file(MANIFEST),
            "authorization_sha256": sha256_file(AUTH),
            "primary_protocol_sha256": sha256_file(PRIMARY_PROTOCOL),
            "best_epoch": best_epoch(trace),
        }, path)
        checkpoints[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    result = {
        "schema_version": "revealnav-mf2-secondary-augmentation-run/1",
        "status": "AUGMENTED_DEVELOPMENT_RUN_COMPLETE",
        "condition": "primary_plus_automatic_secondary",
        "seed": args.seed,
        "hidden_dim": HIDDEN_DIM,
        "sources": {
            "manifest_sha256": sha256_file(MANIFEST),
            "authorization_sha256": sha256_file(AUTH),
            "primary_protocol_sha256": sha256_file(PRIMARY_PROTOCOL),
            "primary_aggregate_sha256": sha256_file(PRIMARY_AGGREGATE),
        },
        "dataset_counts": authorization["counts"],
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
            "balanced_full_ree": full_trace,
            "balanced_history_direct_uad": history_trace,
        },
        "checkpoints": checkpoints,
        "secondary_labels_used_for_training_only": True,
        "topology_only_training_included": False,
        "gold_payload_read": False,
        "paper_result": False,
    }
    atomic_json(run_dir / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "seed": args.seed,
        "full_macro_f1": full_metrics["macro_f1"],
        "history_macro_f1": history_metrics["macro_f1"],
        "full_false_ready": full_metrics["false_ready_rate"],
        "history_false_ready": history_metrics["false_ready_rate"],
        "output": str((run_dir / "result.json").relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
