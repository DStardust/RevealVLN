#!/usr/bin/env python3
"""Print one live status snapshot for the R2R net-advantage pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/phase1/r2r_train_net_advantage"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("pilot", "full"), default="full")
    args = parser.parse_args()
    root = BASE / args.cohort
    progress_path = root / "R2R_TRAIN_NET_ADVANTAGE_PROGRESS.json"
    if not progress_path.is_file():
        print(f"{args.cohort}: not started")
        return
    value = json.loads(progress_path.read_text())
    completed = value["completed"]
    selected = value["selected"]
    age = max(0.0, time.time() - value["updated_unix"])
    single_samples = []
    batch_samples = []
    for path in root.glob("runs/ep_*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "PASS":
            batch_size = max(1, int(row.get("batch_size", 1)))
            sample = float(row["wall_time_s"]) / batch_size
            (batch_samples if batch_size > 1 else single_samples).append(sample)
    elapsed_samples = batch_samples or single_samples
    mean = sum(elapsed_samples) / len(elapsed_samples) if elapsed_samples else 0.0
    workers = max(1, len(value["active"]))
    eta = (selected - completed) * mean / workers if mean else None
    eta_text = f"{eta/60:.1f} min" if eta is not None else "pending"
    active_episodes = sum(
        int(row.get("episodes", 1)) for row in value["active"].values()
    )
    print(
        f"{args.cohort} {value['status']} | episodes {completed}/{selected} "
        f"({completed/max(1,selected):.1%}) | events {value['feature_events']} | "
        f"zero-event {value['zero_event_episodes']} | "
        f"active batches {len(value['active'])} / episodes {active_episodes} | "
        f"failures {len(value['failures'])} | update age {age:.0f}s | "
        f"ETA {eta_text}"
    )
    for name, relative in (
        ("labels", "labels/R2R_TRAIN_NET_ADVANTAGE_MANIFEST.json"),
        ("training", "training/R2R_SPARSE_NET_ADVANTAGE_TRAINING_RESULT.json"),
    ):
        path = root / relative
        if path.is_file():
            payload = json.loads(path.read_text())
            print(name, payload.get("status"), {
                key: payload.get(key) for key in (
                    "training_rows", "positive_rows", "negative_rows", "selected_seed"
                ) if key in payload
            })


if __name__ == "__main__":
    main()
