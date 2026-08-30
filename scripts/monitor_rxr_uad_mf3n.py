#!/usr/bin/env python3
"""One-shot monitor for the persistent MF3N pipeline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/phase1/mf3n_top2_utility_rank29/dataset_v1/runs"
PROTOCOL = ROOT / (
    "artifacts/phase1/mf3n_top2_utility_rank29/dataset_v1/"
    "MF3B_ONLINE_DATA_PROTOCOL.json"
)
MANIFEST = ROOT / (
    "artifacts/phase1/mf3n_top2_utility_rank29/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DEVELOPMENT = ROOT / (
    "artifacts/evaluation/mf3n_top2_utility_development_v1/"
    "MF3N_DEVELOPMENT_SELECTION.json"
)
GATE = ROOT / (
    "artifacts/evaluation/mf3n_top2_utility_shadow_gate_v1/"
    "MF3N_SHADOW_GATE.json"
)


def main() -> None:
    trained = len(list((ROOT / "artifacts/training/mf3n_top2_utility_v1").glob(
        "hidden_*/seed_*/RESULT.json"
    )))
    print(f"MF3N models: {trained}/6")
    if DEVELOPMENT.exists():
        value = json.loads(DEVELOPMENT.read_text())
        print("development:", value["status"])
        if value.get("selected_rule"):
            print("  rule:", value["selected_rule"])
    else:
        print("development: PENDING")
    if MANIFEST.exists():
        value = json.loads(MANIFEST.read_text())
        print("fresh ranks24-29: COMPLETE", value.get("counts"))
    elif PROTOCOL.exists():
        protocol = json.loads(PROTOCOL.read_text())
        total = len(protocol["records"])
        reused = 1303
        complete = sum(
            (path / "RUN_SUMMARY.json").is_file()
            for path in RUNS.glob("ep_*") if path.is_dir()
        ) if RUNS.exists() else 0
        print(
            "fresh ranks24-29: RUNNING",
            {"new_completed": complete, "new_total": total - reused,
             "remaining": total - reused - complete},
        )
    else:
        print("fresh ranks24-29: NOT_STARTED")
    if GATE.exists():
        value = json.loads(GATE.read_text())
        print("shadow gate:", value["status"], value.get("shadow"))
    else:
        print("shadow gate: UNOPENED")


if __name__ == "__main__":
    main()
