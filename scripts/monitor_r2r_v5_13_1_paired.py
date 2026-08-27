#!/usr/bin/env python3
"""Print one live progress snapshot for a V5.13.1 paired evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/evaluation/mf2_r2r_v5_14_net_advantage"
GROUPS = (
    "etp_r1", "v5_6", "net_advantage_only",
    "v5_6_net_advantage", "v5_6_net_advantage_no_return",
)


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    args = parser.parse_args()
    root = BASE / args.split
    selection = load(root / "R2R_V5_13_1_SELECTION.json") or {}
    status = load(root / "RUN_STATUS.json") or {}
    group_counts = {}
    wall_times = []
    failures = 0
    for group in GROUPS:
        passed = 0
        for path in (root / "runs" / group).glob("*/RUN_SUMMARY.json"):
            row = load(path)
            if row and row.get("status") == "PASS":
                passed += 1
                if row.get("wall_time_s") is not None:
                    wall_times.append(float(row["wall_time_s"]))
            else:
                failures += 1
        group_counts[group] = passed
    pid_path = root / "ORCHESTRATOR.pid"
    pid = int(pid_path.read_text()) if pid_path.is_file() else None
    alive = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            pass
    completed = sum(group_counts.values())
    expected = int(status.get("expected", 0))
    slots = max(1, int(status.get("slots", 1)))
    remaining = max(0, expected - completed)
    eta_s = (
        remaining * (sum(wall_times) / len(wall_times)) / slots
        if wall_times else None
    )
    value = {
        "status": status.get("status", "WAITING"),
        "split": args.split,
        "selection_episodes": selection.get("episodes"),
        "orchestrator_pid": pid,
        "orchestrator_alive": alive,
        "completed_pass": completed,
        "expected": expected or None,
        "progress_percent": (
            round(100.0 * completed / expected, 2) if expected else 0.0
        ),
        "group_completed": group_counts,
        "failed_summaries": failures,
        "active": status.get("active", []),
        "eta_minutes": round(eta_s / 60.0, 1) if eta_s is not None else None,
        "attempt_log": str((root / "JOB_ATTEMPTS.json").relative_to(ROOT)),
        "orchestrator_log": str((root / "ORCHESTRATOR.log").relative_to(ROOT)),
        "result_status": (load(root / "R2R_V5_13_1_PAIRED_RESULT.json") or {}).get("status"),
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
