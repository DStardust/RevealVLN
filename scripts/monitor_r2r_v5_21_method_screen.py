#!/usr/bin/env python3
"""Print one live status snapshot for the V5.21 method screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_21_method_screen"


def active_workers() -> list[int]:
    marker = "r2r_consensus_exploration_worker_v5_21.py"
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker.encode() in command:
            active.append(int(entry.name))
    return sorted(active)


def main() -> None:
    protocol = json.loads((OUT / "R2R_V5_21_METHOD_SCREEN_PROTOCOL.json").read_text())
    summaries = []
    for path in sorted((OUT / "runs").glob("*/RUN_SUMMARY.json")):
        try:
            summaries.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            pass
    status_path = OUT / "RUN_STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    passed = sum(row.get("status") == "PASS" for row in summaries)
    failed = [
        {"episode_id": row.get("episode_id"), "seed": row.get("seed"), "error": row.get("error")}
        for row in summaries if row.get("status") == "FAIL"
    ]
    print(json.dumps({
        "state": (
            "COMPLETE" if len(summaries) == protocol["treatment_runs"]
            else "RUNNING" if active_workers() else "NOT_RUNNING"
        ),
        "completed": len(summaries),
        "passed": passed,
        "expected": protocol["treatment_runs"],
        "remaining": protocol["treatment_runs"] - len(summaries),
        "active_workers": active_workers(),
        "failures": failed,
        "orchestrator_status": status.get("status", "NOT_STARTED"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
