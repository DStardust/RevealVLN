#!/usr/bin/env python3
"""One-shot monitor for the detached V5.12 development gate."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_12_reversible_dev_gate"


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


summaries = []
for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
    try:
        summaries.append(load(path))
    except (OSError, json.JSONDecodeError):
        pass
passed = [row for row in summaries if row and row.get("status") == "PASS"]
trials = sum(
    row.get("safety_funnel", {}).get("reversible_alternative_trials", 0)
    for row in passed
)
restores = sum(
    row.get("safety_funnel", {}).get("topology_restores", 0)
    for row in passed
)
pid_path = OUT / "ORCHESTRATOR.pid"
pid = int(pid_path.read_text()) if pid_path.is_file() else None
alive = False
if pid is not None:
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        pass
value = {
    "orchestrator_pid": pid,
    "orchestrator_alive": alive,
    "completed_pass": len(passed),
    "expected": 72,
    "failed_summaries": len(summaries) - len(passed),
    "reversible_trials": trials,
    "verified_topology_restores": restores,
    "run_status": load(OUT / "RUN_STATUS.json"),
    "result_status": (
        load(OUT / "R2R_V5_12_REVERSIBLE_DEV_RESULT.json") or {}
    ).get("status"),
    "log": str((OUT / "ORCHESTRATOR.log").relative_to(ROOT)),
}
print(json.dumps(value, indent=2, sort_keys=True))
