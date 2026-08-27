#!/usr/bin/env python3
"""Compact progress for V5.6 fresh seen paired confirmation."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_6_fresh_seen_confirm"
protocol_path = OUT / "R2R_V5_6_FRESH_SEEN_CONFIRM_PROTOCOL.json"
if not protocol_path.is_file():
    print(json.dumps({"status": "WAITING_FOR_SCREEN"}, indent=2))
    raise SystemExit(0)
protocol = json.loads(protocol_path.read_text())
complete = failed = effective = explores = 0
for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
    row = json.loads(path.read_text())
    if row.get("status") == "PASS":
        complete += 1
        controller = row["controller"]
        effective += controller["effective_commit_interventions"]
        explores += controller["explore_decisions"]
    else:
        failed += 1
running = sum(
    1 for path in (OUT / "runs").glob("*")
    if not (path / "RUN_SUMMARY.json").is_file()
)
print(json.dumps({
    "completed": complete, "expected": protocol["treatment_runs"],
    "running_or_incomplete": running, "failed": failed,
    "effective_commit_interventions": effective,
    "explore_decisions": explores,
}, indent=2))
