#!/usr/bin/env python3
"""Compact V5.6 full-OPP development progress."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_full_opp_v5_6_seen_active_dev"
protocol = json.loads((OUT / "R2R_FULL_OPP_PROTOCOL_V5_6.json").read_text())
completed = failed = commits = effective = explores = 0
for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
    row = json.loads(path.read_text())
    if row.get("status") == "PASS":
        completed += 1
        controller = row["controller"]
        commits += controller["commit_decisions"]
        effective += controller["effective_commit_interventions"]
        explores += controller["explore_decisions"]
    else:
        failed += 1
running = sum(
    1 for path in (OUT / "runs").glob("*")
    if not (path / "RUN_SUMMARY.json").is_file()
)
print(json.dumps({
    "completed": completed, "expected": protocol["treatment_runs"],
    "running_or_incomplete": running, "failed": failed,
    "commit_decisions": commits,
    "effective_commit_interventions": effective,
    "explore_decisions": explores,
}, indent=2))
