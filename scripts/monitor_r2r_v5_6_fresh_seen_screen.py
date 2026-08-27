#!/usr/bin/env python3
"""Compact progress for the locked V5.6 fresh seen screen."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_6_fresh_seen_screen"
protocol = json.loads((OUT / "R2R_V5_6_FRESH_SEEN_SCREEN_PROTOCOL.json").read_text())
complete = failed = active = 0
scenes = set()
metadata = {row["episode_id"]: row for row in protocol["eligible"]}
for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
    row = json.loads(path.read_text())
    if row.get("status") == "PASS":
        complete += 1
        controller = row["controller"]
        if controller["effective_commit_interventions"] + controller["explore_decisions"] > 0:
            active += 1
            scenes.add(metadata[str(row["episode_id"])]["scene_id"])
    else:
        failed += 1
running = sum(
    1 for path in (OUT / "runs").glob("*")
    if not (path / "RUN_SUMMARY.json").is_file()
)
print(json.dumps({
    "screened": complete, "eligible": len(protocol["eligible"]),
    "running_or_incomplete": running, "failed": failed,
    "active": active, "target_active": 30,
    "active_scenes": len(scenes), "target_scenes": 20,
}, indent=2))
