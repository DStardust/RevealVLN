#!/usr/bin/env python3
"""One-command monitor for fresh screening and paired confirmation."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "artifacts/evaluation/mf2_r2r_v5_6_fresh_seen_screen"
CONFIRM = ROOT / "artifacts/evaluation/mf2_r2r_v5_6_fresh_seen_confirm"
screen_result = SCREEN / "R2R_V5_6_FRESH_SEEN_SCREEN_RESULT.json"
confirm_protocol = CONFIRM / "R2R_V5_6_FRESH_SEEN_CONFIRM_PROTOCOL.json"
confirm_result = CONFIRM / "R2R_V5_6_FRESH_SEEN_CONFIRM_RESULT.json"

if confirm_result.is_file():
    value = json.loads(confirm_result.read_text())
    print(json.dumps({
        "stage": "complete", "status": value["status"],
        "scientific_outcome": value["scientific_outcome"],
        "policy_activity": value["policy_activity"],
        "primary_benefit": {
            metric: value["benefit_deltas_treatment_minus_baseline"][metric]
            for metric in ("success", "spl", "ndtw")
        },
    }, indent=2))
elif confirm_protocol.is_file():
    protocol = json.loads(confirm_protocol.read_text())
    complete = failed = effective = explores = 0
    for path in (CONFIRM / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "PASS":
            complete += 1
            effective += row["controller"]["effective_commit_interventions"]
            explores += row["controller"]["explore_decisions"]
        else:
            failed += 1
    print(json.dumps({
        "stage": "paired_confirmation", "completed": complete,
        "expected": protocol["treatment_runs"], "failed": failed,
        "effective_commit_interventions": effective,
        "explore_decisions": explores,
    }, indent=2))
else:
    protocol = json.loads((
        SCREEN / "R2R_V5_6_FRESH_SEEN_SCREEN_PROTOCOL.json"
    ).read_text())
    metadata = {row["episode_id"]: row for row in protocol["eligible"]}
    complete = failed = active = 0
    scenes = set()
    for path in (SCREEN / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "PASS":
            complete += 1
            controller = row["controller"]
            if controller["effective_commit_interventions"] + controller["explore_decisions"] > 0:
                active += 1
                scenes.add(metadata[str(row["episode_id"])]["scene_id"])
        else:
            failed += 1
    print(json.dumps({
        "stage": "fresh_outcome_blind_screen",
        "screened": complete, "eligible": len(protocol["eligible"]),
        "failed": failed, "active": active, "target_active": 30,
        "active_scenes": len(scenes), "target_scenes": 20,
        "screen_result_ready": screen_result.is_file(),
    }, indent=2))
