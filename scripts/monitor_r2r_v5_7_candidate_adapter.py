#!/usr/bin/env python3
"""Compact progress monitor for the concurrent V5.6/V5.7 screens."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V56 = ROOT / "artifacts/evaluation/mf2_r2r_v5_6_fresh_seen_screen"
V57 = ROOT / "artifacts/evaluation/mf2_r2r_v5_7_candidate_adapter_diagnostic"
V58 = ROOT / "artifacts/evaluation/mf2_r2r_v5_8_safe_local_diagnostic"
V510 = ROOT / "artifacts/evaluation/mf2_r2r_v5_10_native_control_diagnostic"
V510_SCREEN = ROOT / "artifacts/evaluation/mf2_r2r_v5_10_fresh_activation_screen"
V510_PAIR = ROOT / "artifacts/evaluation/mf2_r2r_v5_10_paired_seen_gate"


def counts(root: Path) -> dict:
    complete = active = failed = persistent = 0
    for path in (root / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        if row.get("status") != "PASS":
            failed += 1
            continue
        complete += 1
        controller = row["controller"]
        active += int(
            controller["effective_commit_interventions"]
            + controller["explore_decisions"] > 0
        )
        persistent += int(
            row.get("candidate_funnel", {}).get(
                "prefixes_with_two_persistent", 0
            ) > 0
        )
    value = {
        "completed": complete, "active": active, "failed": failed,
        "episodes_with_two_persistent": persistent,
    }
    status = root / "RUN_STATUS.json"
    if status.is_file():
        value["run_status"] = json.loads(status.read_text())
    return value


value = {
    "v5_6_fresh_screen": counts(V56),
    "v5_7_global_adapter_diagnostic": counts(V57),
    "v5_8_safe_local_diagnostic": counts(V58),
    "v5_10_native_control_diagnostic": counts(V510),
    "v5_10_fresh_activation_extension": counts(V510_SCREEN),
    "v5_10_paired_metric_gate": counts(V510_PAIR),
}
result = V57 / "R2R_V5_7_CANDIDATE_ADAPTER_RESULT.json"
if result.is_file():
    value["v5_7_result"] = json.loads(result.read_text())["status"]
v510_result = V510 / "R2R_V5_10_NATIVE_CONTROL_RESULT.json"
if v510_result.is_file():
    value["v5_10_result"] = json.loads(v510_result.read_text())["status"]
print(json.dumps(value, indent=2))
