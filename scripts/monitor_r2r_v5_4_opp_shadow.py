#!/usr/bin/env python3
"""Print compact progress for the V5.4 OPP shadow batch."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_4_opp_shadow_seen_active_dev"
protocol_path = OUT / "R2R_V5_4_OPP_SHADOW_PROTOCOL.json"
protocol = json.loads(protocol_path.read_text())
complete = failed = proposals = accepted = 0
for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
    row = json.loads(path.read_text())
    if row.get("status") == "PASS":
        complete += 1
        controller = row.get("controller") or {}
        row_accepted = int(controller.get("opp_checkpoint_acceptances", 0))
        accepted += row_accepted
        proposals += row_accepted + sum(
            controller.get("opp_checkpoint_suppressions", {}).values()
        )
    else:
        failed += 1
running = sum(1 for path in (OUT / "runs").glob("*")
              if not (path / "RUN_SUMMARY.json").is_file())
print(json.dumps({
    "completed": complete, "expected": protocol["expected_runs"],
    "running_or_incomplete": running, "failed": failed,
    "ree_q_proposals": proposals, "opp_accepted": accepted,
}, indent=2))
