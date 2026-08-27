#!/usr/bin/env python3
"""Run queue50 geometry with the official future as target authority."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_phase0c_cr5_directed_geometry as gate  # noqa: E402


SOURCE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/regrounding_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

gate.INPUT = SOURCE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
gate.PRESCREEN = SOURCE / "CR5_QUEUE50_PRIMARY_MACHINE_PRESCREEN.json"
gate.OUT = OUT_DIR / "CR5_QUEUE50_TARGET_ROUTE_GEOMETRY_V2.json"
gate.EXPECTED_INPUT_SHA256 = (
    "6b70a70e5eb1e25f9522b30209eb56dc2efbf6457377a1aabefdeca6886aee72"
)
gate.EXPECTED_PRESCREEN_SHA256 = (
    "99b8b6f070b73bc65bb6a24268b1c2c436edd08e1789b487807c98cfee648a0f"
)
gate.EXPECTED_SELECTED_COUNT = 50
gate.ELIGIBLE_DISPOSITIONS = {
    "TO_DIRECTED_GEOMETRY",
    "RELOCATE_EARLIER_THEN_3D",
}
gate.OUTPUT_MANIFEST = (
    "MF2-CR5 queue50 official-reference-future target geometry")
gate.OUTPUT_REVISION = "cr5-target-route-authoritative-geometry/2"
gate.OUTPUT_SCOPE = (
    "50 frozen queue primaries from 50 RxR-train trajectories")
gate.TARGET_DIRECTION_POLICY = "official_reference_future_diagnostic"


if __name__ == "__main__":
    raise SystemExit(gate.main())
