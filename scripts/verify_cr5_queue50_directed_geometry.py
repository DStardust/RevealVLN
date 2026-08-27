#!/usr/bin/env python3
"""Run the accepted CR5 directed-geometry gate over the queue50 primaries."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_phase0c_cr5_directed_geometry as gate  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
gate.INPUT = BASE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
gate.PRESCREEN = BASE / "CR5_QUEUE50_PRIMARY_MACHINE_PRESCREEN.json"
gate.OUT = BASE / "CR5_QUEUE50_DIRECTED_GEOMETRY.json"
gate.EXPECTED_INPUT_SHA256 = gate.sha256_file(gate.INPUT)
gate.EXPECTED_PRESCREEN_SHA256 = gate.sha256_file(gate.PRESCREEN)
gate.EXPECTED_SELECTED_COUNT = 50
gate.ELIGIBLE_DISPOSITIONS = {
    "TO_DIRECTED_GEOMETRY",
    "RELOCATE_EARLIER_THEN_3D",
}
gate.OUTPUT_MANIFEST = "MF2-CR5 queue50 deterministic directed 3-D geometry"
gate.OUTPUT_REVISION = "cr5-queue50-directed-geometry/1"
gate.OUTPUT_SCOPE = "50 frozen queue primaries from 50 RxR-train trajectories"


if __name__ == "__main__":
    raise SystemExit(gate.main())
