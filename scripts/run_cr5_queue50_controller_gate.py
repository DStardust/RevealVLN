#!/usr/bin/env python3
"""Run the accepted discrete controller gate over queue50 geometry passes."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase0c_cr5_controller_gate as gate  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
gate.GEOMETRY = BASE / "CR5_QUEUE50_DIRECTED_GEOMETRY.json"
gate.OUT = BASE / "CR5_QUEUE50_CONTROLLER_EXECUTION.json"
gate.EXPECTED_GEOMETRY_SHA256 = gate.sha256_file(gate.GEOMETRY)
gate.EXPECTED_CANDIDATE_COUNT = 38
gate.OUTPUT_MANIFEST = "MF2-CR5 queue50 discrete controller execution"
gate.OUTPUT_REVISION = "cr5-queue50-controller-execution/1"
gate.OUTPUT_SCOPE = "38 queue50 candidates passing directed 3-D geometry"


if __name__ == "__main__":
    raise SystemExit(gate.main())
