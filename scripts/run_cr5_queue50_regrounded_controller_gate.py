#!/usr/bin/env python3
"""Replay the four queue50 candidates recovered by v2 geometry."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase0c_cr5_controller_gate as gate  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/regrounding_v2"
gate.GEOMETRY = BASE / "CR5_QUEUE50_REGROUNDED_CANDIDATES.json"
gate.OUT = BASE / "CR5_QUEUE50_REGROUNDED_CONTROLLER.json"
gate.EXPECTED_GEOMETRY_SHA256 = (
    "309670e690d15c4af1f2924cdf3a93c3f9225804f256ba10f065852fcea78b50"
)
gate.EXPECTED_CANDIDATE_COUNT = 4
gate.OUTPUT_MANIFEST = (
    "MF2-CR5 queue50 target-route re-grounded controller execution")
gate.OUTPUT_REVISION = "cr5-target-route-regrounded-controller/1"
gate.OUTPUT_SCOPE = "four queue50 candidates recovered by generic v2 geometry"


if __name__ == "__main__":
    raise SystemExit(gate.main())
