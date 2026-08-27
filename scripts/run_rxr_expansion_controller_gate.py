#!/usr/bin/env python3
"""Run the accepted deterministic controller on expansion geometry passes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase0c_cr5_controller_gate as gate  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion/geometry"
gate.GEOMETRY = BASE / "RXR_EXPANSION_DIRECTED_GEOMETRY.json"
gate.OUT = BASE / "RXR_EXPANSION_CONTROLLER_EXECUTION.json"
gate.OUTPUT_MANIFEST = "RevealNav RxR expansion controller execution"
gate.OUTPUT_REVISION = "rxr-expansion-controller-execution/1"
gate.OUTPUT_SCOPE = "RxR-train expansion events passing directed 3-D geometry"


def main() -> int:
    geometry = json.loads(gate.GEOMETRY.read_text())
    gate.EXPECTED_GEOMETRY_SHA256 = gate.sha256_file(gate.GEOMETRY)
    gate.EXPECTED_CANDIDATE_COUNT = sum(
        row["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
        for row in geometry["events"]
    )
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
