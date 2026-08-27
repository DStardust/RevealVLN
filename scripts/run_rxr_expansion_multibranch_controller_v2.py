#!/usr/bin/env python3
"""Execute every MF2-CR6 candidate branch twice with the frozen controller."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase0c_cr5_controller_gate as gate  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
gate.GEOMETRY = BASE / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json"
gate.OUT = BASE / "RXR_MULTIBRANCH_CONTROLLER_EXECUTION_V2.json"
gate.OUTPUT_MANIFEST = "RevealNav RxR MF2-CR6 multibranch controller execution"
gate.OUTPUT_REVISION = "rxr-multibranch-controller-execution/2"
gate.OUTPUT_SCOPE = "all branches in RxR-train multibranch geometry survivors"
gate.USE_ALL_BRANCHES = True


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
