#!/usr/bin/env python3
"""Execute every secondary branch twice with the frozen controller."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase0c_cr5_controller_gate as gate  # noqa: E402


BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1/"
    "multibranch"
)
gate.GEOMETRY = BASE / "RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json"
gate.OUT = BASE / "RXR_SECONDARY_MULTIBRANCH_CONTROLLER.json"
gate.OUTPUT_MANIFEST = "RevealNav RxR secondary multibranch controller"
gate.OUTPUT_REVISION = "rxr-secondary-multibranch-controller/1"
gate.OUTPUT_SCOPE = "all branches in train-only secondary geometry survivors"
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
