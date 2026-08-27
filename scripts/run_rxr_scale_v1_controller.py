#!/usr/bin/env python3
"""Execute every automatic scale-v1 branch with the frozen controller."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path("/mnt/daiyang/vla")
sys.path.insert(0, str(ROOT / "scripts"))
import run_phase0c_cr5_controller_gate as gate  # noqa: E402

BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1/automatic/multibranch"
gate.GEOMETRY = BASE / "RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
gate.OUT = BASE / "RXR_SCALE_MULTIBRANCH_CONTROLLER.json"
gate.OUTPUT_MANIFEST = "RevealNav RxR scale-v1 automatic multibranch controller"
gate.OUTPUT_REVISION = "rxr-scale-v1-multibranch-controller/1"
gate.OUTPUT_SCOPE = "all branches in automatic scale-v1 geometry survivors"
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

