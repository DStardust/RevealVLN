#!/usr/bin/env python3
"""Ground automatic scale-v2 branches with directed navmesh geometry."""

import json
from pathlib import Path
import sys

ROOT = Path("/mnt/daiyang/vla")
sys.path.insert(0, str(ROOT / "scripts"))
import verify_phase0c_cr5_directed_geometry as gate  # noqa: E402

BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/automatic"
OUT_DIR = BASE / "multibranch"
gate.INPUT = BASE / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
gate.PRESCREEN = BASE / "branch_factory/RXR_SCALE_MACHINE_PRESCREEN.json"
gate.OUT = OUT_DIR / "RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
gate.ELIGIBLE_DISPOSITIONS = {"TO_DIRECTED_GEOMETRY", "RELOCATE_EARLIER_THEN_3D"}
gate.OUTPUT_MANIFEST = "RevealNav RxR scale-v2 automatic multibranch directed geometry"
gate.OUTPUT_REVISION = "rxr-scale-v2-multibranch-directed-geometry/1"
gate.OUTPUT_SCOPE = "remaining train/development route census candidates"
gate.TARGET_DIRECTION_POLICY = "official_reference_future_diagnostic"
gate.RETAIN_ALL_ALTERNATIVES = True


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prescreen = json.loads(gate.PRESCREEN.read_text())
    gate.EXPECTED_INPUT_SHA256 = gate.sha256_file(gate.INPUT)
    gate.EXPECTED_PRESCREEN_SHA256 = gate.sha256_file(gate.PRESCREEN)
    gate.EXPECTED_SELECTED_COUNT = sum(
        row["prescreen_disposition"] in gate.ELIGIBLE_DISPOSITIONS
        for row in prescreen["events"]
    )
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
