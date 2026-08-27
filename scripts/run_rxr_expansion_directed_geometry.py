#!/usr/bin/env python3
"""Run the accepted target-route geometry gate on expansion survivors."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_phase0c_cr5_directed_geometry as gate  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
OUT_DIR = BASE / "geometry"
gate.INPUT = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
gate.PRESCREEN = BASE / "branch_factory/RXR_MULTIVIEW_MACHINE_PRESCREEN.json"
gate.OUT = OUT_DIR / "RXR_EXPANSION_DIRECTED_GEOMETRY.json"
gate.ELIGIBLE_DISPOSITIONS = {
    "TO_DIRECTED_GEOMETRY",
    "RELOCATE_EARLIER_THEN_3D",
}
gate.OUTPUT_MANIFEST = "RevealNav RxR expansion directed 3-D geometry"
gate.OUTPUT_REVISION = "rxr-expansion-directed-geometry/1"
gate.OUTPUT_SCOPE = "frozen RxR-train expansion primaries passing machine prescreen"
gate.TARGET_DIRECTION_POLICY = "official_reference_future_diagnostic"


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
