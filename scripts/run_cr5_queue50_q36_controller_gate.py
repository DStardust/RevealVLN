#!/usr/bin/env python3
"""Run frozen discrete-controller replays for corrected queue50 event q36."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase0c_cr5_controller_gate as gate  # noqa: E402


REVIEW = ROOT / "artifacts/phase0/phase0c_cr5_queue50/human_review_fast"
gate.GEOMETRY = REVIEW / "CR5_QUEUE50_Q36_CORRECTED_GEOMETRY.json"
gate.OUT = REVIEW / "CR5_QUEUE50_Q36_CORRECTED_CONTROLLER.json"
gate.EXPECTED_GEOMETRY_SHA256 = (
    "80ee7482df1cfd821fa1984c1e4cbf8d88d777ce8d1b750ca118712345b9fea3"
)
gate.EXPECTED_CANDIDATE_COUNT = 1
gate.OUTPUT_MANIFEST = "MF2-CR5 q36 corrected discrete controller execution"
gate.OUTPUT_REVISION = "cr5-q36-corrected-controller/1"
gate.OUTPUT_SCOPE = "one human-challenged RxR-train queue50 event"


if __name__ == "__main__":
    raise SystemExit(gate.main())
