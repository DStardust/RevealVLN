#!/usr/bin/env python3
"""Render strict causal-prefix media for corrected queue50 event q36."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_cr5_causal_prefix_media as media  # noqa: E402


GATE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/causal_gate_q36"
media.ANALYSIS = GATE / "CR5_Q36_CAUSAL_CANDIDATE_ANALYSIS.json"
media.EXPECTED_ANALYSIS_SHA256 = (
    "5e0538c465322eef7f8d6016aed55e339751372a409610aee77d7e1d2614ef09"
)
media.OUT_DIR = GATE
media.MEDIA_DIR = GATE / "private_media"
media.OUT = GATE / "CR5_Q36_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
media.EXPECTED_CAUSAL_COUNT = 1
media.OUTPUT_REVISION = "cr5-q36-corrected-causal-prefix-media/1"


if __name__ == "__main__":
    raise SystemExit(media.main())
