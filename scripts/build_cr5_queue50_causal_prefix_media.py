#!/usr/bin/env python3
"""Render causal ego-FOV prefix media for queue50 frontend survivors."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_cr5_causal_prefix_media as media  # noqa: E402


GATE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/causal_gate"
media.ANALYSIS = GATE / "CR5_QUEUE50_CAUSAL_CANDIDATE_ANALYSIS.json"
media.EXPECTED_ANALYSIS_SHA256 = media.sha256_file(media.ANALYSIS)
media.OUT_DIR = GATE
media.MEDIA_DIR = GATE / "private_media"
media.OUT = GATE / "CR5_QUEUE50_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
media.EXPECTED_CAUSAL_COUNT = 18
media.OUTPUT_REVISION = "cr5-queue50-causal-prefix-media/1"


if __name__ == "__main__":
    raise SystemExit(media.main())
