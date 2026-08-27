#!/usr/bin/env python3
"""Render causal front-view prefix media for expansion survivors."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_cr5_causal_prefix_media as media  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion/causal_frontend"
media.ANALYSIS = BASE / "RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json"
media.OUT_DIR = BASE
media.MEDIA_DIR = BASE / "private_media"
media.OUT = BASE / "RXR_EXPANSION_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
media.OUTPUT_REVISION = "rxr-expansion-causal-prefix-media/1"


def main() -> int:
    document = json.loads(media.ANALYSIS.read_text())
    media.EXPECTED_ANALYSIS_SHA256 = media.sha256_file(media.ANALYSIS)
    media.EXPECTED_CAUSAL_COUNT = sum(
        row["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
        for row in document["events"]
    )
    return media.main()


if __name__ == "__main__":
    raise SystemExit(main())
