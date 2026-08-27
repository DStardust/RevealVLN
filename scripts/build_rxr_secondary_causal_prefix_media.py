#!/usr/bin/env python3
"""Render causal 63-degree prefix media for secondary branch survivors."""

from __future__ import annotations

import json
from pathlib import Path

import build_cr5_causal_prefix_media as media


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
media.ANALYSIS = BASE / (
    "multibranch/RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json"
)
media.OUT_DIR = BASE / "multibranch"
media.MEDIA_DIR = BASE / "multibranch/private_media"
media.OUT = BASE / (
    "multibranch/RXR_SECONDARY_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
)
media.OUTPUT_REVISION = "rxr-secondary-causal-prefix-media/1"
media.USE_ALL_BRANCHES = True
media.REUSE_MEDIA_MANIFEST = None


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
