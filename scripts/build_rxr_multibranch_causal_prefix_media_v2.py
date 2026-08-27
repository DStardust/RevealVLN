#!/usr/bin/env python3
"""Build full-set causal media, reusing identical verified v1 front views."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_cr5_causal_prefix_media as media  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
media.ANALYSIS = V2 / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json"
media.OUT_DIR = V2
media.MEDIA_DIR = V2 / "private_media"
media.OUT = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_MEDIA_MANIFEST_V2.json"
media.OUTPUT_REVISION = "rxr-multibranch-causal-prefix-media/2"
media.USE_ALL_BRANCHES = True
media.REUSE_MEDIA_MANIFEST = BASE / (
    "causal_frontend/RXR_EXPANSION_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
)


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
