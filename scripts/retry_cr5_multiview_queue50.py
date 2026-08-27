#!/usr/bin/env python3
"""Retry selected queue50 branch proposals without overwriting raw evidence."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_multiview_queue50 as queue_runner  # noqa: E402


runner = queue_runner.runner
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
runner.RESULT_DIR = OUT_DIR / "proposal_retries"
runner.SUMMARY = OUT_DIR / "CR5_QUEUE50_PRIMARY_MULTIVIEW_RETRY_RUN.json"
runner.DRY_RUN = OUT_DIR / "CR5_QUEUE50_PRIMARY_MULTIVIEW_RETRY_DRY_RUN.json"


if __name__ == "__main__":
    raise SystemExit(runner.main())
