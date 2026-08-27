#!/usr/bin/env python3
"""Run fixed multi-view branch proposer on queue50 primary events."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase0c_cr5_multiview_branch as runner  # noqa: E402


OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
runner.INPUT = OUT_DIR / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
runner.ACCEPTANCE = OUT_DIR / (
    "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS_ACCEPTANCE.json"
)
runner.EXPECTED_INPUT_SHA = (
    "6b70a70e5eb1e25f9522b30209eb56dc2efbf6457377a1aabefdeca6886aee72"
)
runner.EXPECTED_ACCEPTANCE_SHA = (
    "73133f833c2d8a64c58d834e6fa063ed38349a306302753c01e1f02cfb734af0"
)
runner.OUT_DIR = OUT_DIR
runner.RESULT_DIR = OUT_DIR / "proposals"
runner.SUMMARY = OUT_DIR / "CR5_QUEUE50_PRIMARY_MULTIVIEW_RUN.json"
runner.DRY_RUN = OUT_DIR / "CR5_QUEUE50_PRIMARY_MULTIVIEW_DRY_RUN.json"


if __name__ == "__main__":
    raise SystemExit(runner.main())
