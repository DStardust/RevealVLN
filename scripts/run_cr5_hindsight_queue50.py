#!/usr/bin/env python3
"""Run the frozen CR5 hindsight locator over the accepted queue50 inputs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase0c_cr5_hindsight_locator as runner  # noqa: E402


OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator"
runner.INPUT = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_INPUTS.json"
runner.ACCEPTANCE = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_INPUTS_ACCEPTANCE.json"
runner.EXPECTED_INPUT_SHA = (
    "8e00000ee306369e305c53d580444e1ac3228a6e94c3c424d84f9db5d16ea151"
)
runner.EXPECTED_ACCEPTANCE_SHA = (
    "8729061a065a37a7d54a93035d27781a4e999c11f7fffedf582ab5373e341fb7"
)
runner.OUT_DIR = OUT_DIR
runner.RESULT_DIR = OUT_DIR / "proposals"
runner.SUMMARY = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_RUN.json"
runner.DRY_RUN = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_DRY_RUN.json"


if __name__ == "__main__":
    raise SystemExit(runner.main())
