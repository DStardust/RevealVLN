#!/usr/bin/env python3
"""Run the frozen branch proposer for one scale-v1 lane."""

import os
from pathlib import Path

import run_rxr_multiview_branch_factory as factory


ROOT = Path("/mnt/daiyang/vla")
LANE = os.environ.get("RXR_SCALE_LANE")
if LANE not in {"automatic", "new_gold"}:
    raise SystemExit("RXR_SCALE_LANE must be automatic or new_gold")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1" / LANE
factory.INPUT = BASE / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
factory.OUT_DIR = BASE / "branch_factory"
factory.RESULT_DIR = factory.OUT_DIR / "results"
factory.RUN_DIR = factory.OUT_DIR / "runs"


if __name__ == "__main__":
    raise SystemExit(factory.main())

