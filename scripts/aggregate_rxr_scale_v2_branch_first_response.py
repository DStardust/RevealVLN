#!/usr/bin/env python3
"""Aggregate immutable scale-v2 branch-proposer first responses."""

from pathlib import Path

import aggregate_rxr_multiview_first_response as aggregate
import run_rxr_multiview_branch_factory as factory


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/automatic"
INPUT = BASE / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
OUT_DIR = BASE / "branch_factory"

factory.INPUT = INPUT
factory.OUT_DIR = OUT_DIR
factory.RESULT_DIR = OUT_DIR / "results"
factory.RUN_DIR = OUT_DIR / "runs"
aggregate.INPUT = INPUT
aggregate.OUT_DIR = OUT_DIR
aggregate.REPAIR_DIR = OUT_DIR / "first_response_repairs"
aggregate.ACCEPTED = OUT_DIR / "RXR_SCALE_BRANCH_ACCEPTED.json"
aggregate.PRESCREEN = OUT_DIR / "RXR_SCALE_MACHINE_PRESCREEN.json"


if __name__ == "__main__":
    raise SystemExit(aggregate.main())
