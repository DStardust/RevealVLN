#!/usr/bin/env python3
"""Fail closed on first branch-proposer responses for secondary events."""

from pathlib import Path

import aggregate_rxr_multiview_first_response as aggregate
import run_rxr_multiview_branch_factory as factory


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
INPUT = BASE / "multiview_factory/RXR_SECONDARY_MULTIVIEW_INPUTS.json"
OUT_DIR = BASE / "branch_factory"

factory.INPUT = INPUT
factory.OUT_DIR = OUT_DIR
factory.RESULT_DIR = OUT_DIR / "results"
factory.RUN_DIR = OUT_DIR / "runs"

aggregate.INPUT = INPUT
aggregate.OUT_DIR = OUT_DIR
aggregate.REPAIR_DIR = OUT_DIR / "first_response_repairs"
aggregate.ACCEPTED = OUT_DIR / "RXR_SECONDARY_BRANCH_ACCEPTED.json"
aggregate.PRESCREEN = OUT_DIR / "RXR_SECONDARY_MACHINE_PRESCREEN.json"


if __name__ == "__main__":
    raise SystemExit(aggregate.main())
