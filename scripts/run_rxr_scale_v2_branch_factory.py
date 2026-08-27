#!/usr/bin/env python3
"""Run the frozen branch proposer for one scale-v2 shard."""

from pathlib import Path

import run_rxr_multiview_branch_factory as factory


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/automatic"
factory.INPUT = BASE / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
factory.OUT_DIR = BASE / "branch_factory"
factory.RESULT_DIR = factory.OUT_DIR / "results"
factory.RUN_DIR = factory.OUT_DIR / "runs"


if __name__ == "__main__":
    raise SystemExit(factory.main())
