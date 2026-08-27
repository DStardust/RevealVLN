#!/usr/bin/env python3
"""Run one frozen causal-frontend shard for automatic scale-v1 events."""

from pathlib import Path

import run_rxr_expansion_frontend_shard as frontend


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1/automatic"
frontend.GEOMETRY = BASE / "multibranch/RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
frontend.CONTROLLER = BASE / "multibranch/RXR_SCALE_MULTIBRANCH_CONTROLLER.json"
frontend.INPUTS = BASE / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
frontend.OUT_DIR = BASE / "causal_frontend"
frontend.SHARD_DIR = frontend.OUT_DIR / "frontend_shards"
frontend.RUN_DIR = frontend.OUT_DIR / "runs"


if __name__ == "__main__":
    raise SystemExit(frontend.main())

