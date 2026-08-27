#!/usr/bin/env python3
"""Project the frozen causal frontend onto automatic scale-v1 branches."""

from pathlib import Path

import analyze_cr5_causal_candidates as analysis


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1/automatic"
analysis.GEOMETRY = BASE / "multibranch/RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
analysis.INPUTS = BASE / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
analysis.CONTROLLER = BASE / "multibranch/RXR_SCALE_MULTIBRANCH_CONTROLLER.json"
analysis.SHARDS = BASE / "causal_frontend/frontend_shards"
analysis.OUT = BASE / "multibranch/RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json"
analysis.REVIEW_REQUIRED = False
analysis.USE_ALL_BRANCHES = True
analysis.OUTPUT_REVISION = "rxr-scale-v1-causal-candidate-analysis/1"
analysis.OUTPUT_SCOPE = "automatic train/development scale-v1 controller survivors"


if __name__ == "__main__":
    raise SystemExit(analysis.main())

