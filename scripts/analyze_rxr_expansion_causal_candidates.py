#!/usr/bin/env python3
"""Project frozen frontend outputs onto expansion branches without human labels."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_cr5_causal_candidates as analysis  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
analysis.GEOMETRY = BASE / "geometry/RXR_EXPANSION_DIRECTED_GEOMETRY.json"
analysis.INPUTS = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
analysis.CONTROLLER = BASE / "geometry/RXR_EXPANSION_CONTROLLER_EXECUTION.json"
analysis.SHARDS = BASE / "causal_frontend/frontend_shards"
analysis.OUT = BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json"
analysis.REVIEW_REQUIRED = False
analysis.OUTPUT_REVISION = "rxr-expansion-causal-candidate-analysis/1"
analysis.OUTPUT_SCOPE = "automatic RxR-train expansion controller survivors"


if __name__ == "__main__":
    raise SystemExit(analysis.main())
