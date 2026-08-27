#!/usr/bin/env python3
"""Project causal 63-degree candidates onto every MF2-CR6 branch."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_cr5_causal_candidates as analysis  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
analysis.GEOMETRY = V2 / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json"
analysis.INPUTS = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
analysis.CONTROLLER = V2 / "RXR_MULTIBRANCH_CONTROLLER_EXECUTION_V2.json"
analysis.SHARDS = BASE / "causal_frontend/frontend_shards"
analysis.OUT = V2 / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json"
analysis.REVIEW_REQUIRED = False
analysis.USE_ALL_BRANCHES = True
analysis.OUTPUT_REVISION = "rxr-multibranch-causal-candidate-analysis/2"
analysis.OUTPUT_SCOPE = "automatic RxR-train all-branch controller survivors"


if __name__ == "__main__":
    raise SystemExit(analysis.main())
