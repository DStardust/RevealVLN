#!/usr/bin/env python3
"""Analyze frozen frontend outputs for human-accepted queue50 events."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_cr5_causal_candidates as analysis  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
MULTIVIEW = BASE / "multiview_primary"
REVIEW = BASE / "human_review_fast/daiyang_queue50.jsonl"
GATE = BASE / "causal_gate"
analysis.GEOMETRY = MULTIVIEW / "CR5_QUEUE50_DIRECTED_GEOMETRY.json"
analysis.INPUTS = MULTIVIEW / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
analysis.CONTROLLER = MULTIVIEW / "CR5_QUEUE50_CONTROLLER_EXECUTION.json"
analysis.REVIEW = REVIEW
analysis.SHARDS = GATE / "frontend_shards"
analysis.OUT = GATE / "CR5_QUEUE50_CAUSAL_CANDIDATE_ANALYSIS.json"
analysis.EXPECTED_REVIEW_SHA256 = analysis.sha256_file(REVIEW)
analysis.EXPECTED_ACCEPTED_COUNT = 28
analysis.EXPECTED_REJECTED_COUNT = 6
analysis.OUTPUT_REVISION = "cr5-queue50-causal-candidate-analysis/1"
analysis.OUTPUT_SCOPE = "28 human-accepted queue50 RxR-train events"


if __name__ == "__main__":
    raise SystemExit(analysis.main())
