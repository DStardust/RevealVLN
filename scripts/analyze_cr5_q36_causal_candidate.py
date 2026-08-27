#!/usr/bin/env python3
"""Project the corrected q36 branches through the frozen causal frontend."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_cr5_causal_candidates as gate  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
REVIEW = BASE / "human_review_fast"
CAUSAL = BASE / "causal_gate_q36"
gate.GEOMETRY = REVIEW / "CR5_QUEUE50_Q36_CORRECTED_GEOMETRY.json"
gate.INPUTS = BASE / (
    "multiview_primary/CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json")
gate.CONTROLLER = REVIEW / "CR5_QUEUE50_Q36_CORRECTED_CONTROLLER.json"
gate.REVIEW = REVIEW / "daiyang_auto_reject16.jsonl"
gate.SHARDS = CAUSAL / "frontend_shard"
gate.OUT = CAUSAL / "CR5_Q36_CAUSAL_CANDIDATE_ANALYSIS.json"
gate.EXPECTED_REVIEW_SHA256 = (
    "fcf17ff60bd9e07fa4e66a83741ea47136b8725bde97603cda03aed76f34f5ff"
)
gate.EXPECTED_ACCEPTED_COUNT = 1
gate.EXPECTED_REJECTED_COUNT = 0
gate.ACCEPTED_REVIEW_LABELS = {"SUSPECT_FALSE_REJECT"}
gate.REJECTED_REVIEW_LABELS = set()
gate.REVIEW_EVENT_FILTER = {"q36_ep1049_hv05"}
gate.OUTPUT_REVISION = "cr5-q36-corrected-causal-candidate-analysis/1"
gate.OUTPUT_SCOPE = (
    "one human-challenged, corrected RxR-train queue50 event")


if __name__ == "__main__":
    raise SystemExit(gate.main())
