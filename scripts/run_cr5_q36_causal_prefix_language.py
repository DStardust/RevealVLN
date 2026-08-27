#!/usr/bin/env python3
"""Run strict causal prefix-language closure for corrected queue50 event q36."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_causal_prefix_language as gate  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
CAUSAL = BASE / "causal_gate_q36"
REVIEW = BASE / "human_review_fast"
gate.ANALYSIS = CAUSAL / "CR5_Q36_CAUSAL_CANDIDATE_ANALYSIS.json"
gate.MEDIA = CAUSAL / "CR5_Q36_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
gate.GEOMETRY = REVIEW / "CR5_QUEUE50_Q36_CORRECTED_GEOMETRY.json"
gate.INPUTS = BASE / (
    "multiview_primary/CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json")
gate.RESULT_DIR = CAUSAL / "prefix_language_results"
gate.OUT = CAUSAL / "CR5_Q36_CAUSAL_PREFIX_LANGUAGE_GATE.json"
gate.EXPECTED_ANALYSIS_SHA256 = (
    "5e0538c465322eef7f8d6016aed55e339751372a409610aee77d7e1d2614ef09"
)
gate.EXPECTED_MEDIA_SHA256 = (
    "05ef6acfdfa17e51456d055711b8a65ebe6cc2ed828354d93679338bb4f5cf9e"
)
gate.EXPECTED_EVENT_COUNT = 1
gate.OUTPUT_REVISION = "cr5-q36-corrected-causal-prefix-language-gate/1"


if __name__ == "__main__":
    raise SystemExit(gate.main())
