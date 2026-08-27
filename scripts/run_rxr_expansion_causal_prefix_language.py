#!/usr/bin/env python3
"""Run strict causal prefix-language closure on expansion survivors."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_causal_prefix_language as gate  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
CAUSAL = BASE / "causal_frontend"
gate.ANALYSIS = CAUSAL / "RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json"
gate.MEDIA = CAUSAL / "RXR_EXPANSION_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
gate.PROMPT = ROOT / (
    "artifacts/phase0/phase0c_cr5_causal_gate/"
    "CR5_CAUSAL_PREFIX_LANGUAGE_PROMPT_V1.md")
gate.GEOMETRY = BASE / "geometry/RXR_EXPANSION_DIRECTED_GEOMETRY.json"
gate.INPUTS = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
gate.RESULT_DIR = CAUSAL / "prefix_language_results"
gate.OUT = CAUSAL / "RXR_EXPANSION_CAUSAL_PREFIX_LANGUAGE_GATE.json"
gate.OUTPUT_REVISION = "rxr-expansion-causal-prefix-language-gate/1"


def main() -> int:
    analysis = json.loads(gate.ANALYSIS.read_text())
    gate.EXPECTED_ANALYSIS_SHA256 = gate.sha256_file(gate.ANALYSIS)
    gate.EXPECTED_MEDIA_SHA256 = gate.sha256_file(gate.MEDIA)
    gate.EXPECTED_EVENT_COUNT = sum(
        row["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
        for row in analysis["events"]
    )
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
