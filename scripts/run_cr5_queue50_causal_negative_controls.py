#!/usr/bin/env python3
"""Run strict counterfactual controls for queue50 causal candidates."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_causal_negative_controls as controls  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
CAUSAL = BASE / "causal_gate"
MULTIVIEW = BASE / "multiview_primary"
REVIEW = BASE / "human_review_fast/daiyang_queue50.jsonl"

controls.BASELINE = CAUSAL / "CR5_QUEUE50_CAUSAL_PREFIX_LANGUAGE_GATE.json"
controls.ANALYSIS = CAUSAL / "CR5_QUEUE50_CAUSAL_CANDIDATE_ANALYSIS.json"
controls.MEDIA = CAUSAL / "CR5_QUEUE50_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
controls.GEOMETRY = MULTIVIEW / "CR5_QUEUE50_DIRECTED_GEOMETRY.json"
controls.INPUTS = MULTIVIEW / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
controls.HUMAN_REVIEW = REVIEW
controls.CONTROL_DIR = CAUSAL / "negative_control_results"
controls.MASK_DIR = CAUSAL / "control_media/removed_reveal_views"
controls.OUT = CAUSAL / "CR5_QUEUE50_CAUSAL_NEGATIVE_CONTROLS.json"

controls.baseline.ANALYSIS = controls.ANALYSIS
controls.baseline.MEDIA = controls.MEDIA
controls.baseline.GEOMETRY = controls.GEOMETRY
controls.baseline.INPUTS = controls.INPUTS
controls.baseline.EXPECTED_ANALYSIS_SHA256 = controls.sha256_file(
    controls.ANALYSIS)
controls.baseline.EXPECTED_MEDIA_SHA256 = controls.sha256_file(controls.MEDIA)
controls.baseline.EXPECTED_EVENT_COUNT = 18

controls.EXPECTED = {
    controls.BASELINE: controls.sha256_file(controls.BASELINE),
    controls.ANALYSIS: controls.sha256_file(controls.ANALYSIS),
    controls.MEDIA: controls.sha256_file(controls.MEDIA),
    controls.HUMAN_REVIEW: controls.sha256_file(controls.HUMAN_REVIEW),
}
controls.EXPECTED_CANDIDATE_COUNT = 17
controls.OUTPUT_REVISION = "cr5-queue50-causal-negative-controls/1"
controls.OUTPUT_STATUS = "QUEUE50_CAUSAL_CONTROL_COMPLETE"
controls.OUTPUT_SCOPE = "17 queue50 train-only causal candidates"


if __name__ == "__main__":
    raise SystemExit(controls.main())
