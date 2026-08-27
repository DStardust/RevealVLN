#!/usr/bin/env python3
"""Close secondary causal prefixes using the frozen full-set prompt."""

from __future__ import annotations

import json
from pathlib import Path

import run_cr5_causal_prefix_language as gate


ROOT = Path("/mnt/daiyang/vla")
PRIMARY_V2 = ROOT / (
    "artifacts/phase1/rxr_train_expansion/multibranch_v2"
)
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
MULTIBRANCH = BASE / "multibranch"
gate.ANALYSIS = MULTIBRANCH / (
    "RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json"
)
gate.MEDIA = MULTIBRANCH / (
    "RXR_SECONDARY_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
)
gate.PROMPT = PRIMARY_V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_PROMPT_V2.md"
gate.GEOMETRY = MULTIBRANCH / "RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json"
gate.INPUTS = BASE / "multiview_factory/RXR_SECONDARY_MULTIVIEW_INPUTS.json"
gate.RESULT_DIR = MULTIBRANCH / "prefix_language_results"
gate.OUT = MULTIBRANCH / "RXR_SECONDARY_CAUSAL_PREFIX_LANGUAGE_GATE.json"
gate.OUTPUT_REVISION = "rxr-secondary-causal-prefix-language-gate/1"
gate.USE_ALL_BRANCHES = True
gate.RESPONSE_SCHEMA_VERSION = "revealnav-fullset-causal-prefix-language-v2"
gate.REQUEST_REVISION = "rxr-secondary-causal-prefix-language-request/1"
gate.PAIRWISE_EQUIVALENCE_REUSE = {}
gate.PAIRWISE_REUSE_SOURCE = None


def main() -> int:
    analysis = json.loads(gate.ANALYSIS.read_text())
    media = json.loads(gate.MEDIA.read_text())
    required = {
        row["event_id"] for row in analysis["events"]
        if row["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
    }
    missing = required - set(media["event_ranges"])
    if missing:
        raise SystemExit(f"causal media missing for {len(missing)} events")
    gate.EXPECTED_ANALYSIS_SHA256 = gate.sha256_file(gate.ANALYSIS)
    gate.EXPECTED_MEDIA_SHA256 = gate.sha256_file(gate.MEDIA)
    gate.EXPECTED_EVENT_COUNT = len(required)
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
