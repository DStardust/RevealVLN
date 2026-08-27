#!/usr/bin/env python3
"""Close scale-v2 prefixes with the frozen full-set language gate."""

import json
from pathlib import Path

import run_cr5_causal_prefix_language as gate

ROOT = Path("/mnt/daiyang/vla")
PRIMARY = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/automatic"
MULTIBRANCH = BASE / "multibranch"
gate.ANALYSIS = MULTIBRANCH / "RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json"
gate.MEDIA = MULTIBRANCH / "RXR_SCALE_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
gate.PROMPT = PRIMARY / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_PROMPT_V2.md"
gate.GEOMETRY = MULTIBRANCH / "RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
gate.INPUTS = BASE / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
gate.RESULT_DIR = MULTIBRANCH / "prefix_language_results"
gate.OUT = MULTIBRANCH / "RXR_SCALE_CAUSAL_PREFIX_LANGUAGE_GATE.json"
gate.OUTPUT_REVISION = "rxr-scale-v2-causal-prefix-language-gate/1"
gate.USE_ALL_BRANCHES = True
gate.RESPONSE_SCHEMA_VERSION = "revealnav-fullset-causal-prefix-language-v2"
gate.REQUEST_REVISION = "rxr-scale-v2-causal-prefix-language-request/1"
gate.PAIRWISE_EQUIVALENCE_REUSE = {}
gate.PAIRWISE_REUSE_SOURCE = None


def main() -> int:
    analysis = json.loads(gate.ANALYSIS.read_text())
    media = json.loads(gate.MEDIA.read_text())
    required = {
        row["event_id"]
        for row in analysis["events"]
        if row["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
    }
    if required - set(media["event_ranges"]):
        raise RuntimeError("scale-v2 causal media is incomplete")
    gate.EXPECTED_ANALYSIS_SHA256 = gate.sha256_file(gate.ANALYSIS)
    gate.EXPECTED_MEDIA_SHA256 = gate.sha256_file(gate.MEDIA)
    gate.EXPECTED_EVENT_COUNT = len(required)
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
