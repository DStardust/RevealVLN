#!/usr/bin/env python3
"""Run the frozen CR5 T_X worker against the RxR expansion inputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cr5_queue50_tx_worker as worker  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
CAUSAL = BASE / "causal_frontend"
PLAN = BASE / "tx_gate/RXR_EXPANSION_TX_PLAN.json"
GEOMETRY = BASE / "geometry/RXR_EXPANSION_DIRECTED_GEOMETRY.json"
CONTROLLER = BASE / "geometry/RXR_EXPANSION_CONTROLLER_EXECUTION.json"
ANALYSIS = CAUSAL / "RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json"
LANGUAGE = CAUSAL / "RXR_EXPANSION_CAUSAL_PREFIX_LANGUAGE_GATE.json"


def configure() -> None:
    plan = json.loads(PLAN.read_text())
    expected_plan_sha = os.environ.get("RXR_TX_PLAN_SHA256")
    if not expected_plan_sha or worker.sha256_file(PLAN) != expected_plan_sha:
        raise RuntimeError("sealed RxR T_X plan SHA mismatch")
    sources = plan["source_sha256"]
    mapping = {
        PLAN: expected_plan_sha,
        GEOMETRY: sources[str(GEOMETRY.relative_to(ROOT))],
        CONTROLLER: sources[str(CONTROLLER.relative_to(ROOT))],
        ANALYSIS: sources[str(ANALYSIS.relative_to(ROOT))],
        LANGUAGE: sources[str(LANGUAGE.relative_to(ROOT))],
    }
    worker.BASE = BASE
    worker.CAUSAL = CAUSAL
    worker.GEOMETRY_PATH = GEOMETRY
    worker.CONTROLLER_PATH = CONTROLLER
    worker.ACCEPTANCE_PATH = PLAN
    worker.ANALYSIS_PATH = ANALYSIS
    worker.LANGUAGE_PATH = LANGUAGE
    worker.EXPECTED_SHA256 = mapping
    worker.EVIDENCE_REVISION = "rxr-expansion-resource-conditioned-tx-event/1"
    worker.RUN_REVISION = "rxr-expansion-tx-worker-run/1"


def main() -> int:
    if "--self-test" not in sys.argv:
        configure()
    return worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
