#!/usr/bin/env python3
"""Generate one secondary event's per-branch resource evidence."""

import rxr_multibranch_tx_v2_worker as worker


ROOT = worker.ROOT
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
V2 = BASE / "multibranch"
worker.BASE = BASE
worker.V2 = V2
worker.PLAN = V2 / "RXR_SECONDARY_TX_PLAN.json"
worker.GEOMETRY = V2 / "RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json"
worker.CONTROLLER = V2 / "RXR_SECONDARY_MULTIBRANCH_CONTROLLER.json"
worker.CAUSAL = V2 / "RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json"
worker.LANGUAGE = V2 / "RXR_SECONDARY_CAUSAL_PREFIX_LANGUAGE_GATE.json"


if __name__ == "__main__":
    raise SystemExit(worker.main())
