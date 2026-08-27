#!/usr/bin/env python3
"""Seal, reproduce, and aggregate secondary per-branch resource labels."""

import run_rxr_multibranch_tx_v2 as tx


ROOT = tx.ROOT
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
V2 = BASE / "multibranch"
tx.BASE = BASE
tx.V2 = V2
tx.PLAN = V2 / "RXR_SECONDARY_TX_PLAN.json"
tx.OUT = V2 / "RXR_SECONDARY_TX_GATE.json"
tx.RUNS = V2 / "tx_runs"
tx.WORKER = ROOT / "scripts/rxr_secondary_tx_worker.py"
tx.INDEX = V2 / "RXR_SECONDARY_TRAINING_INDEX.json"
tx.GEOMETRY = V2 / "RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json"
tx.CONTROLLER = V2 / "RXR_SECONDARY_MULTIBRANCH_CONTROLLER.json"
tx.CAUSAL = V2 / "RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json"
tx.LANGUAGE = V2 / "RXR_SECONDARY_CAUSAL_PREFIX_LANGUAGE_GATE.json"


if __name__ == "__main__":
    raise SystemExit(tx.main())
