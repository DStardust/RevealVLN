#!/usr/bin/env python3
"""Seal, reproduce, and aggregate scale-v2 resource labels."""

import run_rxr_multibranch_tx_v2 as tx

ROOT = tx.ROOT
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/automatic"
V2 = BASE / "multibranch"
tx.BASE = BASE
tx.V2 = V2
tx.PLAN = V2 / "RXR_SCALE_TX_PLAN.json"
tx.OUT = V2 / "RXR_SCALE_TX_GATE.json"
tx.RUNS = V2 / "tx_runs"
tx.WORKER = ROOT / "scripts/rxr_scale_v2_tx_worker.py"
tx.INDEX = V2 / "RXR_SCALE_TRAINING_INDEX.json"
tx.GEOMETRY = V2 / "RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
tx.CONTROLLER = V2 / "RXR_SCALE_MULTIBRANCH_CONTROLLER.json"
tx.CAUSAL = V2 / "RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json"
tx.LANGUAGE = V2 / "RXR_SCALE_CAUSAL_PREFIX_LANGUAGE_GATE.json"


if __name__ == "__main__":
    raise SystemExit(tx.main())
