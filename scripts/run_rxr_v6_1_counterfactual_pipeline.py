#!/usr/bin/env python3
"""V6.1 configuration of the reusable RxR V6 pair pipeline."""

from pathlib import Path

import run_rxr_v6_counterfactual_pipeline as base


ROOT = Path(__file__).resolve().parents[1]
base.WORKER = ROOT / "scripts/rxr_v6_1_counterfactual_worker.py"
base.PIPELINE = Path(__file__).resolve()
base.DESIGN = ROOT / (
    "artifacts/design/MF2_POLICY_RELATIVE_REVERSIBLE_ADVANTAGE_V6_1.md"
)
base.MAX_EVENTS_PER_EPISODE = 3
base.CANDIDATE_POLICY = (
    "every 2-4-way identity-stable persistent branch state; frozen ETP native "
    "plus highest causal-REE-probability non-native alternative"
)


if __name__ == "__main__":
    raise SystemExit(base.main())
