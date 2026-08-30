#!/usr/bin/env python3
"""Fixed-cohort RxR val-seen gate for MF3ZJ."""

from __future__ import annotations

from pathlib import Path

import run_rxr_uad_mf3zi_seen as runner


ROOT = Path(__file__).resolve().parents[1]
runner.REVISION = "mf3zj"
runner.TAG = "MF3ZJ"
runner.WORKER = ROOT / "scripts/rxr_uad_mf3zj_worker.py"
runner.SOURCE_DEPENDENCIES = (
    ROOT / "scripts/rxr_uad_mf3zj_controller.py",
    ROOT / "revealnav_mf3/uncertainty_gate.py",
    ROOT / "scripts/rxr_uad_controller_worker_mf3.py",
    ROOT / (
        "artifacts/design/"
        "METHOD_FREEZE_3ZJ_COUNTERFACTUAL_TRANSFER_ARBITRATION.md"
    ),
)
runner.GATE = ROOT / (
    "artifacts/training/mf3zj_counterfactual_transfer_gate_v1/"
    "MF3ZJ_CROSSFIT_GATE.json"
)
runner.OUT = ROOT / (
    "artifacts/evaluation/"
    "mf3zj_counterfactual_transfer_arbitration_rxr_val_seen_v1"
)
runner.PROTOCOL = runner.OUT / "MF3ZJ_RXR_VAL_SEEN_PROTOCOL.json"
runner.PROGRESS = runner.OUT / "MF3ZJ_RXR_VAL_SEEN_PROGRESS.json"
runner.RESULT = runner.OUT / "MF3ZJ_RXR_VAL_SEEN_RESULT.json"
runner.PRIMARY_TREATMENT = (
    "unchanged MF3ZG horizon-consistent UAD primary with a train-only "
    "counterfactual-transfer-screened, globally one-shot runner-up fallback"
)


if __name__ == "__main__":
    raise SystemExit(runner.main())
