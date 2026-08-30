#!/usr/bin/env python3
"""Fresh non-overlapping RxR val-unseen evaluation for frozen MF3ZJ."""

from __future__ import annotations

from pathlib import Path

import run_rxr_uad_mf3zb_unseen as runner


ROOT = Path(__file__).resolve().parents[1]
runner.ENTRYPOINT = Path(__file__).resolve()
runner.REVISION = "mf3zj"
runner.TAG = "MF3ZJ"
runner.FREEZE_STATUS = "MF3ZJ_VAL_SEEN_FROZEN"
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
runner.FREEZE = ROOT / (
    "artifacts/evaluation/mf3zj_counterfactual_transfer_arbitration_freeze_v1/"
    "MF3ZJ_VAL_SEEN_FREEZE.json"
)
runner.PRIORS = (
    ROOT / (
        "artifacts/evaluation/mf3v_uad_rxr_val_unseen_pilot_v2/"
        "MF3V_RXR_VAL_UNSEEN_PROTOCOL.json"
    ),
    ROOT / (
        "artifacts/evaluation/mf3za_uad_rxr_val_unseen_independent_v1/"
        "MF3ZA_RXR_VAL_UNSEEN_PROTOCOL.json"
    ),
    ROOT / (
        "artifacts/evaluation/mf3zc_uad_rxr_val_unseen_holdout_v1/"
        "MF3ZC_RXR_VAL_UNSEEN_PROTOCOL.json"
    ),
    ROOT / (
        "artifacts/evaluation/mf3ze_uad_rxr_val_unseen_holdout_v1/"
        "MF3ZE_RXR_VAL_UNSEEN_PROTOCOL.json"
    ),
    ROOT / (
        "artifacts/evaluation/mf3zg_uad_rxr_val_unseen_holdout_v1/"
        "MF3ZG_RXR_VAL_UNSEEN_PROTOCOL.json"
    ),
)
runner.OUT = ROOT / "artifacts/evaluation/mf3zj_uad_rxr_val_unseen_holdout_v1"
runner.PROTOCOL = runner.OUT / "MF3ZJ_RXR_VAL_UNSEEN_PROTOCOL.json"
runner.PROGRESS = runner.OUT / "MF3ZJ_RXR_VAL_UNSEEN_PROGRESS.json"
runner.RESULT = runner.OUT / "MF3ZJ_RXR_VAL_UNSEEN_RESULT.json"
runner.SELECTION_SALT = (
    "revealnav-mf3zj-rxr-val-unseen-fresh-forty-per-nonexhausted-scene/1"
)
runner.MAX_EPISODES_PER_SCENE = 40
runner.MIN_EPISODES_PER_SCENE = 0


if __name__ == "__main__":
    raise SystemExit(runner.main())
