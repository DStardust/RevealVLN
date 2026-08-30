#!/usr/bin/env python3
"""MF3ZC specialization of the fresh, non-overlapping unseen holdout runner."""

from __future__ import annotations

from pathlib import Path

import run_rxr_uad_mf3zb_unseen as runner


ROOT = Path(__file__).resolve().parents[1]
runner.ENTRYPOINT = Path(__file__).resolve()
runner.REVISION = "mf3zc"
runner.TAG = "MF3ZC"
runner.FREEZE_STATUS = "MF3ZC_VAL_SEEN_FROZEN"
runner.WORKER = ROOT / "scripts/rxr_uad_mf3zc_unseen_worker.py"
runner.SOURCE_DEPENDENCIES = (
    ROOT / "scripts/rxr_uad_mf3zb_unseen_worker.py",
    ROOT / "scripts/rxr_uad_controller_worker_mf3.py",
    ROOT / "scripts/run_rxr_uad_mf3zb_unseen.py",
)
runner.FREEZE = ROOT / (
    "artifacts/evaluation/mf3zc_calibrated_dissent_freeze_v1/"
    "MF3ZC_VAL_SEEN_FREEZE.json"
)
runner.OUT = ROOT / "artifacts/evaluation/mf3zc_uad_rxr_val_unseen_holdout_v1"
runner.PROTOCOL = runner.OUT / "MF3ZC_RXR_VAL_UNSEEN_PROTOCOL.json"
runner.PROGRESS = runner.OUT / "MF3ZC_RXR_VAL_UNSEEN_PROGRESS.json"
runner.RESULT = runner.OUT / "MF3ZC_RXR_VAL_UNSEEN_RESULT.json"
runner.SELECTION_SALT = "revealnav-mf3zc-rxr-val-unseen-fresh-up-to-ten-per-scene/1"


if __name__ == "__main__":
    raise SystemExit(runner.main())
