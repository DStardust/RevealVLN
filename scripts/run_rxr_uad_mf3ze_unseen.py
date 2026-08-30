#!/usr/bin/env python3
"""MF3ZE specialization of the fresh, non-overlapping unseen runner."""

from __future__ import annotations

from pathlib import Path

import run_rxr_uad_mf3zb_unseen as runner


ROOT = Path(__file__).resolve().parents[1]
runner.ENTRYPOINT = Path(__file__).resolve()
runner.REVISION = "mf3ze"
runner.TAG = "MF3ZE"
runner.FREEZE_STATUS = "MF3ZE_VAL_SEEN_FROZEN"
runner.WORKER = ROOT / "scripts/rxr_uad_mf3ze_unseen_worker.py"
runner.SOURCE_DEPENDENCIES = (
    ROOT / "scripts/rxr_uad_mf3zb_unseen_worker.py",
    ROOT / "scripts/rxr_uad_controller_worker_mf3.py",
    ROOT / "scripts/run_rxr_uad_mf3zb_unseen.py",
    ROOT / "revealnav_mf3/action_aligned.py",
)
runner.FREEZE = ROOT / (
    "artifacts/evaluation/mf3ze_action_aligned_freeze_v1/"
    "MF3ZE_VAL_SEEN_FREEZE.json"
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
)
runner.OUT = ROOT / "artifacts/evaluation/mf3ze_uad_rxr_val_unseen_holdout_v1"
runner.PROTOCOL = runner.OUT / "MF3ZE_RXR_VAL_UNSEEN_PROTOCOL.json"
runner.PROGRESS = runner.OUT / "MF3ZE_RXR_VAL_UNSEEN_PROGRESS.json"
runner.RESULT = runner.OUT / "MF3ZE_RXR_VAL_UNSEEN_RESULT.json"
runner.SELECTION_SALT = (
    "revealnav-mf3ze-rxr-val-unseen-fresh-ten-per-nonexhausted-scene/1"
)
# The six English episodes in pLe4wQe7qrG were all consumed by earlier
# development analyses. Keep the scene in the inventory with zero new rows;
# the other ten unseen scenes provide 100 strictly fresh episodes.
runner.MIN_EPISODES_PER_SCENE = 0


if __name__ == "__main__":
    raise SystemExit(runner.main())
