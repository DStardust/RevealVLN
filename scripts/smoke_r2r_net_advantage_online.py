#!/usr/bin/env python3
"""Strict CPU shape smoke for the legacy pilot Net-Advantage checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from revealnav_net_advantage import OnlineNetAdvantageScorer


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/phase1/r2r_train_net_advantage/pilot"
OUTPUT = PILOT / "training/R2R_NET_ADVANTAGE_ONLINE_SMOKE.json"


def main() -> int:
    result = json.loads(
        (PILOT / "training/R2R_SPARSE_NET_ADVANTAGE_TRAINING_RESULT.json").read_text()
    )
    selected = next(
        row for row in result["results"] if row["seed"] == result["selected_seed"]
    )
    checkpoint = ROOT / selected["checkpoint"]["path"]
    scorer = OnlineNetAdvantageScorer.from_checkpoint(
        checkpoint, "cpu", require_online_threshold=False
    )
    manifest = json.loads(
        (PILOT / "labels/R2R_TRAIN_NET_ADVANTAGE_MANIFEST.json").read_text()
    )
    with np.load(ROOT / manifest["arrays"]["path"], allow_pickle=False) as arrays:
        rows = scorer.score_candidates(
            *[
                torch.from_numpy(arrays[key][0].astype(np.float32))
                for key in (
                    "instruction", "current_history", "temporal_history", "native"
                )
            ],
            {"alternative": torch.from_numpy(arrays["alternative"][0].astype(np.float32))},
            float(arrays["immediate_costs"][0, 0]),
            {"alternative": float(arrays["immediate_costs"][0, 1])},
        )
    value = {
        "schema_version": "revealnav-r2r-net-advantage-online-smoke/1",
        "status": "PASS",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "strict_model_load": True,
        "cpu_forward_finite": all(
            np.isfinite(number) for key, number in rows[0].items()
            if key != "branch_id"
        ),
        "output_rows": len(rows),
        "legacy_offline_threshold_rejected": scorer.threshold is None,
        "online_action_selected": False,
        "task_metric_payload_read": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    if not all((value["cpu_forward_finite"], value["legacy_offline_threshold_rejected"])):
        raise RuntimeError("online Net-Advantage smoke failed")
    OUTPUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
