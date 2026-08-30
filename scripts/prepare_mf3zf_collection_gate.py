#!/usr/bin/env python3
"""Seal the train-only expanded MF3ZF proposal gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


SOURCE = ROOT / (
    "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/MF3V_SHADOW_GATE.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZF_COVERAGE_SAFETY.md"
OUT = ROOT / (
    "artifacts/training/mf3zf_expanded_collection_v1/"
    "MF3ZF_COLLECTION_GATE.json"
)
LOWER = 1.6816482543945312
UPPER = 2.6732332706451416


def main() -> int:
    parent = json.loads(SOURCE.read_text())
    if not (
        parent.get("status") == "SHADOW_GATE_PASS"
        and parent["selected_rule"]["training_score_quantile"] == 0.985
        and parent["selected_rule"]["score_upper_quantile"] == 0.995
        and parent["selected_rule"]["score_upper_threshold"] == UPPER
    ):
        raise RuntimeError("MF3V parent gate drift")
    rule = dict(parent["selected_rule"])
    rule["training_score_quantile"] = 0.970
    rule["final_training_threshold"] = LOWER
    payload = {
        "schema_version": "revealnav-mf3zf-train-collection-gate/1",
        "status": "TRAIN_RETURN_COLLECTION_AUTHORIZED",
        "task_metric_run_authorized": False,
        "collection_split": "train", "expected_episodes": 217,
        "expected_scenes": 58, "selected_architecture": parent["selected_architecture"],
        "selected_rule": rule, "checkpoints": parent["checkpoints"],
        "parent_mf3v_gate_sha256": sha256_file(SOURCE),
        "design_sha256": sha256_file(DESIGN),
        "unseen_or_test_read": False,
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZF collection gate drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({"status": payload["status"], "rule": rule}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
