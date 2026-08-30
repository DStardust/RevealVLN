#!/usr/bin/env python3
"""Freeze the MF3ZG public-unseen engineering result only after every gate passes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


SEEN_FREEZE = ROOT / (
    "artifacts/evaluation/mf3zg_core_preserving_hierarchy_freeze_v1/"
    "MF3ZG_VAL_SEEN_FREEZE.json"
)
UNSEEN_ROOT = ROOT / "artifacts/evaluation/mf3zg_uad_rxr_val_unseen_holdout_v1"
PROTOCOL = UNSEEN_ROOT / "MF3ZG_RXR_VAL_UNSEEN_PROTOCOL.json"
RESULT = UNSEEN_ROOT / "MF3ZG_RXR_VAL_UNSEEN_RESULT.json"
OUT = UNSEEN_ROOT / "MF3ZG_RXR_VAL_UNSEEN_FREEZE.json"


def evidence(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    seen = json.loads(SEEN_FREEZE.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    if not (
        seen.get("status") == "MF3ZG_VAL_SEEN_FROZEN"
        and protocol.get("status") == "SEALED_BEFORE_MF3ZG_FRESH_HOLDOUT"
        and protocol.get("current_holdout_used_for_tuning") is False
        and protocol.get("counts", {}).get("overlap_with_all_prior") == 0
        and result.get("status") == "FRESH_HOLDOUT_ADVANTAGE_PASS"
        and result.get("current_holdout_used_for_tuning") is False
        and result.get("action_changes", 0) >= 5
        and all(result["gates"].values())
        and result.get("test_or_test_challenge_accessed") is False
    ):
        raise RuntimeError("MF3ZG unseen freeze precondition failure")
    aggregate = result["aggregate_ensemble_minus_baseline"]
    payload = {
        "schema_version": "revealnav-mf3zg-rxr-val-unseen-freeze/1",
        "status": "MF3ZG_RXR_VAL_UNSEEN_FROZEN",
        "frozen_at": "2026-08-30",
        "scope": result["scope"],
        "result_summary": {
            "episodes": protocol["counts"]["selected_episodes"],
            "runs": result["runs"],
            "action_changes": result["action_changes"],
            "success_delta": aggregate["success"]["mean"],
            "spl_delta": aggregate["spl"]["mean"],
            "ndtw_delta": aggregate["ndtw"]["mean"],
            "utility_delta": aggregate["utility"]["mean"],
            "utility_bootstrap_95pct": aggregate["utility"][
                "scene_cluster_bootstrap_95pct"
            ],
            "all_gates_pass": True,
        },
        "boundaries": {
            "threshold_tuned_on_current_holdout": False,
            "public_val_unseen_is_pristine": False,
            "test_or_test_challenge_accessed": False,
            "paper_result": True,
        },
        "evidence": {
            "seen_freeze": evidence(SEEN_FREEZE),
            "unseen_protocol": evidence(PROTOCOL),
            "unseen_result": evidence(RESULT),
        },
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZG unseen freeze drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "result_summary": payload["result_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
