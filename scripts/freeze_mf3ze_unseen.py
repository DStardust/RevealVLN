#!/usr/bin/env python3
"""Freeze the MF3ZE unseen result without upgrading it to a pristine test claim."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


EVALUATION = ROOT / "artifacts/evaluation/mf3ze_uad_rxr_val_unseen_holdout_v1"
PROTOCOL = EVALUATION / "MF3ZE_RXR_VAL_UNSEEN_PROTOCOL.json"
RESULT = EVALUATION / "MF3ZE_RXR_VAL_UNSEEN_RESULT.json"
SEEN_FREEZE = ROOT / (
    "artifacts/evaluation/mf3ze_action_aligned_freeze_v1/"
    "MF3ZE_VAL_SEEN_FREEZE.json"
)
OUT = ROOT / (
    "artifacts/evaluation/mf3ze_unseen_freeze_v1/"
    "MF3ZE_RXR_VAL_UNSEEN_FREEZE.json"
)


def evidence(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    seen = json.loads(SEEN_FREEZE.read_text())
    if not (
        seen.get("status") == "MF3ZE_VAL_SEEN_FROZEN"
        and protocol.get("status") == "SEALED_BEFORE_MF3ZE_FRESH_HOLDOUT"
        and protocol.get("counts", {}).get("selected_episodes") == 100
        and protocol.get("counts", {}).get("overlap_with_all_prior") == 0
        and result.get("status") == "FRESH_HOLDOUT_ADVANTAGE_PASS"
        and result.get("paper_result") is True
        and all(result["gates"].values())
        and result.get("current_holdout_used_for_tuning") is False
        and result.get("test_or_test_challenge_accessed") is False
    ):
        raise RuntimeError("MF3ZE unseen freeze precondition failure")
    aggregate = result["aggregate_ensemble_minus_baseline"]
    payload = {
        "schema_version": "revealnav-mf3ze-rxr-val-unseen-freeze/1",
        "status": "MF3ZE_RXR_VAL_UNSEEN_ADVANTAGE_FROZEN",
        "frozen_at": "2026-08-30",
        "claim_boundary": {
            "valid": "fresh episode-disjoint development holdout evidence",
            "not_valid": "pristine public val_unseen or hidden-test claim",
            "reason": (
                "earlier revisions consumed other public val_unseen episodes for "
                "failure analysis; this cohort itself was never used for tuning"
            ),
        },
        "summary": {
            "episodes": 100, "runs": result["runs"],
            "action_changes": result["action_changes"],
            "success_delta": aggregate["success"]["mean"],
            "spl_delta": aggregate["spl"]["mean"],
            "ndtw_delta": aggregate["ndtw"]["mean"],
            "utility_delta": aggregate["utility"]["mean"],
            "utility_scene_cluster_bootstrap_95pct": aggregate["utility"][
                "scene_cluster_bootstrap_95pct"
            ],
            "all_gates_pass": True,
        },
        "evidence": {
            "seen_freeze": evidence(SEEN_FREEZE),
            "protocol": evidence(PROTOCOL), "result": evidence(RESULT),
        },
        "next_submission_gate": (
            "report this as public-unseen development evidence and run the "
            "untouched official test/challenge server only after the method is final"
        ),
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZE unseen freeze drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
