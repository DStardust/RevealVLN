#!/usr/bin/env python3
"""Freeze MF3ZI only after its fixed val_seen task-metric gate passes."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEN_ROOT = ROOT / "artifacts/evaluation/mf3zi_causal_uncertainty_arbitration_rxr_val_seen_v1"
PROTOCOL = SEEN_ROOT / "MF3ZI_RXR_VAL_SEEN_PROTOCOL.json"
RESULT = SEEN_ROOT / "MF3ZI_RXR_VAL_SEEN_RESULT.json"
OUT = ROOT / "artifacts/evaluation/mf3zi_causal_uncertainty_arbitration_freeze_v1/MF3ZI_VAL_SEEN_FREEZE.json"


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    if not (
        protocol.get("status") == "SEALED_BEFORE_RXR_VAL_SEEN_TASK_METRICS"
        and protocol.get("public_unseen_authorized") is False
        and result.get("status") == "TASK_METRIC_GATE_PASS"
        and all(result.get("gates", {}).values())
    ):
        raise RuntimeError("MF3ZI val_seen freeze precondition failure")
    aggregate = result["aggregate_ensemble_minus_baseline"]
    payload = {
        "schema_version": "revealnav-mf3zi-val-seen-freeze/1",
        "status": "MF3ZI_VAL_SEEN_FROZEN",
        "frozen_at": "2026-08-30",
        "deployment_boundary": {"val_seen_authorized": True, "val_unseen_authorized": True, "test_or_test_challenge_authorized": False},
        "result_summary": {
            "episodes": len(protocol["selection"]),
            "action_changes": sum(
                json.loads((SEEN_ROOT / "full" / "runs" / f"ensemble_ep_{row['episode_id']}" / "RUN_SUMMARY.json").read_text())["controller"]["actions_changed"]
                for row in protocol["selection"]
            ),
            "success_delta": aggregate["success"]["mean"],
            "spl_delta": aggregate["spl"]["mean"],
            "ndtw_delta": aggregate["ndtw"]["mean"],
            "utility_delta": aggregate["utility"]["mean"],
            "utility_bootstrap_95pct": aggregate["utility"]["scene_bootstrap_95pct"],
            "all_gates_pass": True,
        },
        "boundaries": {"threshold_tuned_on_val_seen": False, "public_val_unseen_is_pristine": False, "test_or_test_challenge_accessed": False},
        "evidence": {"protocol": evidence(PROTOCOL), "result": evidence(RESULT)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZI freeze drift")
    if not OUT.exists():
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "result_summary": payload["result_summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
