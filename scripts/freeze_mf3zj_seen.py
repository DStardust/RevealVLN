#!/usr/bin/env python3
"""Freeze MF3ZJ only after every locked RxR val-seen gate passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEN = ROOT / (
    "artifacts/evaluation/"
    "mf3zj_counterfactual_transfer_arbitration_rxr_val_seen_v1"
)
PROTOCOL = SEEN / "MF3ZJ_RXR_VAL_SEEN_PROTOCOL.json"
RESULT = SEEN / "MF3ZJ_RXR_VAL_SEEN_RESULT.json"
GATE = ROOT / (
    "artifacts/training/mf3zj_counterfactual_transfer_gate_v1/"
    "MF3ZJ_CROSSFIT_GATE.json"
)
MODEL = ROOT / (
    "artifacts/training/mf3zj_counterfactual_transfer_gate_v1/"
    "MF3ZJ_TRANSFER_GATE_MODELS.npz"
)
DESIGN = ROOT / (
    "artifacts/design/METHOD_FREEZE_3ZJ_COUNTERFACTUAL_TRANSFER_ARBITRATION.md"
)
CONTROLLER = ROOT / "scripts/rxr_uad_mf3zj_controller.py"
WORKER = ROOT / "scripts/rxr_uad_mf3zj_worker.py"
OUT = ROOT / (
    "artifacts/evaluation/mf3zj_counterfactual_transfer_arbitration_freeze_v1/"
    "MF3ZJ_VAL_SEEN_FREEZE.json"
)


def evidence(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    gate = json.loads(GATE.read_text())
    if not (
        protocol.get("status") == "SEALED_BEFORE_RXR_VAL_SEEN_TASK_METRICS"
        and protocol.get("public_unseen_authorized") is False
        and result.get("status") == "TASK_METRIC_GATE_PASS"
        and all(result.get("gates", {}).values())
        and gate.get("status") == "SHADOW_GATE_PASS"
        and gate.get("task_metric_run_authorized") is True
    ):
        raise RuntimeError("MF3ZJ val-seen freeze precondition failure")
    aggregate = result["aggregate_ensemble_minus_baseline"]
    comparison = result["aggregate_ensemble_minus_uncertainty"]
    payload = {
        "schema_version": "revealnav-mf3zj-val-seen-freeze/1",
        "status": "MF3ZJ_VAL_SEEN_FROZEN",
        "frozen_at": "2026-08-30",
        "deployment_boundary": {
            "val_seen_authorized": True,
            "val_unseen_authorized": True,
            "test_or_test_challenge_authorized": False,
        },
        "result_summary": {
            "episodes": len(protocol["selection"]),
            "success_delta": aggregate["success"]["mean"],
            "spl_delta": aggregate["spl"]["mean"],
            "ndtw_delta": aggregate["ndtw"]["mean"],
            "utility_delta": aggregate["utility"]["mean"],
            "utility_bootstrap_95pct": aggregate["utility"][
                "scene_bootstrap_95pct"
            ],
            "utility_minus_uncertainty": comparison["utility"]["mean"],
            "all_gates_pass": True,
        },
        "boundaries": {
            "threshold_tuned_on_val_seen": False,
            "current_val_unseen_used_for_tuning": False,
            "test_or_test_challenge_accessed": False,
            "maximum_executed_switches_per_episode": 1,
        },
        "evidence": {
            "protocol": evidence(PROTOCOL),
            "result": evidence(RESULT),
            "gate": evidence(GATE),
            "model": evidence(MODEL),
            "design": evidence(DESIGN),
            "controller": evidence(CONTROLLER),
            "worker": evidence(WORKER),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZJ freeze drift")
    if not OUT.exists():
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "result_summary": payload["result_summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
