#!/usr/bin/env python3
"""Freeze MF3ZB only after every predeclared RxR val_seen gate passes."""

from __future__ import annotations

import json
from pathlib import Path

from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf3zb_temporal_maturity_freeze_v1/MF3ZB_VAL_SEEN_FREEZE.json"
GATE = ROOT / "artifacts/evaluation/mf3zb_temporal_maturity_shadow_gate_v1/MF3ZB_SHADOW_GATE.json"
PROTOCOL = ROOT / "artifacts/evaluation/mf3zb_temporal_maturity_rxr_val_seen_v1/MF3ZB_RXR_VAL_SEEN_PROTOCOL.json"
RESULT = ROOT / "artifacts/evaluation/mf3zb_temporal_maturity_rxr_val_seen_v1/MF3ZB_RXR_VAL_SEEN_RESULT.json"
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZB_TEMPORAL_MATURITY.md"
WORKER = ROOT / "scripts/rxr_uad_controller_worker_mf3.py"
RUNNER = ROOT / "scripts/run_rxr_uad_paired_metrics_mf3.py"


def evidence(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    gate = json.loads(GATE.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    if gate.get("status") != "SHADOW_GATE_PASS":
        raise RuntimeError("MF3ZB shadow gate did not pass")
    if result.get("status") != "TASK_METRIC_GATE_PASS":
        raise RuntimeError("MF3ZB val_seen gate did not pass")
    if not result.get("gates") or not all(result["gates"].values()):
        raise RuntimeError("not every MF3ZB val_seen gate passed")
    if protocol.get("selection_salt") != "revealnav-mf3v-rxr-val-seen-all-scenes/1":
        raise RuntimeError("MF3ZB val_seen pairing is not aligned to MF3V")
    checkpoints = []
    for row in gate["checkpoints"]:
        current = evidence(ROOT / row["path"])
        if current["sha256"] != row["sha256"] or current["bytes"] != row["bytes"]:
            raise RuntimeError("MF3V checkpoint drift before MF3ZB freeze")
        checkpoints.append(current)
    aggregate = result["aggregate_uad_ensemble_minus_baseline"]
    payload = {
        "schema_version": "revealnav-mf3zb-val-seen-freeze/1",
        "status": "MF3ZB_VAL_SEEN_FROZEN",
        "frozen_at": "2026-08-29",
        "scope": "MF3ZB temporal-maturity UAD on frozen MF3V, RxR English val_seen",
        "deployment_boundary": {
            "allowed_switch": "native action to frozen runner-up only",
            "one_intervention_per_episode": True,
            "minimum_decision_step": gate["selected_rule"]["minimum_decision_step"],
            "online_future_frames": False,
            "online_teacher_indices": False,
            "val_unseen_authorized": True,
            "threshold_tuned_on_val_unseen": False,
            "prior_unseen_used_for_failure_analysis": True,
            "next_unseen_must_exclude_all_previously_consumed_episodes": True,
        },
        "selected_rule": gate["selected_rule"],
        "success_summary": {
            "runs": result["runs"],
            "failures": result["failures"],
            "success_delta": aggregate["success"]["mean"],
            "spl_delta": aggregate["spl"]["mean"],
            "ndtw_delta": aggregate["ndtw"]["mean"],
            "utility_delta": aggregate["utility"]["mean"],
            "utility_bootstrap_95pct": aggregate["utility"]["scene_bootstrap_95pct"],
            "all_task_metric_gates_pass": all(result["gates"].values()),
        },
        "evidence": {
            "final_checkpoints": checkpoints,
            "shadow_gate": evidence(GATE),
            "val_seen_protocol": evidence(PROTOCOL),
            "val_seen_result": evidence(RESULT),
            "source_files": [evidence(WORKER), evidence(RUNNER), evidence(DESIGN)],
        },
        "next_gate": "fresh RxR val_unseen holdout excluding every previously evaluated unseen episode",
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZB freeze drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({"status": payload["status"], "success_summary": payload["success_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
