#!/usr/bin/env python3
"""Freeze MF3ZF only after its train-only and unchanged val_seen gates pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


OUT = ROOT / (
    "artifacts/evaluation/mf3zf_coverage_safety_freeze_v1/"
    "MF3ZF_VAL_SEEN_FREEZE.json"
)
GATE = ROOT / (
    "artifacts/training/mf3zf_action_aligned_return_gate_v1/"
    "MF3ZF_CROSSFIT_GATE.json"
)
MODEL = GATE.with_name("MF3ZF_GATE_MODELS.npz")
SEEN_ROOT = ROOT / "artifacts/evaluation/mf3zf_coverage_safety_rxr_val_seen_v1"
PROTOCOL = SEEN_ROOT / "MF3ZF_RXR_VAL_SEEN_PROTOCOL.json"
RESULT = SEEN_ROOT / "MF3ZF_RXR_VAL_SEEN_RESULT.json"
PARENT = ROOT / (
    "artifacts/evaluation/mf3v_horizon_freeze_v1/MF3V_VAL_SEEN_FREEZE.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZF_COVERAGE_SAFETY.md"
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
    parent = json.loads(PARENT.read_text())
    if not (
        gate.get("status") == "SHADOW_GATE_PASS"
        and gate.get("task_metric_run_authorized") is True
        and gate.get("controls", {}).get("unseen_or_test_read") is False
        and gate.get("controls", {}).get("rows") == 217
        and gate.get("controls", {}).get("scenes") == 58
        and result.get("status") == "TASK_METRIC_GATE_PASS"
        and all(result["gates"].values())
        and parent.get("status") == "MF3V_VAL_SEEN_FROZEN"
        and protocol.get("selection_salt")
        == "revealnav-mf3v-rxr-val-seen-all-scenes/1"
        and protocol.get("runs", {}).get("new_total") == 57
    ):
        raise RuntimeError("MF3ZF freeze precondition failure")
    if gate["model"] != evidence(MODEL) | {"members": gate["model"]["members"]}:
        raise RuntimeError("MF3ZF safety model drift")
    aggregate = result["aggregate_uad_ensemble_minus_baseline"]
    payload = {
        "schema_version": "revealnav-mf3zf-val-seen-freeze/1",
        "status": "MF3ZF_VAL_SEEN_FROZEN",
        "frozen_at": "2026-08-30",
        "scope": (
            "coverage-expanded MF3V proposals with a train-only action-aligned "
            "counterfactual-return safety gate, RxR English val_seen"
        ),
        "deployment_boundary": {
            "allowed_switch": "native action to frozen runner-up only",
            "one_proposal_evaluated_per_episode": True,
            "one_intervention_per_episode": True,
            "online_future_frames": False,
            "online_teacher_indices": False,
            "online_oracle_metrics": False,
            "val_unseen_authorized": True,
            "val_unseen_holdout_authorized": True,
            "prior_unseen_used_for_failure_analysis": True,
            "next_unseen_must_exclude_all_previously_consumed_episodes": True,
        },
        "selected_rule": gate["selected_rule"],
        "success_summary": {
            "new_treatment_runs": protocol["runs"]["new_total"],
            "reused_frozen_control_runs": 114,
            "failures": 0,
            "success_delta": aggregate["success"]["mean"],
            "spl_delta": aggregate["spl"]["mean"],
            "ndtw_delta": aggregate["ndtw"]["mean"],
            "utility_delta": aggregate["utility"]["mean"],
            "utility_bootstrap_95pct": aggregate["utility"][
                "scene_bootstrap_95pct"
            ],
            "all_task_metric_gates_pass": True,
        },
        "evidence": {
            "parent_mf3v_freeze": evidence(PARENT),
            "action_aligned_gate": evidence(GATE),
            "safety_model": evidence(MODEL),
            "val_seen_protocol": evidence(PROTOCOL),
            "val_seen_result": evidence(RESULT),
            "source_files": [evidence(WORKER), evidence(RUNNER), evidence(DESIGN)],
        },
        "next_gate": (
            "pre-sealed 200-episode fresh RxR val_unseen power holdout, "
            "excluding every previously consumed episode"
        ),
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZF freeze drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "success_summary": payload["success_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
