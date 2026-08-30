#!/usr/bin/env python3
"""Freeze MF3ZG after its train-only and unchanged val_seen gates pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


OUT = ROOT / (
    "artifacts/evaluation/mf3zg_core_preserving_hierarchy_freeze_v1/"
    "MF3ZG_VAL_SEEN_FREEZE.json"
)
GATE = ROOT / (
    "artifacts/training/mf3zg_hierarchical_core_preserving_gate_v1/"
    "MF3ZG_SHADOW_GATE.json"
)
CORE_MODEL = ROOT / (
    "artifacts/training/mf3ze_action_aligned_return_gate_v1/"
    "MF3ZE_GATE_MODELS.npz"
)
EXPANSION_MODEL = ROOT / (
    "artifacts/training/mf3zf_action_aligned_return_gate_v1/"
    "MF3ZF_GATE_MODELS.npz"
)
SEEN_ROOT = ROOT / (
    "artifacts/evaluation/mf3zg_core_preserving_hierarchy_rxr_val_seen_v1"
)
PROTOCOL = SEEN_ROOT / "MF3ZG_RXR_VAL_SEEN_PROTOCOL.json"
RESULT = SEEN_ROOT / "MF3ZG_RXR_VAL_SEEN_RESULT.json"
PARENT = ROOT / (
    "artifacts/evaluation/mf3ze_action_aligned_freeze_v1/"
    "MF3ZE_VAL_SEEN_FREEZE.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZG_CORE_PRESERVING_HIERARCHY.md"
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
        and gate.get("hierarchy", {}).get(
            "rejected_expansion_consumes_core_opportunity"
        ) is False
        and result.get("status") == "TASK_METRIC_GATE_PASS"
        and all(result["gates"].values())
        and parent.get("status") == "MF3ZE_VAL_SEEN_FROZEN"
        and protocol.get("selection_salt")
        == "revealnav-mf3v-rxr-val-seen-all-scenes/1"
        and protocol.get("runs", {}).get("new_total") == 57
    ):
        raise RuntimeError("MF3ZG freeze precondition failure")
    aggregate = result["aggregate_uad_ensemble_minus_baseline"]
    payload = {
        "schema_version": "revealnav-mf3zg-val-seen-freeze/1",
        "status": "MF3ZG_VAL_SEEN_FROZEN",
        "frozen_at": "2026-08-30",
        "scope": (
            "core-preserving hierarchical action-aligned return gating on "
            "frozen MF3V, RxR English val_seen"
        ),
        "deployment_boundary": {
            "allowed_switch": "native action to frozen runner-up only",
            "independent_core_and_expansion_proposal_budgets": True,
            "maximum_executed_switches_per_episode": 1,
            "online_future_frames": False,
            "online_teacher_indices": False,
            "online_oracle_metrics": False,
            "val_unseen_authorized": True,
            "val_unseen_holdout_authorized": True,
            "prior_unseen_used_for_failure_analysis": True,
            "next_unseen_must_exclude_all_previously_consumed_episodes": True,
        },
        "hierarchy": gate["hierarchy"],
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
            "parent_mf3ze_freeze": evidence(PARENT),
            "hierarchical_gate": evidence(GATE),
            "core_model": evidence(CORE_MODEL),
            "expansion_model": evidence(EXPANSION_MODEL),
            "val_seen_protocol": evidence(PROTOCOL),
            "val_seen_result": evidence(RESULT),
            "source_files": [evidence(WORKER), evidence(RUNNER), evidence(DESIGN)],
        },
        "next_gate": (
            "pre-sealed 400-episode fresh RxR val_unseen power holdout, "
            "excluding every previously consumed episode"
        ),
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZG freeze drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "success_summary": payload["success_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
