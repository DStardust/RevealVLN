#!/usr/bin/env python3
"""Seal the parameter-free MF3ZH composition gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


HIERARCHY_GATE = ROOT / (
    "artifacts/training/mf3zg_hierarchical_core_preserving_gate_v1/"
    "MF3ZG_SHADOW_GATE.json"
)
MF3V_GATE = ROOT / (
    "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/MF3V_SHADOW_GATE.json"
)
DIAGNOSTIC = ROOT / (
    "artifacts/evaluation/mf3zg_uad_rxr_val_unseen_holdout_v1/"
    "MF3ZG_RXR_VAL_UNSEEN_RESULT.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZH_UNCERTAINTY_FLOOR_RESIDUAL.md"
OUT = ROOT / (
    "artifacts/training/mf3zh_uncertainty_floor_residual_gate_v1/"
    "MF3ZH_SHADOW_GATE.json"
)


def evidence(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    hierarchy = json.loads(HIERARCHY_GATE.read_text())
    mf3v = json.loads(MF3V_GATE.read_text())
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    if not (
        hierarchy.get("status") == "SHADOW_GATE_PASS"
        and hierarchy.get("task_metric_run_authorized") is True
        and mf3v.get("status") == "SHADOW_GATE_PASS"
        and mf3v.get("task_metric_run_authorized") is True
        and diagnostic.get("status") == "FRESH_HOLDOUT_ADVANTAGE_FAIL"
        and diagnostic.get("gates", {}).get("utility_lower_95_positive") is True
        and diagnostic.get("gates", {}).get(
            "learned_utility_exceeds_uncertainty"
        ) is False
        and diagnostic.get("current_holdout_used_for_tuning") is False
    ):
        raise RuntimeError("MF3ZH composition source drift")
    payload = {
        "schema_version": "revealnav-mf3zh-uncertainty-floor-gate/1",
        "status": "SHADOW_GATE_PASS",
        "task_metric_run_authorized": True,
        "selected_architecture": hierarchy["selected_architecture"],
        "selected_rule": hierarchy["selected_rule"],
        "hierarchy": hierarchy["hierarchy"],
        "exact_budget_control": mf3v["exact_budget_control"],
        "composition": {
            "learned_residual_priority": True,
            "uncertainty_actions_consume_learned_budget": False,
            "maximum_learned_switches_per_episode": 1,
            "uncertainty_floor_retains_original_multi_step_behavior": True,
        },
        "controls": {
            "numeric_parameters_added_or_refit": False,
            "prior_public_unseen_used_for_failure_analysis": True,
            "next_holdout_used_for_tuning": False,
            "test_or_test_challenge_read": False,
        },
        "sources": {
            "hierarchy_gate": evidence(HIERARCHY_GATE),
            "mf3v_gate": evidence(MF3V_GATE),
            "diagnostic_result": evidence(DIAGNOSTIC),
            "design": evidence(DESIGN),
        },
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZH gate drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "native_margin_max": payload["exact_budget_control"][
            "native_margin_max"
        ],
        "numeric_parameters_added_or_refit": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
