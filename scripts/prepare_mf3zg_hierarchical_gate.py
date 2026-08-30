#!/usr/bin/env python3
"""Build the train-only MF3ZG hierarchical authorization gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


CORE_GATE = ROOT / (
    "artifacts/training/mf3ze_action_aligned_return_gate_v1/"
    "MF3ZE_CROSSFIT_GATE.json"
)
MF3V_GATE = ROOT / (
    "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/MF3V_SHADOW_GATE.json"
)
EXPANSION_GATE = ROOT / (
    "artifacts/training/mf3zf_action_aligned_return_gate_v1/"
    "MF3ZF_CROSSFIT_GATE.json"
)
COLLECTION_GATE = ROOT / (
    "artifacts/training/mf3zf_expanded_collection_v1/"
    "MF3ZF_COLLECTION_GATE.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZG_CORE_PRESERVING_HIERARCHY.md"
OUT = ROOT / (
    "artifacts/training/mf3zg_hierarchical_core_preserving_gate_v1/"
    "MF3ZG_SHADOW_GATE.json"
)
RETURN_THRESHOLD = 0.0


def evidence(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    core = json.loads(CORE_GATE.read_text())
    mf3v = json.loads(MF3V_GATE.read_text())
    expansion = json.loads(EXPANSION_GATE.read_text())
    collection = json.loads(COLLECTION_GATE.read_text())
    if not (
        core.get("status") == "SHADOW_GATE_PASS"
        and mf3v.get("status") == "SHADOW_GATE_PASS"
        and expansion.get("status") == "SHADOW_GATE_PASS"
        and collection.get("status") == "TRAIN_RETURN_COLLECTION_AUTHORIZED"
        and core.get("controls", {}).get("unseen_or_test_read") is False
        and expansion.get("controls", {}).get("unseen_or_test_read") is False
        and expansion.get("controls", {}).get("rows") == 217
    ):
        raise RuntimeError("MF3ZG parent gate drift")
    harm_threshold = float(
        expansion["selected_rule"]["harm_probability_threshold"]
    )
    core_rule = mf3v["selected_rule"]
    expansion_rule = collection["selected_rule"]
    if not (
        float(core_rule["score_upper_threshold"])
        == float(expansion_rule["score_upper_threshold"])
        and float(expansion_rule["final_training_threshold"])
        < float(core_rule["final_training_threshold"])
    ):
        raise RuntimeError("MF3ZG proposal hierarchy drift")
    selected = [
        row for row in expansion["oof_rows"]
        if row["robust_expected_utility"] >= RETURN_THRESHOLD
        and row["upper_harm_probability"] <= harm_threshold
    ]
    targets = [float(row["target_utility"]) for row in selected]
    scenes = sorted({str(row["scene_id"]) for row in selected})
    leave_one_scene_out = [
        sum(
            float(row["target_utility"])
            for row in selected if str(row["scene_id"]) != scene
        )
        for scene in scenes
    ]
    catastrophic = sum(value <= -0.10 for value in targets)
    ungated_catastrophic = int(
        expansion["ungated_mf3v_oof_cohort"]["catastrophic"]
    )
    rule_evidence = {
        "authorized": len(selected),
        "positive": sum(value > 1e-8 for value in targets),
        "negative": sum(value < -1e-8 for value in targets),
        "ties": sum(abs(value) <= 1e-8 for value in targets),
        "catastrophic": catastrophic,
        "total_utility": sum(targets),
        "deployed_mean_utility": sum(targets) / 217,
        "minimum_leave_one_selected_scene_out_total": min(leave_one_scene_out),
    }
    passed = (
        rule_evidence["authorized"] >= 24
        and rule_evidence["total_utility"] > 0.0
        and catastrophic < ungated_catastrophic
        and rule_evidence["minimum_leave_one_selected_scene_out_total"] > 0.0
    )
    payload = {
        "schema_version": "revealnav-mf3zg-hierarchical-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "task_metric_run_authorized": passed,
        "selected_architecture": collection["selected_architecture"],
        "selected_rule": collection["selected_rule"],
        "hierarchy": {
            "core_score_threshold": float(core_rule["final_training_threshold"]),
            "expansion_score_threshold": float(
                expansion_rule["final_training_threshold"]
            ),
            "score_upper_threshold": float(core_rule["score_upper_threshold"]),
            "expansion_return_threshold": RETURN_THRESHOLD,
            "expansion_harm_probability_threshold": harm_threshold,
            "rejected_expansion_consumes_core_opportunity": False,
            "maximum_executed_switches_per_episode": 1,
        },
        "expansion_oof_evidence": rule_evidence,
        "controls": {
            "selection_source": "scene-disjoint RxR-train OOF predictions only",
            "unseen_or_test_read": False,
        },
        "sources": {
            "core_gate": evidence(CORE_GATE),
            "mf3v_proposal_gate": evidence(MF3V_GATE),
            "expansion_gate": evidence(EXPANSION_GATE),
            "collection_gate": evidence(COLLECTION_GATE),
            "design": evidence(DESIGN),
        },
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZG gate drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({"status": payload["status"], **rule_evidence}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
