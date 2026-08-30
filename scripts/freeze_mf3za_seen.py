#!/usr/bin/env python3
"""Freeze the passing MF3ZA val_seen controller before unseen evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf3za_consensus_band_freeze_v1/MF3ZA_VAL_SEEN_FREEZE.json"
GATE = ROOT / "artifacts/evaluation/mf3za_consensus_band_shadow_gate_v1/MF3ZA_SHADOW_GATE.json"
PROTOCOL = ROOT / "artifacts/evaluation/mf3za_consensus_band_rxr_val_seen_v1/MF3ZA_RXR_VAL_SEEN_PROTOCOL.json"
RESULT = ROOT / "artifacts/evaluation/mf3za_consensus_band_rxr_val_seen_v1/MF3ZA_RXR_VAL_SEEN_RESULT.json"
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZA_CONSENSUS_BAND.md"
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
        raise RuntimeError("MF3ZA shadow gate did not pass")
    if result.get("status") != "TASK_METRIC_GATE_PASS" or not all(result["gates"].values()):
        raise RuntimeError("MF3ZA val_seen gate did not pass")
    if protocol.get("selection_salt") != "revealnav-mf3v-rxr-val-seen-all-scenes/1":
        raise RuntimeError("MF3ZA val_seen pairing is not aligned to MF3V")
    checkpoints = []
    for row in gate["checkpoints"]:
        path = ROOT / row["path"]
        current = evidence(path)
        if current["sha256"] != row["sha256"] or current["bytes"] != row["bytes"]:
            raise RuntimeError("MF3V checkpoint drift before MF3ZA freeze")
        checkpoints.append(current)
    aggregate = result["aggregate_uad_ensemble_minus_baseline"]
    payload = {
        "schema_version": "revealnav-mf3za-val-seen-freeze/1",
        "status": "MF3ZA_VAL_SEEN_FROZEN",
        "frozen_at": "2026-08-29",
        "scope": "MF3ZA consensus-band tail recovery on frozen MF3V, RxR English val_seen",
        "deployment_boundary": {
            "allowed_switch": "native action to frozen runner-up only",
            "one_intervention_per_episode": True,
            "online_future_frames": False,
            "online_teacher_indices": False,
            "val_unseen_authorized": True,
            "threshold_tuned_on_val_unseen": False,
        },
        "selected_rule": gate["selected_rule"],
        "success_summary": {
            "runs": 171,
            "failures": 0,
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
            "source_files": [
                evidence(WORKER), evidence(RUNNER), evidence(DESIGN),
                evidence(ROOT / "scripts/evaluate_rxr_uad_adaptive_tail_mf3z.py"),
                evidence(ROOT / "scripts/promote_mf3za_shadow_gate.py"),
            ],
        },
        "next_gate": "independent non-overlapping RxR val_unseen evaluation with no threshold tuning",
    }
    if OUT.exists() and json.loads(OUT.read_text()) != payload:
        raise RuntimeError("MF3ZA freeze drift")
    if not OUT.exists():
        atomic_json(OUT, payload)
    print(json.dumps({"status": payload["status"], "success_summary": payload["success_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
