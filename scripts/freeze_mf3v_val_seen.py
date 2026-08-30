#!/usr/bin/env python3
"""Create and verify the immutable MF3V val_seen handoff record."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf3v_horizon_freeze_v1"
LOCK = OUT / "MF3V_VAL_SEEN_FREEZE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"freeze input is not a regular project-local file: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build() -> dict:
    gate_path = ROOT / "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/MF3V_SHADOW_GATE.json"
    result_path = ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1/MF3V_RXR_VAL_SEEN_RESULT.json"
    protocol_path = ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1/MF3V_RXR_VAL_SEEN_PROTOCOL.json"
    gate = json.loads(gate_path.read_text())
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if gate.get("status") != "SHADOW_GATE_PASS" or gate.get("task_metric_run_authorized") is not True:
        raise RuntimeError("MF3V shadow gate is not passed")
    if result.get("status") != "TASK_METRIC_GATE_PASS" or not all(result.get("gates", {}).values()):
        raise RuntimeError("MF3V val_seen task-metric gate is not passed")
    if result.get("public_unseen_authorized") is not False:
        raise RuntimeError("unseen authorization boundary drift")
    if protocol.get("public_unseen_authorized") is not False:
        raise RuntimeError("sealed protocol unseen boundary drift")
    rule = gate["selected_rule"]
    checkpoint_paths = [
        f"artifacts/training/mf3v_horizon_ranker_v1/fold_final/seed_{seed}/horizon_ranker_mf3v.pt"
        for seed in (20260826, 20260827, 20260828)
    ]
    source_paths = [
        "scripts/rxr_uad_controller_worker_mf3.py",
        "scripts/run_rxr_uad_paired_metrics_mf3.py",
        "scripts/train_rxr_uad_horizon_mf3v.py",
        "scripts/evaluate_rxr_uad_horizon_mf3v.py",
        "revealnav_mf3/uad.py",
        "revealnav_mf3/__init__.py",
        "artifacts/design/METHOD_FREEZE_3V_HORIZON_CONSISTENT_UAD.md",
        "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/MF3V_SHADOW_GATE.json",
        "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1/MF3V_RXR_VAL_SEEN_PROTOCOL.json",
        "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1/MF3V_RXR_VAL_SEEN_RESULT.json",
    ]
    return {
        "schema_version": "revealnav-mf3v-val-seen-freeze/1",
        "status": "MF3V_VAL_SEEN_FROZEN",
        "frozen_at": "2026-08-29",
        "scope": "MF3V horizon-consistent UAD, RxR English val_seen task metrics",
        "deployment_boundary": {
            "online_future_frames": False,
            "online_teacher_indices": False,
            "allowed_switch": "native action to frozen runner-up only",
            "val_unseen_authorized": False,
        },
        "training_protocol": {
            "horizon": int(rule["horizon"]),
            "hidden_dim": int(rule["hidden_dim"]),
            "seeds": [20260826, 20260827, 20260828],
            "fold": "final",
            "optimizer_steps": 800,
            "training_score_quantile": float(rule["training_score_quantile"]),
            "mad_weight": float(rule["mad_weight"]),
            "policy_risk_beta": float(rule["policy_risk_beta"]),
            "final_training_threshold": float(rule["final_training_threshold"]),
            "score_upper_threshold": float(rule["score_upper_threshold"]),
            "persistence_steps": int(rule["persistence_steps"]),
        },
        "evidence": {
            "shadow_gate": file_record(str(gate_path.relative_to(ROOT))),
            "val_seen_protocol": file_record(str(protocol_path.relative_to(ROOT))),
            "val_seen_result": file_record(str(result_path.relative_to(ROOT))),
            "source_files": [file_record(path) for path in source_paths],
            "final_checkpoints": [file_record(path) for path in checkpoint_paths],
        },
        "success_summary": {
            "runs": 171,
            "failures": 0,
            "utility_delta": float(result["aggregate_uad_ensemble_minus_baseline"]["utility"]["mean"]),
            "utility_bootstrap_95pct": result["aggregate_uad_ensemble_minus_baseline"]["utility"]["scene_bootstrap_95pct"],
            "ndtw_delta": float(result["aggregate_uad_ensemble_minus_baseline"]["ndtw"]["mean"]),
            "spl_delta": float(result["aggregate_uad_ensemble_minus_baseline"]["spl"]["mean"]),
            "success_delta": float(result["aggregate_uad_ensemble_minus_baseline"]["success"]["mean"]),
        },
        "next_gate": "MF3V RxR val_unseen preflight with no threshold tuning",
    }


def main() -> int:
    value = build()
    OUT.mkdir(parents=True, exist_ok=True)
    part = LOCK.with_name(LOCK.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, LOCK)
    reread = json.loads(LOCK.read_text())
    if reread != value:
        raise RuntimeError("freeze record round-trip mismatch")
    print(json.dumps({
        "status": value["status"],
        "path": str(LOCK.relative_to(ROOT)),
        "source_count": len(value["evidence"]["source_files"]),
        "checkpoint_count": len(value["evidence"]["final_checkpoints"]),
        "next_gate": value["next_gate"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
