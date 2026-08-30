#!/usr/bin/env python3
"""Audit and fit the pre-sealed MF3ZK-DSR v1 train-development revision."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (SCRIPTS, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from revealnav_mf3.distributional_switch import ensemble_checkpoint  # noqa: E402
from revealnav_mf3.dsr_selection import (  # noqa: E402
    nested_distributional_fit,
    proposal_support_audit,
    risk_coverage_diagnostic,
    scene_cluster_bootstrap,
    stratified_equal_budget_baselines,
)
from train_mf3zk_joint_action_aligned_gate import (  # noqa: E402
    _vector,
    sha256_file,
)
from seal_mf3zk_dsr_protocol import (  # noqa: E402
    PROTOCOL,
    REVISION,
    OUT,
    atomic_json,
    verify_protocol,
)


AUDIT = OUT / "MF3ZK_DSR_PROPOSAL_SUPPORT_AUDIT.json"
RESULT = OUT / "MF3ZK_DSR_TRAIN_DEVELOPMENT_RESULT.json"
GATE = OUT / "MF3ZK_DSR_GATE.pt"
SCHEMA = "revealnav-mf3zk-dsr-train-development-result/1"
BOOTSTRAP_REPLICATES = 10_000


def _identity(row: dict) -> tuple[str, str, int]:
    return (
        str(row["dataset"]), str(row["episode_id"]),
        int(row["decision"].get("step", row.get("decision_step", -1))),
    )


def _arrays(
    protocol: dict, rows: list[dict], indices: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if indices is None:
        selected = rows
    else:
        selected = [rows[int(index)] for index in indices]
    matrix = np.stack([_vector(row) for row in selected]).astype(np.float64)
    target = np.asarray([float(row["target"]) for row in selected], dtype=np.float64)
    scenes = np.asarray([str(row["scene_id"]) for row in selected])
    datasets = np.asarray([str(row["dataset"]) for row in selected])
    scene_assignment = protocol["selection"]["outer_scene_assignment"]
    outer_folds = np.asarray([
        int(scene_assignment[scene]) for scene in scenes
    ], dtype=np.int64)
    canonical = {
        (
            value["identity"]["dataset"], value["identity"]["episode_id"],
            int(value["identity"]["decision_step"]),
        ): value
        for value in protocol["source_inventory"]["canonical_rows"]
    }
    for row, vector, fold in zip(selected, matrix, outer_folds, strict=True):
        item = canonical.get(_identity(row))
        if item is None or item["scene_id"] != str(row["scene_id"]):
            raise RuntimeError("DSR canonical row identity drift")
        if int(item["outer_fold"]) != int(fold):
            raise RuntimeError("DSR sealed outer-fold drift")
        digest = __import__("hashlib").sha256(
            np.asarray(vector, dtype="<f8").tobytes()
        ).hexdigest()
        if digest != item["causal_vector_sha256"]:
            raise RuntimeError("DSR causal vector hash drift")
    return selected, matrix, target, scenes, datasets, outer_folds


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _audit(protocol: dict, rows: list[dict]) -> dict:
    _, _, _, _, _, outer_folds = _arrays(protocol, rows)
    audit = proposal_support_audit(rows, outer_folds)
    return {
        "schema_version": "revealnav-mf3zk-dsr-proposal-support-audit/1",
        "revision": REVISION,
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "source_provenance_verified": True,
        "public_split_access": {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        },
        **audit,
    }


def _arm_config(protocol: dict) -> dict:
    selection = protocol["selection"]
    return {
        "outer_folds": int(selection["outer_folds"]),
        "inner_folds": int(selection["inner_folds"]),
        "weight_decay_grid": list(selection["weight_decay_grid"]),
        "seeds": list(protocol["model"]["ensemble_seeds"]),
        "learning_rate": float(selection["learning_rate"]),
        "training_steps": int(selection["training_steps"]),
        "inner_fold_salt": str(selection["inner_fold_salt"]),
    }


def _matched_baseline_failures(
    fit: dict, baselines: dict,
) -> list[str]:
    gate = fit["outer_oof"]
    low = baselines["fold_domain_matched"]["baselines"]["low_native_margin"]
    failures = []
    if gate["evidence"]["total_utility"] <= low["overall"]["total_utility"]:
        failures.append("utility_not_above_fold_domain_matched_low_margin")
    if gate["evidence"]["catastrophic_rate"] > low["overall"]["catastrophic_rate"]:
        failures.append("catastrophic_rate_above_fold_domain_matched_low_margin")
    for domain, evidence in gate["domains"].items():
        baseline = low["domains"][domain]
        if evidence["total_utility"] <= baseline["total_utility"]:
            failures.append(
                f"{domain}:utility_not_above_fold_domain_matched_low_margin"
            )
        if evidence["catastrophic_rate"] > baseline["catastrophic_rate"]:
            failures.append(
                f"{domain}:catastrophic_rate_above_fold_domain_matched_low_margin"
            )
    return failures


def _fit_arm(
    protocol: dict, rows: list[dict], arm: str,
) -> tuple[dict, list]:
    indices = np.asarray([
        index for index, row in enumerate(rows)
        if arm == "joint" or row["dataset"] == arm
    ], dtype=np.int64)
    selected, matrix, target, scenes, datasets, outer_folds = _arrays(
        protocol, rows, indices
    )
    fit = nested_distributional_fit(
        matrix, target, scenes, datasets, outer_folds, _arm_config(protocol)
    )
    final_models = fit.pop("final_models")
    if "outer_oof" not in fit:
        return {
            "arm": arm,
            "rows": len(selected),
            "scenes": len(set(scenes)),
            **_jsonable(fit),
            "status": "TRAIN_DEVELOPMENT_FAIL",
            "equal_budget": None,
            "risk_coverage": None,
            "scene_cluster_bootstrap": None,
        }, []

    gate_mask = fit["outer_oof"]["authorized_mask"]
    baselines = stratified_equal_budget_baselines(
        selected, target, gate_mask, outer_folds
    )
    baseline_masks = baselines.pop("internal_masks")
    matched_low_margin = baseline_masks["fold_domain_matched"][
        "low_native_margin"
    ]
    baseline_failures = _matched_baseline_failures(fit, baselines)
    failures = list(fit["failure_reasons"]) + baseline_failures
    bootstrap = scene_cluster_bootstrap(
        gate_mask, target, scenes, datasets,
        comparator_mask=matched_low_margin,
        replicates=BOOTSTRAP_REPLICATES,
    )
    curve = risk_coverage_diagnostic(
        fit["outer_oof"]["lower_q20"], target, scenes
    )
    oof_rows = []
    for index, row in enumerate(selected):
        oof_rows.append({
            "dataset": str(row["dataset"]),
            "scene_id": str(row["scene_id"]),
            "episode_id": str(row["episode_id"]),
            "decision_step": int(row["decision"]["step"]),
            "outer_fold": int(outer_folds[index]),
            "target_delta_utility": float(target[index]),
            "lower_q20_utility": float(fit["outer_oof"]["lower_q20"][index]),
            "median_q50_utility": float(fit["outer_oof"]["median_q50"][index]),
            "upper_q80_utility": float(fit["outer_oof"]["upper_q80"][index]),
            "authorized": bool(gate_mask[index]),
        })
    return {
        "arm": arm,
        "rows": len(selected),
        "scenes": len(set(scenes)),
        **_jsonable(fit),
        "status": "TRAIN_DEVELOPMENT_PASS" if not failures
        else "TRAIN_DEVELOPMENT_FAIL",
        "failure_reasons": failures,
        "equal_budget": _jsonable(baselines),
        "risk_coverage": _jsonable(curve),
        "scene_cluster_bootstrap": _jsonable(bootstrap),
        "outer_oof_rows": oof_rows,
    }, final_models if not failures else []


def _save_gate(models: list, protocol: dict) -> dict:
    if not models:
        raise RuntimeError("cannot save an unauthorized DSR gate")
    payload = ensemble_checkpoint(models, metadata={
        "revision": REVISION,
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "public_unseen_authorized": False,
    })
    GATE.parent.mkdir(parents=True, exist_ok=True)
    part = GATE.with_name(GATE.name + ".part")
    torch.save(payload, part)
    os.replace(part, GATE)
    return {
        "path": str(GATE.relative_to(ROOT)),
        "bytes": int(GATE.stat().st_size),
        "sha256": sha256_file(GATE),
        "members": len(models),
        "public_unseen_authorized": False,
    }


def _audit_failure_result(protocol: dict, rows: list[dict], audit: dict) -> dict:
    return {
        "schema_version": SCHEMA,
        "status": "TRAIN_DEVELOPMENT_FAIL",
        "revision": REVISION,
        "source_provenance_verified": True,
        "public_unseen_authorized": False,
        "old_confirmation_reused": False,
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "proposal_support_audit": audit,
        "outer_folds": [],
        "joint": None,
        "rxr_only": None,
        "r2r_only": None,
        "risk_coverage": {},
        "equal_budget": {"global": {}, "fold_domain_matched": {}},
        "ablations": {"status": "NOT_RUN_AFTER_SUPPORT_AUDIT_FAIL"},
        "failure_reasons": list(audit["failure_reasons"]),
        "public_split_access": protocol["public_split_access"],
        "gate_artifact": None,
    }


def run_audit(protocol_path: Path) -> int:
    protocol, rows, _ = verify_protocol(protocol_path)
    audit = _audit(protocol, rows)
    atomic_json(AUDIT, audit)
    if audit["status"] != "PROPOSAL_SUPPORT_AUDIT_PASS":
        atomic_json(RESULT, _audit_failure_result(protocol, rows, audit))
        print(json.dumps({"status": audit["status"], "result": str(RESULT)}))
        return 2
    print(json.dumps({"status": audit["status"], "audit": str(AUDIT)}))
    return 0


def run_fit(protocol_path: Path) -> int:
    protocol, rows, _ = verify_protocol(protocol_path)
    current_audit = _audit(protocol, rows)
    if not AUDIT.is_file() or json.loads(AUDIT.read_text()) != current_audit:
        raise RuntimeError("DSR proposal-support audit is absent or drifted")
    if current_audit["status"] != "PROPOSAL_SUPPORT_AUDIT_PASS":
        raise RuntimeError("DSR fit is prohibited after support-audit failure")
    arms = {}
    joint_models = []
    for arm in ("joint", "RxR", "R2R"):
        arm_result, models = _fit_arm(protocol, rows, arm)
        arms[arm] = arm_result
        if arm == "joint":
            joint_models = models
    joint_pass = arms["joint"]["status"] == "TRAIN_DEVELOPMENT_PASS"
    gate = _save_gate(joint_models, protocol) if joint_pass else None
    result = {
        "schema_version": SCHEMA,
        "status": "TRAIN_DEVELOPMENT_PASS" if joint_pass
        else "TRAIN_DEVELOPMENT_FAIL",
        "revision": REVISION,
        "source_provenance_verified": True,
        "public_unseen_authorized": False,
        "old_confirmation_reused": False,
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "proposal_support_audit": current_audit,
        "model_fit_status": (
            "PASS" if all("outer_oof" in value for value in arms.values())
            else "FAIL"
        ),
        "joint_scientific_status": "PASS" if joint_pass else "FAIL",
        "single_domain_control_status": {
            domain: arms[domain]["status"] for domain in ("RxR", "R2R")
        },
        "outer_folds": arms["joint"].get("outer_folds", []),
        "joint": arms["joint"],
        "rxr_only": arms["RxR"],
        "r2r_only": arms["R2R"],
        "risk_coverage": {
            arm: value["risk_coverage"] for arm, value in arms.items()
        },
        "equal_budget": {
            arm: value["equal_budget"] for arm, value in arms.items()
        },
        "ablations": {
            "status": "FROZEN_FOR_SEPARATE_BATCH_ONLY_IF_MAINLINE_PASS",
            "public_split_access": False,
        },
        "failure_reasons": list(arms["joint"]["failure_reasons"]),
        "public_split_access": protocol["public_split_access"],
        "gate_artifact": gate,
    }
    atomic_json(RESULT, result)
    print(json.dumps({"status": result["status"], "result": str(RESULT)}))
    return 0 if joint_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "fit"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.protocol.resolve() != PROTOCOL.resolve():
        raise RuntimeError("DSR trainer accepts only the frozen protocol path")
    if args.command == "audit":
        return run_audit(args.protocol)
    return run_fit(args.protocol)


if __name__ == "__main__":
    raise SystemExit(main())
