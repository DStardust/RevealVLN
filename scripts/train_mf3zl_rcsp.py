#!/usr/bin/env python3
"""Audit and fit the pre-sealed MF3ZL-RCSP train-development revision."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (SCRIPTS, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from collect_mf3zl_exact_replay import (  # noqa: E402
    AUDIT,
    MANIFEST,
    OUT,
    PROTOCOL,
    atomic_json,
    inventory,
    sha256_file,
    stable_hash,
    verify_protocol,
)
from revealnav_mf3.dsr_selection import (  # noqa: E402
    nested_distributional_fit,
    scene_cluster_bootstrap,
    stratified_equal_budget_baselines,
)
from revealnav_mf3.rcsp import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    POLICY_FEATURE_NAMES,
    policy_features,
)
from revealnav_mf3.rcsp_selection import (  # noqa: E402
    nested_rcsp_fit,
    rcsp_risk_coverage_diagnostic,
)
from seal_mf3zk_dsr_protocol import (  # noqa: E402
    PROTOCOL as DSR_PROTOCOL,
    verify_protocol as verify_dsr_protocol,
)
from train_mf3zk_joint_action_aligned_gate import _vector  # noqa: E402


RESULT = OUT / "MF3ZL_RCSP_TRAIN_DEVELOPMENT_RESULT.json"
GATE = OUT / "gates/MF3ZL_RCSP_MODEL.pt"
SCHEMA = "revealnav-mf3zl-rcsp-train-development-result/1"


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


def _checked_feature(pointer: dict) -> dict[str, np.ndarray]:
    path = ROOT / pointer["path"]
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(pointer["bytes"])
        or sha256_file(path) != str(pointer["sha256"])
    ):
        raise RuntimeError(f"MF3ZL feature provenance drift: {pointer['path']}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name], dtype=np.float64) for name in (
            "instruction", "checkpoint", "native", "alternative"
        )}
    if any(value.shape != (768,) or not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("MF3ZL feature shape/value drift")
    return arrays


def _canonical_rows() -> list[dict]:
    _, old_rows, _ = verify_dsr_protocol(DSR_PROTOCOL)
    manifest = json.loads(MANIFEST.read_text())
    if (
        manifest.get("status") != "DENSE_EXACT_REPLAY_READY"
        or manifest.get("source_protocol_sha256") != sha256_file(PROTOCOL)
        or manifest.get("public_split_access") is not False
    ):
        raise RuntimeError("MF3ZL dense source manifest drift")
    rows = []
    for row in old_rows:
        rows.append({
            "dataset": str(row["dataset"]),
            "tier": str(row["tier"]),
            "scene_id": str(row["scene_id"]),
            "episode_id": str(row["episode_id"]),
            "decision": row["decision"],
            "target": float(row["target"]),
            "arrays": row["arrays"],
            "feature": row["feature"],
            "source": "mf3zk_dsr_v1_existing_exact",
        })
    for raw in manifest["records"]:
        if not (
            raw.get("exact_prefix_verified") is True
            and raw.get("exact_one_switch_verified") is True
            and math.isclose(
                float(raw["target"]), float(raw["delta"]["utility"]),
                rel_tol=0.0, abs_tol=0.0,
            )
        ):
            raise RuntimeError("MF3ZL dense exact-label contract drift")
        rows.append({
            "dataset": str(raw["dataset"]),
            "tier": str(raw["tier"]),
            "scene_id": str(raw["scene_id"]),
            "episode_id": str(raw["episode_id"]),
            "decision": raw["decision"],
            "target": float(raw["target"]),
            "arrays": _checked_feature(raw["feature"]),
            "feature": raw["feature"],
            "source": "mf3zl_dense_exact_replay",
        })
    seen = {}
    canonical = []
    for row in rows:
        identity = (
            row["dataset"], row["episode_id"], int(row["decision"]["step"])
        )
        signature = stable_hash({
            "dataset": row["dataset"], "scene_id": row["scene_id"],
            "episode_id": row["episode_id"], "tier": row["tier"],
            "decision": row["decision"], "target": row["target"],
            "feature": row["feature"],
        })
        if identity in seen:
            if seen[identity] != signature:
                raise RuntimeError("conflicting MF3ZL canonical exact identity")
            continue
        seen[identity] = signature
        canonical.append(row)
    canonical.sort(key=lambda row: (
        row["dataset"], row["scene_id"], row["episode_id"],
        int(row["decision"]["step"]),
    ))
    return canonical


def _arrays(protocol: dict, rows: list[dict]) -> dict:
    policy = np.stack([policy_features(row["decision"]) for row in rows])
    semantic = {
        "policy": policy,
        "instruction": np.stack([row["arrays"]["instruction"] for row in rows]),
        "history": np.stack([row["arrays"]["checkpoint"] for row in rows]),
        "native": np.stack([row["arrays"]["native"] for row in rows]),
        "runner": np.stack([row["arrays"]["alternative"] for row in rows]),
    }
    engineered = {
        "engineered": np.stack([_vector(row) for row in rows]).astype(np.float64)
    }
    target = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    scenes = np.asarray([str(row["scene_id"]) for row in rows])
    datasets = np.asarray([str(row["dataset"]) for row in rows])
    episodes = np.asarray([str(row["episode_id"]) for row in rows])
    tiers = np.asarray([str(row["tier"]) for row in rows])
    mapping = protocol["training"]["outer_scene_assignment"]
    if not set(scenes) <= set(mapping):
        raise RuntimeError("MF3ZL row scene is outside sealed outer mapping")
    folds = np.asarray([int(mapping[scene]) for scene in scenes], dtype=np.int64)
    if (
        any(not np.isfinite(value).all() for value in (*semantic.values(), *engineered.values()))
        or not np.isfinite(target).all()
    ):
        raise RuntimeError("MF3ZL non-finite train-development input")
    return {
        "semantic": semantic, "engineered": engineered,
        "target": target, "scenes": scenes, "datasets": datasets,
        "episodes": episodes, "tiers": tiers, "outer_folds": folds,
    }


def audit_data() -> tuple[dict, list[dict], dict]:
    protocol, _ = verify_protocol()
    value = json.loads(AUDIT.read_text())
    if (
        value.get("status") != "TRAIN_DATA_SUPPORT_PASS"
        or value.get("rcsp_training_authorized") is not True
        or value.get("public_unseen_authorized") is not False
        or value.get("source_protocol_sha256") != sha256_file(PROTOCOL)
        or value.get("source_manifest") != inventory(MANIFEST)
    ):
        raise RuntimeError("MF3ZL data support gate does not authorize training")
    rows = _canonical_rows()
    expected = sum(
        int(value["domains"][domain]["combined_unique_exact_events"])
        for domain in ("RxR", "R2R")
    )
    if len(rows) != expected:
        raise RuntimeError("MF3ZL canonical row count differs from data audit")
    arrays = _arrays(protocol, rows)
    return protocol, rows, arrays


def _config(protocol: dict) -> dict:
    source = protocol["training"]
    return {
        "outer_folds": int(source["outer_folds"]),
        "inner_folds": int(source["inner_folds"]),
        "weight_decay_grid": list(source["weight_decay_grid"]),
        "seeds": list(source["seeds"]),
        "learning_rate": float(source["learning_rate"]),
        "dual_learning_rate": float(source["dual_learning_rate"]),
        "training_steps": int(source["training_steps"]),
        "inner_fold_salt": str(source["inner_fold_salt"]),
    }


def _subset(rows: list[dict], arrays: dict, mask: np.ndarray):
    selected_rows = [row for row, keep in zip(rows, mask, strict=True) if keep]
    return selected_rows, {
        "semantic": {name: value[mask] for name, value in arrays["semantic"].items()},
        "engineered": {name: value[mask] for name, value in arrays["engineered"].items()},
        **{
            name: arrays[name][mask] for name in (
                "target", "scenes", "datasets", "episodes", "tiers", "outer_folds"
            )
        },
    }


def _fit_rcsp(
    protocol: dict, rows: list[dict], arrays: dict,
    *, arm: str, representation: str, risk_constrained: bool,
) -> tuple[dict, list]:
    mask = np.ones(len(rows), dtype=bool)
    if arm in {"RxR", "R2R"}:
        mask &= arrays["datasets"] == arm
    elif arm.startswith("tier:"):
        mask &= arrays["tiers"] == arm.split(":", 1)[1]
    elif arm != "joint":
        raise ValueError("unknown MF3ZL training arm")
    selected_rows, selected = _subset(rows, arrays, mask)
    inputs = (
        selected["semantic"] if representation == "semantic"
        else selected["engineered"]
    )
    fit = nested_rcsp_fit(
        selected_rows, inputs, selected["target"], selected["scenes"],
        selected["datasets"], selected["episodes"], selected["outer_folds"],
        _config(protocol), risk_constrained=risk_constrained,
        representation=representation,
    )
    models = fit.pop("final_models")
    oof_rows = []
    if "outer_oof" in fit:
        logits = fit["outer_oof"]["switch_logit"]
        gate = fit["outer_oof"]["authorized_mask"]
        for index, row in enumerate(selected_rows):
            oof_rows.append({
                "dataset": row["dataset"], "scene_id": row["scene_id"],
                "episode_id": row["episode_id"],
                "decision_step": int(row["decision"]["step"]),
                "tier": row["tier"],
                "outer_fold": int(selected["outer_folds"][index]),
                "target_delta_utility": float(selected["target"][index]),
                "switch_logit": float(logits[index]),
                "authorized": bool(gate[index]),
            })
        fit["risk_coverage"] = rcsp_risk_coverage_diagnostic(
            logits, selected["target"], selected["scenes"]
        )
        baseline_rows = [
            {**row, "target": float(value)}
            for row, value in zip(selected_rows, selected["target"], strict=True)
        ]
        controls = stratified_equal_budget_baselines(
            baseline_rows, selected["target"], gate, selected["outer_folds"],
            seed=20260830,
        )
        comparator = controls["internal_masks"]["fold_domain_matched"][
            "low_native_margin"
        ]
        fit["scene_cluster_bootstrap"] = scene_cluster_bootstrap(
            gate, selected["target"], selected["scenes"], selected["datasets"],
            comparator_mask=comparator, replicates=10_000, seed=20260830,
        )
    return {
        "arm": arm,
        "representation": representation,
        "risk_constrained": risk_constrained,
        "rows": len(selected_rows),
        "scenes": len(set(selected["scenes"])),
        **_jsonable(fit),
        "outer_oof_rows": oof_rows,
    }, models


def _fit_dsr_expanded(protocol: dict, rows: list[dict], arrays: dict) -> dict:
    dsr_protocol = json.loads(DSR_PROTOCOL.read_text())
    config = {
        "outer_folds": 5, "inner_folds": 4,
        "weight_decay_grid": [0.0001, 0.001, 0.01],
        "seeds": list(dsr_protocol["model"]["ensemble_seeds"]),
        "learning_rate": float(dsr_protocol["selection"]["learning_rate"]),
        "training_steps": int(dsr_protocol["selection"]["training_steps"]),
        "inner_fold_salt": str(dsr_protocol["selection"]["inner_fold_salt"]),
    }
    fit = nested_distributional_fit(
        arrays["engineered"]["engineered"], arrays["target"], arrays["scenes"],
        arrays["datasets"], arrays["outer_folds"], config,
    )
    fit.pop("final_models", None)
    return {
        "arm": "joint",
        "control": "frozen_dsr_v1_on_expanded_data",
        "source_dsr_protocol_sha256": sha256_file(DSR_PROTOCOL),
        **_jsonable(fit),
    }


def _save_gate(models: list, protocol: dict) -> dict:
    if not models:
        raise RuntimeError("cannot save an empty MF3ZL gate")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "revision": "mf3zl_rcsp_v1",
        "policy_feature_names": list(POLICY_FEATURE_NAMES),
        "embedding_dim": 768,
        "rank": 4,
        "decision_rule": "switch_logit > 0",
        "state_dicts": [
            {name: value.detach().cpu() for name, value in model.state_dict().items()}
            for model in models
        ],
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
    }
    GATE.parent.mkdir(parents=True, exist_ok=True)
    part = GATE.with_name(GATE.name + ".part")
    torch.save(payload, part)
    os.replace(part, GATE)
    return inventory(GATE)


def fit() -> int:
    if RESULT.exists() or GATE.exists():
        raise RuntimeError("refusing to overwrite MF3ZL train-development outputs")
    protocol, rows, arrays = audit_data()
    main, models = _fit_rcsp(
        protocol, rows, arrays, arm="joint",
        representation="semantic", risk_constrained=True,
    )
    controls = {}
    complete_oof = "outer_oof" in main
    if complete_oof:
        for name, kwargs in (
            ("rcsp_28d", {
                "arm": "joint", "representation": "engineered_28d",
                "risk_constrained": True,
            }),
            ("rcsp_no_risk", {
                "arm": "joint", "representation": "semantic",
                "risk_constrained": False,
            }),
            ("rxr_only", {
                "arm": "RxR", "representation": "semantic",
                "risk_constrained": True,
            }),
            ("r2r_only", {
                "arm": "R2R", "representation": "semantic",
                "risk_constrained": True,
            }),
            ("core_only", {
                "arm": "tier:core", "representation": "semantic",
                "risk_constrained": True,
            }),
            ("expansion_only", {
                "arm": "tier:expansion", "representation": "semantic",
                "risk_constrained": True,
            }),
        ):
            value, _ = _fit_rcsp(protocol, rows, arrays, **kwargs)
            controls[name] = value
        controls["dsr_v1_expanded"] = _fit_dsr_expanded(protocol, rows, arrays)
    passed = main.get("status") == "NESTED_RCSP_PASS"
    gate = _save_gate(models, protocol) if passed else None
    result = {
        "schema_version": SCHEMA,
        "status": "TRAIN_DEVELOPMENT_PASS" if passed else "TRAIN_DEVELOPMENT_FAIL",
        "revision": "mf3zl_rcsp_v1",
        "source_provenance_verified": True,
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "source_data_audit": inventory(AUDIT),
        "source_manifest": inventory(MANIFEST),
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "domains": dict(Counter(row["dataset"] for row in rows)),
        "mainline": main,
        "controls_run_only_after_complete_main_outer_oof": complete_oof,
        "controls": controls,
        "model": gate,
        "checkpoint_created": gate is not None,
        "old_confirmation_reused": False,
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
        "public_split_access": {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        },
    }
    atomic_json(RESULT, result)
    print(json.dumps({
        "status": result["status"], "rows": result["rows"],
        "scenes": result["scenes"], "checkpoint_created": result["checkpoint_created"],
        "failure_reasons": main.get("failure_reasons", []),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-data", "fit"))
    args = parser.parse_args()
    if args.command == "audit-data":
        protocol, rows, arrays = audit_data()
        print(json.dumps({
            "status": "TRAIN_DATA_SUPPORT_PASS",
            "protocol_sha256": sha256_file(PROTOCOL),
            "rows": len(rows),
            "scenes": len(set(arrays["scenes"])),
            "domains": dict(Counter(arrays["datasets"])),
            "public_unseen_authorized": False,
        }, indent=2, sort_keys=True))
        return 0
    return fit()


if __name__ == "__main__":
    raise SystemExit(main())
