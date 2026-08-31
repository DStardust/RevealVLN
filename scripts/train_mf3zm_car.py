#!/usr/bin/env python3
"""Train and audit the bounded MF3ZM-CAR v1 revision.

Only the sealed 1,540-row train-development population is consumed.  This
entry point intentionally exposes no confirmation or public-evaluation
command.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/training/mf3zm_car_v1"
PROTOCOL = OUT / "MF3ZM_CAR_PROTOCOL.json"
RESULT = OUT / "MF3ZM_CAR_TRAIN_DEVELOPMENT_RESULT.json"
GATE = OUT / "gates/MF3ZM_CAR_MODEL.pt"
AUDIT = ROOT / (
    "artifacts/training/mf3zl_rcsp_v1r1_audit_fix_v2/"
    "MF3ZL_V1R1_DATA_SUPPORT_AUDIT_CORRECTED.json"
)
V1R1_TRAINER = ROOT / "scripts/train_mf3zl_rcsp_v1r1.py"
V1R1_PROTOCOL = ROOT / (
    "artifacts/training/mf3zl_rcsp_v1r1_train/"
    "MF3ZL_RCSP_V1R1_TRAIN_PROTOCOL.json"
)
DSR_TRAINER = ROOT / "scripts/train_mf3zl_rcsp.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load project module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v1r1():
    return _load_module(V1R1_TRAINER, "sealed_v1r1_car_loader")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or ROOT not in resolved.parents:
        raise RuntimeError(f"invalid project-local source: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError(f"stale atomic output: {part}")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


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


def _canonical_rows() -> list[dict]:
    loader = _v1r1()
    rows = loader._canonical_rows()
    if len(rows) == 0:
        raise RuntimeError("CAR source population is empty")
    return rows


def _identity_hash(rows: list[dict]) -> str:
    return stable_hash([
        {
            "dataset": row["dataset"],
            "scene_id": row["scene_id"],
            "episode_id": row["episode_id"],
            "step": int(row["decision"]["step"]),
            "target": float(row["target"]),
            "feature": row["feature"],
        }
        for row in rows
    ])


def _source_files() -> dict:
    return {
        "method_revision": inventory(ROOT / "METHOD_REVISION_3ZM_CAR.md"),
        "car_model": inventory(ROOT / "revealnav_mf3/car.py"),
        "car_selection": inventory(ROOT / "revealnav_mf3/car_selection.py"),
        "trainer": inventory(Path(__file__).resolve()),
        "v1r1_trainer": inventory(V1R1_TRAINER),
        "v1r1_protocol": inventory(V1R1_PROTOCOL),
        "data_audit": inventory(AUDIT),
        "dsr_trainer": inventory(DSR_TRAINER),
    }


def build_protocol() -> dict:
    audit = json.loads(AUDIT.read_text())
    v1r1_protocol = json.loads(V1R1_PROTOCOL.read_text())
    rows = _canonical_rows()
    domains = dict(Counter(row["dataset"] for row in rows))
    consumed = list(v1r1_protocol.get("known_consumed_scene_ids", []))
    return {
        "schema_version": "revealnav-mf3zm-car-protocol/1",
        "status": "SEALED_BEFORE_MF3ZM_CAR_TRAINING",
        "revision": "mf3zm_car_v1",
        "purpose": "criterion-aligned train-development diagnostic only",
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "domain_counts": domains,
        "canonical_identity_sha256": _identity_hash(rows),
        "frozen_components": [
            "ETP-R1 policy", "MF3V proposal", "MF3ZG hierarchy",
            "exact one-switch labels",
            "utility 0.50 nDTW + 0.25 SDTW + 0.25 SPL",
            "RCSP v1.1 semantic representation and rank-4 architecture",
            "39-scene outer assignment",
        ],
        "consumed_confirmation_reused": False,
        "known_consumed_scene_ids": consumed,
        "training": {
            "outer_folds": 5,
            "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [20260830, 20260831, 20260832],
            "learning_rate": 0.005,
            "dual_learning_rate": 0.05,
            "dual_cap": 100.0,
            "training_steps": 800,
            "inner_fold_salt": "mf3zm-car-v1-inner-scenes/1",
            "decision_rule": "switch_logit > 0",
            "main_representation": "semantic",
            "main_risk_mode": "hard",
            "main_scene_constraint": True,
            "use_cuda": True,
            "continue_after_fold_failure": True,
        },
        "controls": [
            "car_no_scene_constraint",
            "car_soft_risk",
            "car_28d",
            "car_policy_only",
            "car_no_risk",
            "rxr_only_car",
            "r2r_only_car",
            "dsr_v1_expanded_data",
            "fold_domain_matched_target_free_baselines",
        ],
        "source_files": _source_files(),
        "source_protocol_sha256": sha256_file(V1R1_PROTOCOL),
        "source_audit_status": audit.get("status"),
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "authorization": {
            "train_development": True,
            "confirmation": False,
            "public_unseen": False,
        },
    }


def seal() -> int:
    if PROTOCOL.exists():
        raise RuntimeError("CAR protocol already exists; refusing to reseal")
    value = build_protocol()
    if value["source_audit_status"] != "TRAIN_DATA_SUPPORT_PASS":
        raise RuntimeError("source data audit is not PASS")
    rows = _canonical_rows()
    if value["rows"] != len(rows):
        raise RuntimeError("CAR source changed while sealing")
    if set(value["known_consumed_scene_ids"]) & {
        row["scene_id"] for row in rows
    }:
        raise RuntimeError("consumed confirmation scene entered CAR source")
    atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "rows": value["rows"],
        "scenes": value["scenes"],
        "domain_counts": value["domain_counts"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def verify_protocol() -> dict:
    if not PROTOCOL.is_file() or PROTOCOL.is_symlink():
        raise RuntimeError("CAR protocol unavailable")
    value = json.loads(PROTOCOL.read_text())
    expected_public = {
        "val_seen": False, "val_unseen": False,
        "test": False, "test_challenge": False,
    }
    if (
        value.get("status") != "SEALED_BEFORE_MF3ZM_CAR_TRAINING"
        or value.get("revision") != "mf3zm_car_v1"
        or value.get("public_split_access") != expected_public
        or value.get("authorization", {}).get("public_unseen") is not False
        or value.get("consumed_confirmation_reused") is not False
    ):
        raise RuntimeError("CAR protocol semantics drift")
    for item in value["source_files"].values():
        path = ROOT / item["path"]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != int(item["bytes"])
            or sha256_file(path) != str(item["sha256"])
        ):
            raise RuntimeError(f"CAR source drift: {item['path']}")
    if sha256_file(V1R1_PROTOCOL) != value["source_protocol_sha256"]:
        raise RuntimeError("v1r1 protocol drift")
    audit = json.loads(AUDIT.read_text())
    if audit.get("status") != "TRAIN_DATA_SUPPORT_PASS":
        raise RuntimeError("CAR data audit no longer passes")
    rows = _canonical_rows()
    if (
        len(rows) != int(value["rows"])
        or len({row["scene_id"] for row in rows}) != int(value["scenes"])
        or dict(Counter(row["dataset"] for row in rows)) != value["domain_counts"]
        or _identity_hash(rows) != value["canonical_identity_sha256"]
    ):
        raise RuntimeError("CAR canonical source drift")
    if set(value["known_consumed_scene_ids"]) & {
        row["scene_id"] for row in rows
    }:
        raise RuntimeError("consumed confirmation scene entered CAR source")
    return value


def _load_data() -> tuple[dict, list[dict], dict]:
    protocol = verify_protocol()
    loader = _v1r1()
    rows = loader._canonical_rows()
    parent_protocol = json.loads(
        (ROOT / "artifacts/training/mf3zl_rcsp_v1/"
         "MF3ZL_RCSP_PROTOCOL.json").read_text()
    )
    arrays = loader._arrays(parent_protocol, rows)
    if len(rows) != int(protocol["rows"]):
        raise RuntimeError("CAR data row count drift")
    return protocol, rows, arrays


def _subset(rows: list[dict], arrays: dict, mask: np.ndarray):
    selected_rows = [row for row, keep in zip(rows, mask, strict=True) if keep]
    return selected_rows, {
        "semantic": {
            name: value[mask] for name, value in arrays["semantic"].items()
        },
        "engineered": {
            name: value[mask] for name, value in arrays["engineered"].items()
        },
        "policy_only": {
            "policy_only": arrays["semantic"]["policy"][mask]
        },
        **{
            name: arrays[name][mask] for name in (
                "target", "scenes", "datasets", "episodes", "tiers",
                "outer_folds",
            )
        },
    }


def _config(protocol: dict) -> dict:
    source = protocol["training"]
    return {
        "outer_folds": int(source["outer_folds"]),
        "inner_folds": int(source["inner_folds"]),
        "weight_decay_grid": list(source["weight_decay_grid"]),
        "seeds": list(source["seeds"]),
        "learning_rate": float(source["learning_rate"]),
        "dual_learning_rate": float(source["dual_learning_rate"]),
        "dual_cap": float(source["dual_cap"]),
        "training_steps": int(source["training_steps"]),
        "inner_fold_salt": str(source["inner_fold_salt"]),
        "use_cuda": bool(source["use_cuda"]),
    }


def _fit_arm(
    protocol: dict,
    rows: list[dict],
    arrays: dict,
    *,
    arm: str = "joint",
    representation: str = "semantic",
    risk_mode: str = "hard",
    scene_constraint: bool = True,
) -> tuple[dict, list[torch.nn.Module]]:
    from revealnav_mf3.car_selection import nested_car_fit

    mask = np.ones(len(rows), dtype=bool)
    if arm in {"RxR", "R2R"}:
        mask &= arrays["datasets"] == arm
    elif arm != "joint":
        raise ValueError(f"unknown CAR arm: {arm}")
    selected_rows, selected = _subset(rows, arrays, mask)
    inputs = {
        "semantic": selected["semantic"],
        "engineered_28d": selected["engineered"],
        "policy_only": selected["policy_only"],
    }[representation]
    fit = nested_car_fit(
        selected_rows, inputs, selected["target"], selected["scenes"],
        selected["datasets"], selected["episodes"], selected["outer_folds"],
        _config(protocol), representation=representation,
        risk_mode=risk_mode, scene_constraint=scene_constraint,
        continue_after_fold_failure=True,
    )
    models = fit.pop("final_models")
    return {
        "arm": arm,
        "representation": representation,
        "risk_mode": risk_mode,
        "scene_constraint": bool(scene_constraint),
        "rows": len(selected_rows),
        "scenes": len(set(selected["scenes"])),
        **_jsonable(fit),
    }, models


def _fit_dsr_control(
    protocol: dict, rows: list[dict], arrays: dict,
) -> dict:
    """Run the already frozen DSR selector as a data-only diagnostic control."""

    module = _load_module(DSR_TRAINER, "sealed_dsr_expanded_control")
    value = module._fit_dsr_expanded(protocol, rows, arrays)
    return _jsonable(value)


def _save_gate(models: list[torch.nn.Module], protocol: dict) -> dict:
    from revealnav_mf3.car import (
        CAR_CHECKPOINT_SCHEMA, CAR_POLICY_FEATURE_NAMES,
    )
    if not models:
        raise RuntimeError("cannot save an empty CAR gate")
    payload = {
        "schema_version": CAR_CHECKPOINT_SCHEMA,
        "revision": "mf3zm_car_v1",
        "policy_feature_names": list(CAR_POLICY_FEATURE_NAMES),
        "embedding_dim": 768,
        "rank": 4,
        "decision_rule": "switch_logit > 0",
        "state_dicts": [
            {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            }
            for model in models
        ],
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
    }
    GATE.parent.mkdir(parents=True, exist_ok=True)
    part = GATE.with_name(GATE.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError("stale CAR gate partial")
    torch.save(payload, part)
    os.replace(part, GATE)
    return inventory(GATE)


def fit() -> int:
    if RESULT.exists() or GATE.exists():
        raise RuntimeError("refusing to overwrite CAR outputs")
    protocol, rows, arrays = _load_data()
    main, models = _fit_arm(protocol, rows, arrays)
    controls: dict[str, dict] = {}
    control_errors: dict[str, dict] = {}
    specs = (
        ("car_no_scene_constraint", {"scene_constraint": False}),
        ("car_soft_risk", {"risk_mode": "soft"}),
        ("car_28d", {"representation": "engineered_28d"}),
        ("car_policy_only", {"representation": "policy_only"}),
        ("car_no_risk", {"risk_mode": "none"}),
        ("rxr_only_car", {"arm": "RxR"}),
        ("r2r_only_car", {"arm": "R2R"}),
    )
    for name, kwargs in specs:
        try:
            value, _ = _fit_arm(protocol, rows, arrays, **kwargs)
            controls[name] = value
        except Exception as exc:  # preserve diagnostic and fail overall
            control_errors[name] = {
                "status": "CONTROL_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
    try:
        controls["dsr_v1_expanded_data"] = _fit_dsr_control(
            protocol, rows, arrays
        )
    except Exception as exc:
        control_errors["dsr_v1_expanded_data"] = {
            "status": "CONTROL_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    main_pass = main.get("status") == "NESTED_CAR_PASS"
    gate = _save_gate(models, protocol) if main_pass else None
    result = {
        "schema_version": "revealnav-mf3zm-car-result/1",
        "status": "TRAIN_DEVELOPMENT_PASS" if main_pass else "TRAIN_DEVELOPMENT_FAIL",
        "revision": "mf3zm_car_v1",
        "source_protocol": inventory(PROTOCOL),
        "source_audit": inventory(AUDIT),
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "domains": dict(Counter(row["dataset"] for row in rows)),
        "mainline": main,
        "controls_run_independently_of_mainline": True,
        "controls": controls,
        "control_errors": control_errors,
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
    atomic_json(RESULT, _jsonable(result))
    print(json.dumps({
        "status": result["status"],
        "rows": result["rows"],
        "scenes": result["scenes"],
        "checkpoint_created": result["checkpoint_created"],
        "main_failure_reasons": main.get("failure_reasons", []),
        "control_error_count": len(control_errors),
    }, indent=2, sort_keys=True))
    return 0 if main_pass and not control_errors else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "audit-data", "fit"))
    args = parser.parse_args()
    if args.command == "seal":
        return seal()
    if args.command == "audit-data":
        value = verify_protocol()
        print(json.dumps({
            "status": "TRAIN_DATA_SUPPORT_PASS",
            "revision": value["revision"],
            "rows": value["rows"],
            "scenes": value["scenes"],
            "domain_counts": value["domain_counts"],
            "public_unseen_authorized": False,
        }, indent=2, sort_keys=True))
        return 0
    return fit()


if __name__ == "__main__":
    raise SystemExit(main())
