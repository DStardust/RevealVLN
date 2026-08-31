#!/usr/bin/env python3
"""Train MF3ZL-RCSP v1.1 on the sealed expanded development population."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1r1_train"
PROTOCOL = OUT / "MF3ZL_RCSP_V1R1_TRAIN_PROTOCOL.json"
RESULT = OUT / "MF3ZL_RCSP_V1R1_TRAIN_DEVELOPMENT_RESULT.json"
GATE = OUT / "gates/MF3ZL_RCSP_V1R1_MODEL.pt"
AUDIT = ROOT / (
    "artifacts/training/mf3zl_rcsp_v1r1_audit_fix_v2/"
    "MF3ZL_V1R1_DATA_SUPPORT_AUDIT_CORRECTED.json"
)
V1R1_MANIFEST = ROOT / (
    "artifacts/training/mf3zl_rcsp_v1r1/MF3ZL_R2R_VARIANT_MANIFEST.json"
)
PARENT_MANIFEST = ROOT / (
    "artifacts/training/mf3zl_rcsp_v1/MF3ZL_EXACT_REPLAY_MANIFEST.json"
)
PARENT_PROTOCOL = ROOT / "artifacts/training/mf3zl_rcsp_v1/MF3ZL_RCSP_PROTOCOL.json"
DSR_PROTOCOL = ROOT / "artifacts/training/mf3zk_dsr_v1/MF3ZK_DSR_PROTOCOL.json"
V1R1_AUDIT_SCRIPT = ROOT / "scripts/audit_mf3zl_rcsp_v1r1_fix_v2.py"
PARENT_TRAINER = ROOT / "scripts/train_mf3zl_rcsp.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load project module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parent_trainer():
    return _load_module(PARENT_TRAINER, "sealed_parent_rcsp_loader")


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if ROOT not in resolved.parents or not path.is_file() or path.is_symlink():
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


def _checked_feature(pointer: dict) -> dict[str, np.ndarray]:
    path = ROOT / str(pointer["path"])
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(pointer["bytes"])
        or sha256_file(path) != str(pointer["sha256"])
    ):
        raise RuntimeError(f"expanded feature provenance drift: {path}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            name: np.asarray(payload[name], dtype=np.float64)
            for name in ("instruction", "checkpoint", "native", "alternative")
        }
    if any(value.shape != (768,) or not np.isfinite(value).all()
           for value in arrays.values()):
        raise RuntimeError("expanded feature shape/value drift")
    return arrays


def _normal_row(raw: dict, *, source: str, arrays: dict | None = None) -> dict:
    if not (
        raw.get("exact_prefix_verified") is True
        and raw.get("exact_one_switch_verified") is True
        and math.isclose(
            float(raw["target"]), float(raw["delta"]["utility"]),
            rel_tol=0.0, abs_tol=0.0,
        )
    ):
        raise RuntimeError(f"{source} exact-label contract drift")
    feature = raw["feature"]
    return {
        "dataset": str(raw["dataset"]),
        "tier": str(raw["tier"]),
        "scene_id": str(raw["scene_id"]),
        "episode_id": str(raw["episode_id"]),
        "decision": raw["decision"],
        "target": float(raw["target"]),
        "arrays": arrays if arrays is not None else _checked_feature(feature),
        "feature": feature,
        "source": source,
    }


def _canonical_rows() -> list[dict]:
    parent = _parent_trainer()
    _, old_rows, _ = parent.verify_dsr_protocol(parent.DSR_PROTOCOL)
    parent_manifest = json.loads(PARENT_MANIFEST.read_text())
    v1r1_manifest = json.loads(V1R1_MANIFEST.read_text())
    if parent_manifest.get("status") != "DENSE_EXACT_REPLAY_READY":
        raise RuntimeError("parent dense manifest status drift")
    if v1r1_manifest.get("status") != "R2R_VARIANT_EXACT_REPLAY_READY":
        raise RuntimeError("v1r1 manifest status drift")
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
    for raw in parent_manifest["records"]:
        rows.append(_normal_row(raw, source="mf3zl_parent_dense_exact"))
    for raw in v1r1_manifest["records"]:
        rows.append(_normal_row(raw, source="mf3zl_v1r1_variant_exact"))
    seen: dict[tuple[str, str, str, int], str] = {}
    canonical = []
    for row in rows:
        identity = (
            row["dataset"], row["scene_id"], row["episode_id"],
            int(row["decision"]["step"]),
        )
        signature = stable_hash({
            "dataset": row["dataset"], "scene_id": row["scene_id"],
            "episode_id": row["episode_id"], "tier": row["tier"],
            "decision": row["decision"], "target": row["target"],
            "feature": row["feature"],
        })
        if identity in seen:
            if seen[identity] != signature:
                raise RuntimeError("conflicting expanded exact identity")
            continue
        seen[identity] = signature
        canonical.append(row)
    canonical.sort(key=lambda row: (
        row["dataset"], row["scene_id"], row["episode_id"],
        int(row["decision"]["step"]),
    ))
    return canonical


def _identity_hash(rows: list[dict]) -> str:
    return stable_hash([
        {
            "dataset": row["dataset"],
            "scene_id": row["scene_id"],
            "episode_id": row["episode_id"],
            "step": int(row["decision"]["step"]),
            "target": row["target"],
            "feature": row["feature"],
        }
        for row in rows
    ])


def _source_snapshot() -> dict:
    return {
        "audit": inventory(AUDIT),
        "v1r1_manifest": inventory(V1R1_MANIFEST),
        "parent_manifest": inventory(PARENT_MANIFEST),
        "parent_protocol": inventory(PARENT_PROTOCOL),
        "dsr_protocol": inventory(DSR_PROTOCOL),
    }


def _implementation_snapshot() -> dict:
    paths = {
        "method_revision": ROOT / "METHOD_REVISION_3ZL_RCSP_V1R1_TRAIN.md",
        "trainer": Path(__file__).resolve(),
        "rcsp_v1_1": ROOT / "revealnav_mf3/rcsp_v1_1.py",
        "rcsp_selection_v1_1": ROOT / "revealnav_mf3/rcsp_selection_v1_1.py",
        "dsr_selection": ROOT / "revealnav_mf3/dsr_selection.py",
        "parent_loader": PARENT_TRAINER,
        "audit_correction": V1R1_AUDIT_SCRIPT,
    }
    return {name: inventory(path) for name, path in paths.items()}


def build_protocol() -> dict:
    audit = json.loads(AUDIT.read_text())
    rows = _canonical_rows()
    return {
        "schema_version": "revealnav-mf3zl-v1r1-train-protocol/1",
        "status": "SEALED_BEFORE_MF3ZL_V1R1_TRAINING",
        "revision": "mf3zl_rcsp_v1r1_train",
        "algorithm_revision": "mf3zl_rcsp_v1_1_zero_relative_delta",
        "data_revision": "mf3zl_rcsp_v1r1",
        "purpose": "expanded train-development fit; no confirmation or public split",
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "domain_counts": dict(Counter(row["dataset"] for row in rows)),
        "canonical_identity_sha256": _identity_hash(rows),
        "training": {
            "outer_folds": 5,
            "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [20260830, 20260831, 20260832],
            "learning_rate": 0.005,
            "dual_learning_rate": 0.05,
            "training_steps": 800,
            "inner_fold_salt": "mf3zl-rcsp-v1-inner-scenes/1",
            "representation": "semantic",
            "risk_constrained": True,
            "decision_rule": "switch_logit > 0",
            "checkpoint_only_if_nested_pass": True,
        },
        "frozen_components": [
            "ETP-R1 policy", "MF3V proposal", "MF3ZG hierarchy",
            "exact one-switch labels", "utility 0.50 nDTW + 0.25 SDTW + 0.25 SPL",
        ],
        "consumed_confirmation_reused": False,
        "known_consumed_scene_ids": json.loads(DSR_PROTOCOL.read_text()).get(
            "known_consumed_scene_ids", []
        ),
        "source_files": _source_snapshot(),
        "implementation_files": _implementation_snapshot(),
        "source_audit_status": audit.get("status"),
        "public_split_access": {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        },
        "authorization": {
            "training": False,
            "confirmation": False,
            "public_unseen": False,
        },
    }


def seal() -> int:
    if PROTOCOL.exists():
        raise RuntimeError("v1r1 train protocol already exists; refusing reseal")
    value = build_protocol()
    if value["source_audit_status"] != "TRAIN_DATA_SUPPORT_PASS":
        raise RuntimeError("corrected data audit does not pass")
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
        raise RuntimeError("v1r1 train protocol unavailable")
    value = json.loads(PROTOCOL.read_text())
    if (
        value.get("status") != "SEALED_BEFORE_MF3ZL_V1R1_TRAINING"
        or value.get("revision") != "mf3zl_rcsp_v1r1_train"
        or value.get("algorithm_revision") != "mf3zl_rcsp_v1_1_zero_relative_delta"
        or value.get("public_split_access") != {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        }
    ):
        raise RuntimeError("v1r1 train protocol semantics drift")
    for section in ("source_files", "implementation_files"):
        for item in value[section].values():
            path = ROOT / item["path"]
            if (
                not path.is_file() or path.is_symlink()
                or path.stat().st_size != int(item["bytes"])
                or sha256_file(path) != str(item["sha256"])
            ):
                raise RuntimeError(f"v1r1 train source drift: {item['path']}")
    if json.loads(AUDIT.read_text()).get("status") != "TRAIN_DATA_SUPPORT_PASS":
        raise RuntimeError("corrected data audit is no longer PASS")
    rows = _canonical_rows()
    if (
        len(rows) != int(value["rows"])
        or len({row["scene_id"] for row in rows}) != int(value["scenes"])
        or dict(Counter(row["dataset"] for row in rows)) != value["domain_counts"]
        or _identity_hash(rows) != value["canonical_identity_sha256"]
    ):
        raise RuntimeError("v1r1 train canonical source drift")
    if set(value.get("known_consumed_scene_ids", [])) & {
        row["scene_id"] for row in rows
    }:
        raise RuntimeError("consumed confirmation scene entered training")
    return value


def _arrays(parent_protocol: dict, rows: list[dict]) -> dict:
    parent = _parent_trainer()
    arrays = parent._arrays(parent_protocol, rows)
    # The parent helper is reused only for deterministic feature construction;
    # model fitting below uses the v1.1 selection backend.
    return arrays


def _config(protocol: dict) -> dict:
    return dict(protocol["training"])


def _subset(rows: list[dict], arrays: dict, mask: np.ndarray):
    selected_rows = [row for row, keep in zip(rows, mask, strict=True) if keep]
    return selected_rows, {
        "semantic": {
            name: value[mask] for name, value in arrays["semantic"].items()
        },
        "engineered": {
            name: value[mask] for name, value in arrays["engineered"].items()
        },
        **{
            name: arrays[name][mask] for name in (
                "target", "scenes", "datasets", "episodes", "tiers", "outer_folds"
            )
        },
    }


def _fit_arm(protocol: dict, rows: list[dict], arrays: dict, *, arm: str,
             representation: str = "semantic", risk_constrained: bool = True):
    from revealnav_mf3.rcsp_selection_v1_1 import (
        nested_rcsp_fit, rcsp_risk_coverage_diagnostic,
    )
    mask = np.ones(len(rows), dtype=bool)
    if arm in {"RxR", "R2R"}:
        mask &= arrays["datasets"] == arm
    elif arm.startswith("tier:"):
        mask &= arrays["tiers"] == arm.split(":", 1)[1]
    elif arm != "joint":
        raise ValueError(f"unknown v1r1 training arm: {arm}")
    selected_rows, selected = _subset(rows, arrays, mask)
    inputs = selected["semantic"] if representation == "semantic" else selected["engineered"]
    fit = nested_rcsp_fit(
        selected_rows, inputs, selected["target"], selected["scenes"],
        selected["datasets"], selected["episodes"], selected["outer_folds"],
        _config(protocol), risk_constrained=risk_constrained,
        representation=representation,
    )
    models = fit.pop("final_models")
    if "outer_oof" in fit:
        logits = np.asarray(fit["outer_oof"]["switch_logit"])
        fit["risk_coverage"] = rcsp_risk_coverage_diagnostic(
            logits, selected["target"], selected["scenes"]
        )
    return {
        "arm": arm,
        "representation": representation,
        "risk_constrained": risk_constrained,
        "rows": len(selected_rows),
        "scenes": len(set(selected["scenes"])),
        **_jsonable(fit),
    }, models


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


def _save_gate(models: list, protocol: dict) -> dict:
    from revealnav_mf3.rcsp_v1_1 import (
        CHECKPOINT_SCHEMA, POLICY_FEATURE_NAMES,
    )
    if not models:
        raise RuntimeError("cannot save empty v1r1 gate")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "revision": "mf3zl_rcsp_v1r1_train",
        "policy_feature_names": list(POLICY_FEATURE_NAMES),
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
        raise RuntimeError("stale v1r1 gate partial")
    torch.save(payload, part)
    os.replace(part, GATE)
    return inventory(GATE)


def fit() -> int:
    if RESULT.exists() or GATE.exists():
        raise RuntimeError("refusing to overwrite v1r1 training output")
    protocol = verify_protocol()
    rows = _canonical_rows()
    arrays = _arrays(json.loads(PARENT_PROTOCOL.read_text()), rows)
    main, models = _fit_arm(protocol, rows, arrays, arm="joint")
    # Controls are intentionally run only after a complete mainline OOF.  They
    # use the same sealed data and configuration and never authorize public use.
    controls = {}
    complete_oof = "outer_oof" in main
    if complete_oof:
        for name, kwargs in (
            ("rcsp_28d", {"representation": "engineered_28d"}),
            ("rcsp_no_risk", {"risk_constrained": False}),
            ("rxr_only", {"arm": "RxR"}),
            ("r2r_only", {"arm": "R2R"}),
        ):
            arm = kwargs.pop("arm", "joint")
            value, _ = _fit_arm(protocol, rows, arrays, arm=arm, **kwargs)
            controls[name] = value
    passed = main.get("status") == "NESTED_RCSP_PASS"
    gate = _save_gate(models, protocol) if passed else None
    result = {
        "schema_version": "revealnav-mf3zl-v1r1-train-result/1",
        "status": "TRAIN_DEVELOPMENT_PASS" if passed else "TRAIN_DEVELOPMENT_FAIL",
        "revision": "mf3zl_rcsp_v1r1_train",
        "algorithm_revision": "mf3zl_rcsp_v1_1_zero_relative_delta",
        "source_protocol": inventory(PROTOCOL),
        "source_audit": inventory(AUDIT),
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
    atomic_json(RESULT, _jsonable(result))
    print(json.dumps({
        "status": result["status"],
        "rows": result["rows"],
        "scenes": result["scenes"],
        "checkpoint_created": result["checkpoint_created"],
        "failure_reasons": main.get("failure_reasons", []),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


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
            "rows": value["rows"],
            "scenes": value["scenes"],
            "domain_counts": value["domain_counts"],
            "public_unseen_authorized": False,
        }, indent=2, sort_keys=True))
        return 0
    return fit()


if __name__ == "__main__":
    raise SystemExit(main())
