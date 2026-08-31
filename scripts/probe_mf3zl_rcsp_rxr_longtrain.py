#!/usr/bin/env python3
"""One fixed 3x-duration RxR-only RCSP development diagnostic."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_SCRIPT = ROOT / "scripts/probe_mf3zl_rcsp_rxr_v1_1.py"
BASE_PROTOCOL = ROOT / "artifacts/training/mf3zl_rcsp_rxr_probe_v1_1/MF3ZL_RXR_PROBE_PROTOCOL.json"
OUT = ROOT / "artifacts/training/mf3zl_rcsp_rxr_longtrain"
PROTOCOL = OUT / "MF3ZL_RXR_LONGTRAIN_PROTOCOL.json"
RESULT = OUT / "MF3ZL_RXR_LONGTRAIN_RESULT.json"
PARENT_PROTOCOL = ROOT / "artifacts/training/mf3zl_rcsp_v1/MF3ZL_RCSP_PROTOCOL.json"
PARENT_MANIFEST = ROOT / "artifacts/training/mf3zl_rcsp_v1/MF3ZL_EXACT_REPLAY_MANIFEST.json"


def _base_module():
    spec = importlib.util.spec_from_file_location("sealed_rxr_v11_probe", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed RxR v1.1 probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError(f"stale partial output: {part}")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def build_protocol() -> dict:
    base = _base_module()
    base_protocol = base.verify_protocol()
    if base_protocol.get("source_rows") != 997 or base_protocol.get("source_scenes") != 38:
        raise RuntimeError("sealed RxR v1.1 support drift")
    implementation = {
        "method_revision": inventory(ROOT / "METHOD_REVISION_3ZL_RCSP_RXR_LONGTRAIN.md"),
        "longtrain_probe": inventory(Path(__file__).resolve()),
        "base_probe": inventory(BASE_SCRIPT),
        "rcsp_v1_1": inventory(ROOT / "revealnav_mf3/rcsp_v1_1.py"),
        "rcsp_selection_v1_1": inventory(ROOT / "revealnav_mf3/rcsp_selection_v1_1.py"),
        "dsr_selection": inventory(ROOT / "revealnav_mf3/dsr_selection.py"),
        "parent_loader": inventory(ROOT / "scripts/train_mf3zl_rcsp.py"),
    }
    return {
        "schema_version": "revealnav-mf3zl-rxr-longtrain-protocol/1",
        "status": "SEALED_BEFORE_RXR_LONGTRAIN_DIAGNOSTIC",
        "revision": "mf3zl_rcsp_rxr_longtrain_v1",
        "parent_revision": "mf3zl_rcsp_rxr_probe_v1_1",
        "purpose": "single optimization-duration diagnostic; not a public or joint result",
        "source_rows": 997,
        "source_scenes": 38,
        "source_domain": "RxR",
        "selection_is_outcome_blind": True,
        "training_contract": {
            "representation": "semantic",
            "risk_constrained": True,
            "outer_folds": 5,
            "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "decision_rule": "switch_logit > 0",
            "baseline_training_steps": 800,
            "training_steps": 2400,
            "duration_multiplier": 3,
            "duration_search": False,
            "no_checkpoint": True,
        },
        "zero_relative_delta_correction": base_protocol[
            "zero_relative_delta_correction"
        ],
        "source_files": {
            "base_protocol": inventory(BASE_PROTOCOL),
            "parent_protocol": inventory(PARENT_PROTOCOL),
            "parent_manifest": inventory(PARENT_MANIFEST),
        },
        "implementation_files": implementation,
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "authorization": {
            "training_checkpoint": False,
            "confirmation": False,
            "public_unseen": False,
        },
    }


def seal() -> int:
    if PROTOCOL.exists():
        raise RuntimeError("longtrain protocol already exists; refusing reseal")
    value = build_protocol()
    atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "revision": value["revision"],
        "training_steps": value["training_contract"]["training_steps"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def verify_protocol() -> dict:
    if not PROTOCOL.is_file() or PROTOCOL.is_symlink():
        raise RuntimeError("longtrain protocol unavailable")
    value = json.loads(PROTOCOL.read_text())
    if (
        value.get("status") != "SEALED_BEFORE_RXR_LONGTRAIN_DIAGNOSTIC"
        or value.get("revision") != "mf3zl_rcsp_rxr_longtrain_v1"
        or value.get("training_contract", {}).get("training_steps") != 2400
        or value.get("training_contract", {}).get("duration_search") is not False
        or value.get("public_split_access") != {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        }
    ):
        raise RuntimeError("longtrain protocol semantics drift")
    for section in ("source_files", "implementation_files"):
        for item in value[section].values():
            path = ROOT / item["path"]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(item["bytes"])
                or sha256_file(path) != str(item["sha256"])
            ):
                raise RuntimeError(f"longtrain source drift: {item['path']}")
    return value


def run() -> int:
    if RESULT.exists():
        raise RuntimeError("refusing to overwrite longtrain result")
    protocol = verify_protocol()
    base = _base_module()
    parent_protocol = json.loads(PARENT_PROTOCOL.read_text())
    trainer = base._trainer_module()
    rows = [row for row in trainer._canonical_rows() if row["dataset"] == "RxR"]
    if len(rows) != int(protocol["source_rows"]):
        raise RuntimeError("longtrain RxR row count drift")
    arrays = trainer._arrays(parent_protocol, rows)
    config = trainer._config(parent_protocol)
    config["training_steps"] = int(protocol["training_contract"]["training_steps"])
    from revealnav_mf3.rcsp_selection_v1_1 import nested_rcsp_fit

    zero_delta_mask = np.linalg.norm(
        arrays["semantic"]["runner"] - arrays["semantic"]["native"], axis=1
    ) <= 1e-8
    fit = nested_rcsp_fit(
        rows,
        arrays["semantic"],
        arrays["target"],
        arrays["scenes"],
        arrays["datasets"],
        arrays["episodes"],
        arrays["outer_folds"],
        config,
        risk_constrained=True,
        representation="semantic",
    )
    fit.pop("final_models", None)
    fit = trainer._jsonable(fit)
    result = {
        "schema_version": "revealnav-mf3zl-rxr-longtrain-result/1",
        "status": (
            "RXR_ONLY_LONGTRAIN_DIAGNOSTIC_PASS"
            if fit.get("status") == "NESTED_RCSP_PASS"
            else "RXR_ONLY_LONGTRAIN_DIAGNOSTIC_FAIL"
        ),
        "algorithm_fit_status": fit.get("status"),
        "revision": "mf3zl_rcsp_rxr_longtrain_v1",
        "parent_revision": "mf3zl_rcsp_rxr_probe_v1_1",
        "source_protocol": inventory(PROTOCOL),
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "training_steps": 2400,
        "zero_relative_delta_count": int(zero_delta_mask.sum()),
        "checkpoint_created": False,
        "training_checkpoint_authorized": False,
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "diagnostic": fit,
    }
    atomic_json(RESULT, result)
    print(json.dumps({
        "status": result["status"],
        "algorithm_fit_status": result["algorithm_fit_status"],
        "rows": result["rows"],
        "scenes": result["scenes"],
        "training_steps": result["training_steps"],
        "checkpoint_created": False,
        "failure_reasons": fit.get("failure_reasons", []),
    }, indent=2, sort_keys=True))
    return 0 if result["status"] == "RXR_ONLY_LONGTRAIN_DIAGNOSTIC_PASS" else 2


def monitor() -> int:
    if RESULT.is_file():
        value = json.loads(RESULT.read_text())
        print(json.dumps({
            key: value.get(key)
            for key in (
                "status", "algorithm_fit_status", "rows", "scenes",
                "training_steps", "checkpoint_created",
            )
        }, indent=2, sort_keys=True))
    else:
        print(json.dumps({"status": "PENDING", "protocol": PROTOCOL.is_file()}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "run", "monitor"))
    args = parser.parse_args()
    if args.command == "seal":
        return seal()
    if args.command == "run":
        return run()
    return monitor()


if __name__ == "__main__":
    raise SystemExit(main())
