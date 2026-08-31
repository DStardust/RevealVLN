#!/usr/bin/env python3
"""Pre-registered RxR-only diagnostic for the RCSP v1.1 correction.

This diagnostic uses the same sealed RxR development rows and nested selection
contract as the v1 probe.  It is a new revision because the original v1 model
aborted on legitimate zero ``runner - native`` semantic differences.
"""

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

OUT = ROOT / "artifacts/training/mf3zl_rcsp_rxr_probe_v1_1"
PROTOCOL = OUT / "MF3ZL_RXR_PROBE_PROTOCOL.json"
RESULT = OUT / "MF3ZL_RXR_PROBE_RESULT.json"
ABORT = ROOT / "artifacts/training/mf3zl_rcsp_rxr_probe_v1/MF3ZL_RXR_PROBE_ABORT.json"
PARENT_OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1"
PARENT_PROTOCOL = PARENT_OUT / "MF3ZL_RCSP_PROTOCOL.json"
PARENT_SELECTION = PARENT_OUT / "MF3ZL_EXACT_REPLAY_SELECTION.json"
PARENT_MANIFEST = PARENT_OUT / "MF3ZL_EXACT_REPLAY_MANIFEST.json"
DSR_PROTOCOL = ROOT / "artifacts/training/mf3zk_dsr_v1/MF3ZK_DSR_PROTOCOL.json"


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


def _trainer_module():
    path = ROOT / "scripts/train_mf3zl_rcsp.py"
    spec = importlib.util.spec_from_file_location("mf3zl_parent_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed parent trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _implementation_paths() -> tuple[Path, ...]:
    return (
        ROOT / "scripts/probe_mf3zl_rcsp_rxr_v1_1.py",
        ROOT / "scripts/train_mf3zl_rcsp.py",
        ROOT / "revealnav_mf3/rcsp_v1_1.py",
        ROOT / "revealnav_mf3/rcsp_selection_v1_1.py",
        ROOT / "revealnav_mf3/dsr_selection.py",
    )


def _zero_relative_delta_events(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        difference = (
            np.asarray(row["arrays"]["alternative"], dtype=float)
            - np.asarray(row["arrays"]["native"], dtype=float)
        )
        if float(np.linalg.norm(difference)) <= 1e-8:
            result.append({
                "dataset": row["dataset"],
                "episode_id": row["episode_id"],
                "scene_id": row["scene_id"],
                "step": int(row["decision"]["step"]),
                "tier": row["tier"],
            })
    return result


def build_protocol() -> dict:
    trainer = _trainer_module()
    rows = [row for row in trainer._canonical_rows() if row["dataset"] == "RxR"]
    scenes = sorted({row["scene_id"] for row in rows})
    if len(rows) < 300 or len(scenes) < 30:
        raise RuntimeError("sealed RxR source does not meet diagnostic support")
    return {
        "schema_version": "revealnav-mf3zl-rxr-probe-protocol/2",
        "status": "SEALED_BEFORE_RXR_DIAGNOSTIC",
        "revision": "mf3zl_rcsp_rxr_probe_v1_1",
        "parent_algorithm_revision": "mf3zl_rcsp_v1_1_zero_relative_delta",
        "purpose": "early domain-specific diagnostic; not a public or joint result",
        "source_rows": len(rows),
        "source_scenes": len(scenes),
        "source_domain": "RxR",
        "selection_is_outcome_blind": True,
        "training_contract": {
            "representation": "semantic",
            "risk_constrained": True,
            "outer_folds": 5,
            "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "decision_rule": "switch_logit > 0",
            "no_checkpoint": True,
        },
        "zero_relative_delta_correction": {
            "status": "PRE_REGISTERED_CORRECTNESS_RULE",
            "condition": "norm(runner_embedding - native_embedding) <= 1e-8",
            "action": "retain_event_and_use_zero_relative_semantic_vector",
            "absolute_instruction_history_native_runner_norms_still_required": True,
            "selection_or_outcome_fields_used": [],
            "events": _zero_relative_delta_events(rows),
        },
        "public_split_access": {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        },
        "authorization": {
            "training_checkpoint": False,
            "confirmation": False,
            "public_unseen": False,
        },
        "source_files": {
            "parent_protocol": inventory(PARENT_PROTOCOL),
            "parent_selection": inventory(PARENT_SELECTION),
            "parent_manifest": inventory(PARENT_MANIFEST),
            "dsr_protocol": inventory(DSR_PROTOCOL),
        },
        "implementation_files": {
            str(path.relative_to(ROOT)): inventory(path)
            for path in _implementation_paths()
        },
    }


def seal() -> int:
    if PROTOCOL.exists():
        raise RuntimeError("RxR probe protocol already exists; refusing reseal")
    protocol = build_protocol()
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({
        "status": protocol["status"],
        "source_rows": protocol["source_rows"],
        "source_scenes": protocol["source_scenes"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def verify_protocol() -> dict:
    if not PROTOCOL.is_file() or PROTOCOL.is_symlink():
        raise RuntimeError("RxR probe protocol is unavailable")
    protocol = json.loads(PROTOCOL.read_text())
    if (
        protocol.get("status") != "SEALED_BEFORE_RXR_DIAGNOSTIC"
        or protocol.get("schema_version") != "revealnav-mf3zl-rxr-probe-protocol/2"
        or protocol.get("revision") != "mf3zl_rcsp_rxr_probe_v1_1"
        or protocol.get("public_split_access") != {
            "test": False, "test_challenge": False,
            "val_seen": False, "val_unseen": False,
        }
    ):
        raise RuntimeError("RxR probe protocol semantics drift")
    for section in ("source_files", "implementation_files"):
        for item in protocol[section].values():
            path = ROOT / item["path"]
            if (
                path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                raise RuntimeError(f"RxR probe source drift: {item['path']}")
    return protocol


def run() -> int:
    if RESULT.exists():
        raise RuntimeError("refusing to overwrite RxR probe result")
    protocol = verify_protocol()
    trainer = _trainer_module()
    # Do not call parent audit_data: its joint gate is intentionally FAIL because
    # R2R has 253 events. The probe reads the same sealed canonical loader and
    # filters only the pre-declared RxR domain.
    all_rows = trainer._canonical_rows()
    rows = [row for row in all_rows if row["dataset"] == "RxR"]
    if len(rows) != int(protocol["source_rows"]):
        raise RuntimeError("RxR probe row count drift")
    arrays = trainer._arrays(json.loads(PARENT_PROTOCOL.read_text()), rows)
    parent_protocol = json.loads(PARENT_PROTOCOL.read_text())
    zero_delta_mask = np.linalg.norm(
        arrays["semantic"]["runner"] - arrays["semantic"]["native"],
        axis=1,
    ) <= 1e-8
    from revealnav_mf3.rcsp_selection_v1_1 import nested_rcsp_fit

    fit = nested_rcsp_fit(
        rows,
        arrays["semantic"],
        arrays["target"],
        arrays["scenes"],
        arrays["datasets"],
        arrays["episodes"],
        arrays["outer_folds"],
        trainer._config(parent_protocol),
        risk_constrained=True,
        representation="semantic",
    )
    # Models are deliberately discarded: this artifact is a diagnostic only.
    fit.pop("final_models", None)
    result = {
        "schema_version": "revealnav-mf3zl-rxr-probe-result/2",
        "status": (
            "RXR_ONLY_DIAGNOSTIC_PASS"
            if fit.get("status") == "NESTED_RCSP_PASS"
            else "RXR_ONLY_DIAGNOSTIC_FAIL"
        ),
        "algorithm_fit_status": fit.get("status"),
        "revision": "mf3zl_rcsp_rxr_probe_v1_1",
        "parent_algorithm_revision": "mf3zl_rcsp_v1_1_zero_relative_delta",
        "source_protocol": inventory(PROTOCOL),
        "source_parent_manifest": inventory(PARENT_MANIFEST),
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "checkpoint_created": False,
        "zero_relative_delta_count": int(zero_delta_mask.sum()),
        "zero_relative_delta_events": protocol["zero_relative_delta_correction"]["events"],
        "training_checkpoint_authorized": False,
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
        "public_split_access": {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        },
        "diagnostic": fit,
    }
    atomic_json(RESULT, result)
    print(json.dumps({
        "status": result["status"],
        "algorithm_fit_status": result["algorithm_fit_status"],
        "rows": result["rows"],
        "scenes": result["scenes"],
        "checkpoint_created": False,
        "failure_reasons": fit.get("failure_reasons", []),
    }, indent=2, sort_keys=True))
    return 0 if result["status"] == "RXR_ONLY_DIAGNOSTIC_PASS" else 2


def monitor() -> int:
    if RESULT.is_file():
        print(RESULT.read_text(), end="")
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
