#!/usr/bin/env python3
"""Versioned read-only correction for the v1r1 combined-data audit."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V1R1_SCRIPT = ROOT / "scripts/collect_mf3zl_rcsp_v1r1.py"
V1R1_OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1r1"
V1R1_PROTOCOL = V1R1_OUT / "MF3ZL_RCSP_V1R1_PROTOCOL.json"
V1R1_SELECTION = V1R1_OUT / "MF3ZL_R2R_VARIANT_SELECTION.json"
V1R1_TARGETS = V1R1_OUT / "MF3ZL_R2R_VARIANT_TARGETS.json"
V1R1_TARGET_PROGRESS = V1R1_OUT / "MF3ZL_R2R_VARIANT_TARGET_PROGRESS.json"
V1R1_MANIFEST = V1R1_OUT / "MF3ZL_R2R_VARIANT_MANIFEST.json"
OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1r1_audit_fix"
PROTOCOL = OUT / "MF3ZL_V1R1_AUDIT_FIX_PROTOCOL.json"
AUDIT = OUT / "MF3ZL_V1R1_DATA_SUPPORT_AUDIT_CORRECTED.json"
PARENT_PROTOCOL = ROOT / "artifacts/training/mf3zl_rcsp_v1/MF3ZL_RCSP_PROTOCOL.json"
PARENT_MANIFEST = ROOT / "artifacts/training/mf3zl_rcsp_v1/MF3ZL_EXACT_REPLAY_MANIFEST.json"
DSR_PROTOCOL = ROOT / "artifacts/training/mf3zk_dsr_v1/MF3ZK_DSR_PROTOCOL.json"


def _collector():
    spec = importlib.util.spec_from_file_location(
        "sealed_mf3zl_v1r1_collector", V1R1_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed v1r1 collector")
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


def stable_hash(value) -> str:
    return sha256_file_value(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode())


def sha256_file_value(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


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


def build_protocol() -> dict:
    collector = _collector()
    collector.verify_protocol()
    return {
        "schema_version": "revealnav-mf3zl-v1r1-audit-fix-protocol/1",
        "status": "SEALED_BEFORE_V1R1_AUDIT_CORRECTION",
        "revision": "mf3zl_rcsp_v1r1_audit_fix",
        "parent_revision": "mf3zl_rcsp_v1r1",
        "purpose": "read-only correction of a documented audit field-name mismatch",
        "correction": {
            "old_field": "identity.step",
            "actual_field": "identity.decision_step",
            "rollouts_rerun": False,
            "labels_changed": False,
            "gate_relaxed": False,
        },
        "source_files": {
            "v1r1_protocol": inventory(V1R1_PROTOCOL),
            "v1r1_selection": inventory(V1R1_SELECTION),
            "v1r1_targets": inventory(V1R1_TARGETS),
            "v1r1_target_progress": inventory(V1R1_TARGET_PROGRESS),
            "v1r1_manifest": inventory(V1R1_MANIFEST),
            "parent_protocol": inventory(PARENT_PROTOCOL),
            "parent_manifest": inventory(PARENT_MANIFEST),
            "dsr_protocol": inventory(DSR_PROTOCOL),
        },
        "implementation_files": {
            "audit_fix_script": inventory(Path(__file__).resolve()),
            "sealed_v1r1_collector": inventory(V1R1_SCRIPT),
        },
        "data_gate": {
            "minimum_combined_exact_events_per_domain": 300,
            "minimum_development_scenes_per_domain": 30,
            "maximum_conflicting_identities": 0,
        },
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "authorization": {
            "rcsp_training": False,
            "confirmation": False,
            "public_unseen": False,
        },
    }


def seal() -> int:
    if PROTOCOL.exists():
        raise RuntimeError("audit-fix protocol already exists; refusing reseal")
    value = build_protocol()
    atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "revision": value["revision"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def verify_protocol() -> dict:
    if not PROTOCOL.is_file() or PROTOCOL.is_symlink():
        raise RuntimeError("audit-fix protocol unavailable")
    value = json.loads(PROTOCOL.read_text())
    if (
        value.get("status") != "SEALED_BEFORE_V1R1_AUDIT_CORRECTION"
        or value.get("revision") != "mf3zl_rcsp_v1r1_audit_fix"
        or value.get("public_split_access") != {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        }
    ):
        raise RuntimeError("audit-fix protocol semantics drift")
    for section in ("source_files", "implementation_files"):
        for item in value[section].values():
            path = ROOT / item["path"]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(item["bytes"])
                or sha256_file(path) != str(item["sha256"])
            ):
                raise RuntimeError(f"audit-fix source drift: {item['path']}")
    return value


def _identity(row: dict, *, canonical: bool) -> tuple[str, str, int]:
    if canonical:
        value = row.get("identity")
        if not isinstance(value, dict):
            raise RuntimeError("canonical identity missing")
        step = value.get("decision_step")
        if step is None:
            raise RuntimeError("canonical decision_step missing")
        return str(value["dataset"]), str(value["episode_id"]), int(step)
    event = row.get("event_identity", {})
    step = row.get("decision_step", event.get("step"))
    if step is None:
        raise RuntimeError("manifest decision step missing")
    return str(row["dataset"]), str(row["episode_id"]), int(step)


def _scene(row: dict, *, canonical: bool) -> str:
    return str(row["scene_id"] if not canonical else row["scene_id"])


def _records() -> tuple[list[dict], list[dict], list[dict]]:
    dsr = json.loads(DSR_PROTOCOL.read_text())
    existing = list(dsr["source_inventory"]["canonical_rows"])
    parent_manifest = json.loads(PARENT_MANIFEST.read_text())
    parent_dense = list(parent_manifest.get("records", []))
    extension = list(json.loads(V1R1_MANIFEST.read_text())["records"])
    return existing, parent_dense, extension


def audit() -> int:
    if AUDIT.exists():
        raise RuntimeError("refusing to overwrite corrected audit")
    protocol = verify_protocol()
    collector = _collector()
    sealed, selection = collector.verify_protocol()
    manifest = json.loads(V1R1_MANIFEST.read_text())
    targets = json.loads(V1R1_TARGETS.read_text())
    target_progress = json.loads(V1R1_TARGET_PROGRESS.read_text())
    if (
        manifest.get("status") != "R2R_VARIANT_EXACT_REPLAY_READY"
        or manifest.get("complete_population_executed") is not True
        or manifest.get("public_split_access") is not False
        or manifest.get("source_protocol_sha256") != sha256_file(V1R1_PROTOCOL)
        or target_progress.get("status") != "COMPLETE"
        or target_progress.get("failed") != 0
        or targets.get("status") != "SEALED_AFTER_COMPLETE_NATIVE_SHADOW_BEFORE_TREATMENTS"
        or targets.get("source_selection_sha256") != sha256_file(V1R1_SELECTION)
    ):
        raise RuntimeError("v1r1 sealed execution state drift")
    existing, parent_dense, extension = _records()
    failures: list[str] = []
    by_identity: dict[tuple[str, str, int], tuple[str, str]] = {}
    scene_by_identity: dict[tuple[str, str, int], str] = {}

    def add(row: dict, *, canonical: bool, source: str) -> None:
        key = _identity(row, canonical=canonical)
        signature = stable_hash(row)
        scene = _scene(row, canonical=canonical)
        previous = by_identity.get(key)
        if previous is not None and previous[0] != signature:
            failures.append("conflicting_exact_identity")
        elif previous is None:
            by_identity[key] = (signature, source)
            scene_by_identity[key] = scene

    for row in existing:
        add(row, canonical=True, source="existing_exact")
    for row in parent_dense:
        add(row, canonical=False, source="parent_dense")
    for row in extension:
        add(row, canonical=False, source="v1r1_variant")
        if row.get("dataset") != "R2R":
            failures.append("v1r1_non_r2r_record")
        if row.get("exact_prefix_verified") is not True:
            failures.append("v1r1_prefix_not_verified")
        if row.get("exact_one_switch_verified") is not True:
            failures.append("v1r1_one_switch_not_verified")
        if row.get("public_split_access") is not False:
            failures.append("v1r1_public_split_access")
        if row.get("target") != row.get("delta", {}).get("utility"):
            failures.append("v1r1_target_delta_mismatch")

    domains = {}
    for domain in ("RxR", "R2R"):
        keys = [key for key in by_identity if key[0] == domain]
        scenes = {scene_by_identity[key] for key in keys}
        domains[domain] = {
            "combined_unique_exact_events": len(keys),
            "combined_development_scenes": len(scenes),
            "existing_exact_events": sum(
                1 for row in existing if _identity(row, canonical=True)[0] == domain
            ),
            "parent_dense_exact_events": sum(
                1 for row in parent_dense if _identity(row, canonical=False)[0] == domain
            ),
            "v1r1_variant_exact_events": sum(
                1 for row in extension if _identity(row, canonical=False)[0] == domain
            ),
        }
        if domains[domain]["combined_unique_exact_events"] < 300:
            failures.append(f"{domain}:fewer_than_300_exact_events")
        if domains[domain]["combined_development_scenes"] < 30:
            failures.append(f"{domain}:fewer_than_30_development_scenes")

    parent_episode_ids = {
        _identity(row, canonical=True)[1] for row in existing
    } | {
        _identity(row, canonical=False)[1] for row in parent_dense
    }
    extension_episode_ids = {
        _identity(row, canonical=False)[1] for row in extension
    }
    if parent_episode_ids & extension_episode_ids:
        failures.append("historical_episode_overlap")
    manifest_counts = manifest.get("counts", {})
    actual_positive = sum(float(row["target"]) > 0 for row in extension)
    actual_catastrophic = sum(bool(row.get("catastrophic")) for row in extension)
    if (
        int(manifest_counts.get("exact_events", -1)) != len(extension)
        or int(manifest_counts.get("positive", -1)) != actual_positive
        or int(manifest_counts.get("catastrophic", -1)) != actual_catastrophic
        or int(manifest_counts.get("conflicting_identities", -1)) != 0
    ):
        failures.append("v1r1_manifest_count_mismatch")
    value = {
        "schema_version": "revealnav-mf3zl-v1r1-audit-fix-result/1",
        "status": "TRAIN_DATA_SUPPORT_PASS" if not failures else "TRAIN_DATA_SUPPORT_FAIL",
        "revision": "mf3zl_rcsp_v1r1_audit_fix",
        "parent_revision": "mf3zl_rcsp_v1r1",
        "source_protocol": inventory(PROTOCOL),
        "source_v1r1_protocol": inventory(V1R1_PROTOCOL),
        "source_v1r1_manifest": inventory(V1R1_MANIFEST),
        "source_parent_manifest": inventory(PARENT_MANIFEST),
        "source_selection": inventory(V1R1_SELECTION),
        "source_targets": inventory(V1R1_TARGETS),
        "correction_applied": "identity.decision_step field name",
        "rollouts_rerun": False,
        "labels_changed": False,
        "complete_population_executed": True,
        "adaptive_stopping_used": False,
        "untouched_scenes_consumed": False,
        "public_split_access": False,
        "domains": domains,
        "counts": {
            "existing_canonical_rows": len(existing),
            "parent_dense_rows": len(parent_dense),
            "v1r1_variant_rows": len(extension),
            "combined_unique_rows": len(by_identity),
            "conflicting_identities": sum(
                1 for reason in failures if reason == "conflicting_exact_identity"
            ),
        },
        "failure_reasons": sorted(set(failures)),
        "rcsp_training_authorized": not failures,
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
    }
    atomic_json(AUDIT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "audit"))
    args = parser.parse_args()
    return seal() if args.command == "seal" else audit()


if __name__ == "__main__":
    raise SystemExit(main())
