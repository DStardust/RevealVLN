#!/usr/bin/env python3
"""Read-only v2 correction for the sealed v1r1 data-support audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_SCRIPT = ROOT / "scripts/audit_mf3zl_rcsp_v1r1_fix.py"
V1R1_OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1r1"
V1R1_PROTOCOL = V1R1_OUT / "MF3ZL_RCSP_V1R1_PROTOCOL.json"
V1R1_SELECTION = V1R1_OUT / "MF3ZL_R2R_VARIANT_SELECTION.json"
V1R1_TARGETS = V1R1_OUT / "MF3ZL_R2R_VARIANT_TARGETS.json"
V1R1_TARGET_PROGRESS = V1R1_OUT / "MF3ZL_R2R_VARIANT_TARGET_PROGRESS.json"
V1R1_MANIFEST = V1R1_OUT / "MF3ZL_R2R_VARIANT_MANIFEST.json"
OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1r1_audit_fix_v2"
PROTOCOL = OUT / "MF3ZL_V1R1_AUDIT_FIX_V2_PROTOCOL.json"
AUDIT = OUT / "MF3ZL_V1R1_DATA_SUPPORT_AUDIT_CORRECTED.json"
PARENT_PROTOCOL = ROOT / "artifacts/training/mf3zl_rcsp_v1/MF3ZL_RCSP_PROTOCOL.json"
PARENT_MANIFEST = ROOT / "artifacts/training/mf3zl_rcsp_v1/MF3ZL_EXACT_REPLAY_MANIFEST.json"
DSR_PROTOCOL = ROOT / "artifacts/training/mf3zk_dsr_v1/MF3ZK_DSR_PROTOCOL.json"


def _base():
    spec = importlib.util.spec_from_file_location("v1r1_audit_fix_v1", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load first audit correction")
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


def build_protocol() -> dict:
    base = _base()
    base.verify_protocol()
    return {
        "schema_version": "revealnav-mf3zl-v1r1-audit-fix-v2-protocol/1",
        "status": "SEALED_BEFORE_V1R1_AUDIT_CORRECTION_V2",
        "revision": "mf3zl_rcsp_v1r1_audit_fix_v2",
        "parent_revision": "mf3zl_rcsp_v1r1_audit_fix",
        "purpose": "read-only correction of event-identity and record-schema checks",
        "correction": {
            "identity_key": ["dataset", "scene_id", "episode_id", "decision_step"],
            "episode_id_global_uniqueness_assumed": False,
            "manifest_level_public_split_flag_used": True,
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
            "audit_fix_v2_script": inventory(Path(__file__).resolve()),
            "audit_fix_v1_script": inventory(BASE_SCRIPT),
            "sealed_v1r1_collector": inventory(
                ROOT / "scripts/collect_mf3zl_rcsp_v1r1.py"
            ),
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
        raise RuntimeError("audit-fix v2 protocol already exists; refusing reseal")
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
        raise RuntimeError("audit-fix v2 protocol unavailable")
    value = json.loads(PROTOCOL.read_text())
    if (
        value.get("status") != "SEALED_BEFORE_V1R1_AUDIT_CORRECTION_V2"
        or value.get("revision") != "mf3zl_rcsp_v1r1_audit_fix_v2"
        or value.get("correction", {}).get("identity_key")
        != ["dataset", "scene_id", "episode_id", "decision_step"]
        or value.get("public_split_access") != {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        }
    ):
        raise RuntimeError("audit-fix v2 protocol semantics drift")
    for section in ("source_files", "implementation_files"):
        for item in value[section].values():
            path = ROOT / item["path"]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(item["bytes"])
                or sha256_file(path) != str(item["sha256"])
            ):
                raise RuntimeError(f"audit-fix v2 source drift: {item['path']}")
    return value


def _identity(row: dict, *, canonical: bool) -> tuple[str, str, str, int]:
    if canonical:
        value = row.get("identity")
        if not isinstance(value, dict) or value.get("decision_step") is None:
            raise RuntimeError("canonical identity schema drift")
        return (
            str(value["dataset"]), str(row["scene_id"]),
            str(value["episode_id"]), int(value["decision_step"]),
        )
    event = row.get("event_identity", {})
    step = row.get("decision_step", event.get("step"))
    if step is None:
        raise RuntimeError("manifest decision step schema drift")
    return (
        str(row["dataset"]), str(row["scene_id"]),
        str(row["episode_id"]), int(step),
    )


def _feature_is_intact(pointer: dict) -> bool:
    if not isinstance(pointer, dict):
        return False
    path = ROOT / str(pointer.get("path", ""))
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == int(pointer.get("bytes", -1))
        and sha256_file(path) == str(pointer.get("sha256", ""))
    )


def audit() -> int:
    if AUDIT.exists():
        raise RuntimeError("refusing to overwrite corrected v2 audit")
    protocol = verify_protocol()
    collector = _base()._collector()
    collector.verify_protocol()
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
        or int(targets.get("counts", {}).get("events", -1))
        != len(manifest.get("records", []))
    ):
        raise RuntimeError("v1r1 sealed execution state drift")
    base = _base()
    existing, parent_dense, extension = base._records()
    failures: list[str] = []
    signatures: dict[tuple[str, str, str, int], str] = {}
    scenes: dict[tuple[str, str, str, int], str] = {}

    def add(row: dict, *, canonical: bool) -> None:
        key = _identity(row, canonical=canonical)
        signature = stable_hash(row)
        previous = signatures.get(key)
        if previous is not None and previous != signature:
            failures.append("conflicting_exact_identity")
        elif previous is None:
            signatures[key] = signature
            scenes[key] = key[1]

    for row in existing:
        add(row, canonical=True)
    for row in parent_dense:
        add(row, canonical=False)
    for row in extension:
        add(row, canonical=False)
        if row.get("dataset") != "R2R":
            failures.append("v1r1_non_r2r_record")
        if row.get("exact_prefix_verified") is not True:
            failures.append("v1r1_prefix_not_verified")
        if row.get("exact_one_switch_verified") is not True:
            failures.append("v1r1_one_switch_not_verified")
        if row.get("target") != row.get("delta", {}).get("utility"):
            failures.append("v1r1_target_delta_mismatch")
        if not _feature_is_intact(row.get("feature")):
            failures.append("v1r1_feature_provenance_drift")

    domains = {}
    for domain in ("RxR", "R2R"):
        keys = [key for key in signatures if key[0] == domain]
        domains[domain] = {
            "combined_unique_exact_events": len(keys),
            "combined_development_scenes": len({scenes[key] for key in keys}),
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
        "schema_version": "revealnav-mf3zl-v1r1-audit-fix-v2-result/1",
        "status": "TRAIN_DATA_SUPPORT_PASS" if not failures else "TRAIN_DATA_SUPPORT_FAIL",
        "revision": "mf3zl_rcsp_v1r1_audit_fix_v2",
        "parent_revision": "mf3zl_rcsp_v1r1",
        "source_protocol": inventory(PROTOCOL),
        "source_v1r1_protocol": inventory(V1R1_PROTOCOL),
        "source_v1r1_manifest": inventory(V1R1_MANIFEST),
        "source_parent_manifest": inventory(PARENT_MANIFEST),
        "correction_applied": [
            "identity.decision_step field name",
            "scene-qualified event identity",
            "manifest-level public_split_access",
        ],
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
            "combined_unique_rows": len(signatures),
            "conflicting_identities": failures.count("conflicting_exact_identity"),
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
