#!/usr/bin/env python3
"""Merge accepted primary features with train-only automatic secondary labels."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
PRIMARY = BASE / (
    "multibranch_v2/RXR_MULTIBRANCH_FEATURE_MANIFEST_V2_AUTHORIZED.json"
)
PRIMARY_AUTH = BASE / (
    "multibranch_v2/RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"
)
SECONDARY_DIR = BASE / "secondary_expansion_v1/multibranch"
SECONDARY = SECONDARY_DIR / "RXR_SECONDARY_FEATURE_MANIFEST.json"
SECONDARY_GATE = SECONDARY_DIR / "RXR_SECONDARY_FEATURE_GATE.json"
TOPOLOGY_ONLY = SECONDARY_DIR / "RXR_SECONDARY_TOPOLOGY_ONLY_MANIFEST.json"
OUT = BASE / "RXR_SECONDARY_AUGMENTED_FEATURE_MANIFEST_V1.json"
AUTH = BASE / "RXR_SECONDARY_AUGMENTED_TRAINING_AUTHORIZATION_V1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def load(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"unsafe or missing input: {path}")
    return json.loads(path.read_text())


def relocated(record: dict, source_manifest: Path, label_source: str) -> dict:
    source_path = (source_manifest.parent / record["path"]).resolve()
    if (
        ROOT not in source_path.parents
        or not source_path.is_file()
        or source_path.is_symlink()
        or source_path.stat().st_size != record["bytes"]
        or sha256_file(source_path) != record["sha256"]
    ):
        raise RuntimeError(f"feature provenance failure: {record['event_id']}")
    output = dict(record)
    output["path"] = os.path.relpath(source_path, OUT.parent)
    output["label_source"] = label_source
    output["quality_role"] = (
        "human_audited_primary" if label_source == "primary_human_audited"
        else "automatic_train_only"
    )
    return output


def main() -> int:
    primary = load(PRIMARY)
    primary_auth = load(PRIMARY_AUTH)
    secondary = load(SECONDARY)
    secondary_gate = load(SECONDARY_GATE)
    topology_only = load(TOPOLOGY_ONLY)
    if not (
        primary.get("schema_version") == "revealnav-mf2-feature-manifest/1"
        and primary.get("metadata", {}).get("training_authorized") is True
        and primary_auth.get("status") == "TRAINING_AUTHORIZATION_PASS"
        and primary_auth.get("training_authorized") is True
        and primary_auth["training_manifest"]["sha256"] == sha256_file(PRIMARY)
        and secondary.get("schema_version") == "revealnav-mf2-feature-manifest/1"
        and secondary.get("metadata", {}).get("training_authorized") is True
        and secondary.get("metadata", {}).get("label_source")
        == "automatic_secondary_pseudolabel"
        and secondary.get("metadata", {}).get("evaluation_use_authorized") is False
        and secondary_gate.get("status")
        == "FEATURE_GATE_PASS_AUTOMATIC_TRAIN_READY"
        and secondary_gate.get("training_authorized") is True
        and secondary_gate["manifest"]["sha256"] == sha256_file(SECONDARY)
    ):
        raise RuntimeError("primary/secondary training authorization failed")

    primary_records = [
        relocated(row, PRIMARY, "primary_human_audited")
        for row in primary["records"]
    ]
    secondary_records = [
        relocated(row, SECONDARY, "automatic_secondary_pseudolabel")
        for row in secondary["records"]
    ]
    primary_ids = {row["event_id"] for row in primary_records}
    secondary_ids = {row["event_id"] for row in secondary_records}
    topology_only_ids = {row["event_id"] for row in topology_only["records"]}
    if len(primary_ids) != len(primary_records):
        raise RuntimeError("duplicate primary event ids")
    if len(secondary_ids) != len(secondary_records):
        raise RuntimeError("duplicate secondary event ids")
    if primary_ids & secondary_ids:
        raise RuntimeError("primary and secondary events overlap")

    gates = {
        "primary_authorization_bound": True,
        "secondary_feature_gate_bound": True,
        "primary_secondary_event_ids_disjoint": True,
        "secondary_records_train_only": bool(secondary_records) and all(
            row["split"] == "train"
            and row["label_source"] == "automatic_secondary_pseudolabel"
            and row["quality_role"] == "automatic_train_only"
            for row in secondary_records
        ),
        "development_is_primary_only": all(
            row["label_source"] == "primary_human_audited"
            for row in primary_records if row["split"] == "development"
        ),
        "gold_is_primary_only": all(
            row["label_source"] == "primary_human_audited"
            for row in primary_records if row["split"] == "gold"
        ),
        "no_topology_only_records": not (
            (primary_ids | secondary_ids) & topology_only_ids
        ),
    }
    if not all(gates.values()):
        raise RuntimeError("augmented manifest leakage gate failed")

    records = primary_records + secondary_records
    split_counts = Counter(row["split"] for row in records)
    source_counts = Counter(row["label_source"] for row in records)
    manifest = {
        "schema_version": "revealnav-mf2-feature-manifest/1",
        "records": records,
        "metadata": {
            "training_authorized": True,
            "augmentation_protocol": "primary_plus_automatic_secondary_train_only_v1",
            "primary_manifest_sha256": sha256_file(PRIMARY),
            "secondary_manifest_sha256": sha256_file(SECONDARY),
            "secondary_feature_gate_sha256": sha256_file(SECONDARY_GATE),
            "development_source": "unchanged_primary_human_audited_split",
            "gold_source": "unchanged_primary_human_audited_split",
            "secondary_evaluation_use_authorized": False,
            "topology_only_training_included": False,
            "normalized_budgets": [1.5, 2.0, 3.0, 4.0],
            "paper_result": False,
        },
    }
    atomic_json(OUT, manifest)

    # Exercise the production reader for train/development only. Gold shards are
    # deliberately not opened in this development experiment.
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from revealnav_mf2 import RevealFeatureDataset

    train = RevealFeatureDataset(OUT, "train")
    development = RevealFeatureDataset(OUT, "development")
    gates.update({
        "merged_train_manifest_loads": len(train) == split_counts["train"],
        "unchanged_development_manifest_loads": (
            len(development) == split_counts["development"]
        ),
    })
    authorized = all(gates.values())
    authorization = {
        "schema_version": "revealnav-mf2-augmentation-authorization/1",
        "status": (
            "AUGMENTED_DEVELOPMENT_EXPERIMENT_AUTHORIZED"
            if authorized else "AUGMENTED_AUTHORIZATION_FAIL"
        ),
        "training_authorized": authorized,
        "scope": (
            "fixed h128 three-seed primary-only versus primary-plus-automatic-"
            "secondary ablation; development only"
        ),
        "manifest": {
            "path": str(OUT.relative_to(ROOT)),
            "sha256": sha256_file(OUT),
        },
        "sources": {
            str(PRIMARY.relative_to(ROOT)): sha256_file(PRIMARY),
            str(PRIMARY_AUTH.relative_to(ROOT)): sha256_file(PRIMARY_AUTH),
            str(SECONDARY.relative_to(ROOT)): sha256_file(SECONDARY),
            str(SECONDARY_GATE.relative_to(ROOT)): sha256_file(SECONDARY_GATE),
            str(TOPOLOGY_ONLY.relative_to(ROOT)): sha256_file(TOPOLOGY_ONLY),
        },
        "counts": {
            "records": len(records),
            "by_split": dict(sorted(split_counts.items())),
            "by_label_source": dict(sorted(source_counts.items())),
        },
        "gates": gates,
        "gold_payload_read": False,
        "paper_result": False,
    }
    atomic_json(AUTH, authorization)
    print(json.dumps({
        "status": authorization["status"],
        "counts": authorization["counts"],
        "manifest": str(OUT.relative_to(ROOT)),
        "manifest_sha256": sha256_file(OUT),
    }, indent=2, sort_keys=True))
    return 0 if authorized else 1


if __name__ == "__main__":
    raise SystemExit(main())
