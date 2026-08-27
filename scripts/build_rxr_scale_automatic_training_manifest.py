#!/usr/bin/env python3
"""Build the Gold-free automatic-scale training manifest."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
PRIMARY = BASE / "RXR_SECONDARY_AUGMENTED_FEATURE_MANIFEST_V1.json"
PRIMARY_AUTH = BASE / "RXR_SECONDARY_AUGMENTED_TRAINING_AUTHORIZATION_V1.json"
WAVES = {
    "scale_v1": BASE / "scale_v1/automatic/multibranch",
    "scale_v2": BASE / "scale_v2/automatic/multibranch",
}
WAVE_LABEL_SOURCES = {
    "scale_v1": "automatic_scale_pseudolabel",
    "scale_v2": "automatic_scale_v2_pseudolabel",
}
OUT = BASE / "scale_v2/model_training"
MANIFEST = BASE / "RXR_SCALE_AUTOMATIC_TRAINING_MANIFEST.json"
AUTHORIZATION = OUT / "RXR_SCALE_AUTOMATIC_TRAINING_AUTHORIZATION.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or ROOT not in path.resolve().parents:
        raise RuntimeError(f"unsafe or missing input: {path}")
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def relocate(record: dict, source: Path, wave: str) -> dict:
    path = (source.parent / record["path"]).resolve()
    if not (
        ROOT in path.parents and path.is_file() and not path.is_symlink()
        and path.stat().st_size == record["bytes"]
        and sha256_file(path) == record["sha256"]
    ):
        raise RuntimeError(f"feature provenance failure: {record['event_id']}")
    return {
        **record,
        "path": os.path.relpath(path, MANIFEST.parent),
        "source_wave": wave,
    }


def main() -> int:
    primary = load(PRIMARY)
    primary_auth = load(PRIMARY_AUTH)
    if not (
        primary.get("schema_version") == "revealnav-mf2-feature-manifest/1"
        and primary.get("metadata", {}).get("training_authorized") is True
        and primary_auth.get("status")
        == "AUGMENTED_DEVELOPMENT_EXPERIMENT_AUTHORIZED"
        and primary_auth.get("training_authorized") is True
        and primary_auth["manifest"]["sha256"] == sha256_file(PRIMARY)
    ):
        raise RuntimeError("primary training authorization failed")

    sources = {
        str(PRIMARY.relative_to(ROOT)): sha256_file(PRIMARY),
        str(PRIMARY_AUTH.relative_to(ROOT)): sha256_file(PRIMARY_AUTH),
    }
    records = [
        relocate(row, PRIMARY, "validated_base")
        for row in primary["records"] if row["split"] in {"train", "development"}
    ]
    excluded_automatic_development = 0
    for wave, directory in WAVES.items():
        feature = directory / "RXR_SCALE_FEATURE_MANIFEST.json"
        gate_path = directory / "RXR_SCALE_FEATURE_GATE.json"
        manifest = load(feature)
        gate = load(gate_path)
        if not (
            manifest.get("schema_version") == "revealnav-mf2-feature-manifest/1"
            and manifest.get("metadata", {}).get("training_authorized") is True
            and manifest.get("metadata", {}).get("label_source")
            == WAVE_LABEL_SOURCES[wave]
            and manifest.get("metadata", {}).get("evaluation_use_authorized") is False
            and gate.get("status") == "FEATURE_GATE_PASS_AUTOMATIC_SCALE_READY"
            and gate.get("training_authorized") is True
            and gate["manifest"]["sha256"] == sha256_file(feature)
        ):
            raise RuntimeError(f"{wave} feature gate failed")
        sources[str(feature.relative_to(ROOT))] = sha256_file(feature)
        sources[str(gate_path.relative_to(ROOT))] = sha256_file(gate_path)
        excluded_automatic_development += sum(
            row["split"] == "development" for row in manifest["records"]
        )
        records.extend(
            relocate(row, feature, wave)
            for row in manifest["records"] if row["split"] == "train"
        )

    ids = [row["event_id"] for row in records]
    train_scenes = {row["scene_id"] for row in records if row["split"] == "train"}
    development = [row for row in records if row["split"] == "development"]
    development_scenes = {row["scene_id"] for row in development}
    counts = Counter(row["split"] for row in records)
    source_counts = Counter(row["source_wave"] for row in records)
    gates = {
        "unique_event_ids": len(ids) == len(set(ids)),
        "train_and_development_present": counts["train"] > 0 and counts["development"] == 68,
        "scene_disjoint_train_development": not (train_scenes & development_scenes),
        "development_is_unchanged_human_audited_only": all(
            row.get("source_wave") == "validated_base"
            and row.get("label_source") == "primary_human_audited"
            for row in development
        ),
        "automatic_scale_records_are_train_only": all(
            row["split"] == "train"
            for row in records if row["source_wave"] in WAVES
        ),
        "no_gold_records": all(row["split"] != "gold" for row in records),
    }
    if not all(gates.values()):
        raise RuntimeError("automatic-scale training population gate failed")

    manifest = {
        "schema_version": "revealnav-mf2-feature-manifest/1",
        "records": records,
        "metadata": {
            "training_authorized": True,
            "protocol": "validated_base_plus_strict_automatic_scale_train_only/1",
            "development_source": "unchanged_68_human_audited_scene_heldout_events",
            "automatic_development_records_excluded": excluded_automatic_development,
            "gold_records_included": False,
            "gold_feature_payload_read": False,
            "normalized_budgets": [1.5, 2.0, 3.0, 4.0],
            "paper_result": False,
        },
    }
    atomic_json(MANIFEST, manifest)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from revealnav_mf2 import RevealFeatureDataset
    train = RevealFeatureDataset(MANIFEST, "train")
    heldout = RevealFeatureDataset(MANIFEST, "development")
    gates.update({
        "production_reader_loads_train": len(train) == counts["train"],
        "production_reader_loads_development": len(heldout) == 68,
    })
    authorization = {
        "schema_version": "revealnav-mf2-scale-training-authorization/1",
        "status": "AUTOMATIC_SCALE_TRAINING_AUTHORIZED" if all(gates.values()) else "AUTOMATIC_SCALE_TRAINING_BLOCKED",
        "training_authorized": all(gates.values()),
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(MANIFEST),
        },
        "sources": sources,
        "counts": {
            "by_split": dict(sorted(counts.items())),
            "by_source_wave": dict(sorted(source_counts.items())),
            "excluded_automatic_development": excluded_automatic_development,
        },
        "gates": gates,
        "gold_feature_payload_read": False,
        "paper_result": False,
    }
    atomic_json(AUTHORIZATION, authorization)
    print(json.dumps({
        "status": authorization["status"],
        "counts": authorization["counts"],
        "manifest": str(MANIFEST.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
