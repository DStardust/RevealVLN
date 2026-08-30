#!/usr/bin/env python3
"""Correct completed MF3 online shards with non-current native indices."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/phase1/mf3b_uad_online/dataset_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    repaired = []
    for summary_path in sorted((DATA / "runs").glob("ep_*/RUN_SUMMARY.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "SHADOW_PASS" or not summary.get(
            "online_feature"
        ):
            continue
        feature_path = ROOT / summary["online_feature"]["path"]
        with np.load(feature_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        native = arrays["native_index"].astype(np.int64, copy=True)
        mask = arrays["candidate_mask"]
        valid = native >= 0
        invalid = valid & ~mask[
            np.arange(native.shape[0]), native.clip(min=0)
        ]
        count = int(invalid.sum())
        if not count:
            continue
        old_sha256 = sha256_file(feature_path)
        native[invalid] = -1
        arrays["native_index"] = native
        part = feature_path.with_name(feature_path.name + ".part")
        with part.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(part, feature_path)
        new_sha256 = sha256_file(feature_path)
        summary["online_feature"].update({
            "bytes": feature_path.stat().st_size,
            "sha256": new_sha256,
            "native_index_current_local_correction": {
                "corrected_rows": count,
                "rule": "set native_index=-1 when its slot is absent from current candidate_mask",
                "source_embeddings_or_teacher_labels_changed": False,
                "old_sha256": old_sha256,
            },
        })
        part_summary = summary_path.with_name(summary_path.name + ".part")
        part_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(part_summary, summary_path)
        repaired.append({
            "episode_id": summary["episode_id"],
            "corrected_rows": count,
            "old_sha256": old_sha256,
            "new_sha256": new_sha256,
        })
    report = {
        "schema_version": "revealnav-mf3b-native-index-correction/1",
        "status": "PASS",
        "reason": (
            "native global argmax may be a previously observed non-current ghost; "
            "such rows must delegate and therefore carry native_index=-1"
        ),
        "repaired_shards": len(repaired),
        "corrected_rows": sum(row["corrected_rows"] for row in repaired),
        "records": repaired,
        "source_embeddings_or_teacher_labels_changed": False,
    }
    path = DATA / "MF3B_NATIVE_INDEX_CORRECTION.json"
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)
    print(json.dumps({
        "status": report["status"],
        "repaired_shards": report["repaired_shards"],
        "corrected_rows": report["corrected_rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
