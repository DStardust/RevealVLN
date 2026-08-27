#!/usr/bin/env python3
"""Validate the self-contained blank three-lane new-Gold review package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
OUT = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1/new_gold/review_package"
MANIFEST = OUT / "RXR_NEW_GOLD_REVIEW_MANIFEST.json"
REPORT = OUT / "RXR_NEW_GOLD_REVIEW_PACKAGE_ACCEPTANCE.json"
LANES = ("R1", "R2", "R3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures = []
    manifest = json.loads(MANIFEST.read_text())
    items = manifest.get("items", [])
    ids = [row.get("event_id") for row in items]
    if not (
        manifest.get("status") == "PENDING_THREE_INDEPENDENT_HUMAN_REVIEWS"
        and len(items) == 900
        and len(ids) == len(set(ids))
        and manifest.get("human_labels_created") == 0
        and manifest.get("old_gold_payload_read") is False
        and manifest.get("gold_authorized") is False
    ):
        failures.append("manifest contract")
    media_count = 0
    media_bytes = 0
    for item in items:
        if len(item.get("panoramas", [])) != 3 or not item.get("context"):
            failures.append("media cardinality: " + str(item.get("event_id")))
            continue
        for record in item["panoramas"] + item["context"]:
            path = OUT / record["path"]
            if not (
                path.is_file()
                and not path.is_symlink()
                and OUT.resolve() in path.resolve().parents
                and path.stat().st_size == record["bytes"]
                and sha256_file(path) == record["sha256"]
            ):
                failures.append("media integrity: " + str(path))
            media_count += 1
            media_bytes += record["bytes"]
    for lane in LANES:
        lane_record = manifest["review_lanes"][lane]
        template = ROOT / lane_record["template"]
        reviewer = ROOT / lane_record["reviewer"]
        rows = [json.loads(line) for line in template.read_text().splitlines() if line.strip()]
        if not (
            sha256_file(template) == lane_record["template_sha256"]
            and sha256_file(reviewer) == lane_record["reviewer_sha256"]
            and len(rows) == 900
            and {row["event_id"] for row in rows} == set(ids)
            and all(
                row["review_lane"] == lane
                and row["reviewer_id"] is None
                and row["reviewer_type"] == "HUMAN"
                and row["event_valid"] is None
                for row in rows
            )
        ):
            failures.append("blank lane contract: " + lane)
        html = reviewer.read_text()
        if "/mnt/" in html or "file://" in html:
            failures.append("nonportable reviewer: " + lane)
    if media_count != manifest.get("media_files") or media_bytes != manifest.get("logical_media_bytes"):
        failures.append("media aggregate")
    parts = list(OUT.rglob("*.part"))
    if parts:
        failures.append("stale part files")
    report = {
        "schema_version": "revealnav-new-gold-review-package-acceptance/1",
        "status": "PACKAGE_PASS_PENDING_THREE_HUMAN_REVIEWS" if not failures else "PACKAGE_FAIL",
        "failures": failures,
        "counts": {
            "review_candidates": len(items),
            "review_lanes": 3,
            "blank_review_rows": len(items) * 3,
            "media_files": media_count,
            "logical_media_bytes": media_bytes,
        },
        "manifest_sha256": sha256_file(MANIFEST),
        "human_labels_created": 0,
        "three_reviewer_agreement_measured": False,
        "gold_authorized": False,
    }
    part = REPORT.with_name(REPORT.name + ".part")
    part.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(part, REPORT)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

