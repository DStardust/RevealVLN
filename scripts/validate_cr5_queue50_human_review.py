#!/usr/bin/env python3
"""Validate the uploaded queue50 human review without rewriting it."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/human_review_fast"
MANIFEST = BASE / "CR5_QUEUE50_FAST_REVIEW_MANIFEST.json"
REVIEW = BASE / "daiyang_queue50.jsonl"
OUT = BASE / "CR5_QUEUE50_HUMAN_REVIEW_ACCEPTANCE.json"
FIELDS = (
    "two_distinct_executable_exits",
    "alternative_is_not_incoming_closed_or_duplicate",
    "instruction_uniquely_selects_target",
    "decision_center_and_temporal_order_are_reasonable",
)
KEYS = {
    "reviewer_id", "reviewer_type", "event_id", *FIELDS,
    "final_label", "reason_codes", "comment_zh",
}
LABELS = {"ACCEPT", "REJECT", "AMBIGUOUS"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not REVIEW.is_file() or REVIEW.is_symlink():
        raise SystemExit("review is missing or is a symlink")
    manifest = json.loads(MANIFEST.read_text())
    expected_ids = {row["event_id"] for row in manifest["items"]}
    rows = [json.loads(line) for line in REVIEW.read_text().splitlines()
            if line.strip()]
    if len(rows) != 34 or len(expected_ids) != 34:
        raise SystemExit("review must contain exactly 34 rows")
    if len({row.get("event_id") for row in rows}) != len(rows):
        raise SystemExit("duplicate review event")
    if {row.get("event_id") for row in rows} != expected_ids:
        raise SystemExit("review event closure mismatch")
    for row in rows:
        if set(row) != KEYS:
            raise SystemExit("review keys mismatch: " + row["event_id"])
        if (not isinstance(row["reviewer_id"], str)
                or not row["reviewer_id"].strip()
                or row["reviewer_type"] != "HUMAN"
                or row["final_label"] not in LABELS
                or not isinstance(row["reason_codes"], list)
                or len(row["reason_codes"]) != len(set(row["reason_codes"]))
                or not all(isinstance(value, str)
                           for value in row["reason_codes"])
                or not isinstance(row["comment_zh"], str)
                or any(row[field] not in {True, False, None}
                       for field in FIELDS)):
            raise SystemExit("review value contract: " + row["event_id"])
        values = [row[field] for field in FIELDS]
        if row["final_label"] == "ACCEPT" and (
                values != [True] * 4 or row["reason_codes"]):
            raise SystemExit("ACCEPT consistency: " + row["event_id"])
        if row["final_label"] == "REJECT" and (
                False not in values or not row["reason_codes"]):
            raise SystemExit("REJECT consistency: " + row["event_id"])
        if row["final_label"] == "AMBIGUOUS" and any(
                value is not None for value in values):
            raise SystemExit("AMBIGUOUS consistency: " + row["event_id"])
    counts = Counter(row["final_label"] for row in rows)
    output = {
        "manifest": "MF2-CR5 queue50 human review acceptance",
        "revision": "cr5-queue50-human-review-acceptance/1",
        "status": "PASS",
        "sources": {
            "review_manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(MANIFEST),
            },
            "uploaded_human_review": {
                "path": str(REVIEW.relative_to(ROOT)),
                "sha256": sha256_file(REVIEW),
            },
        },
        "review_count": len(rows),
        "unique_event_count": len(expected_ids),
        "label_counts": dict(sorted(counts.items())),
        "accepted_event_ids": sorted(
            row["event_id"] for row in rows
            if row["final_label"] == "ACCEPT"),
        "rejected_event_ids": sorted(
            row["event_id"] for row in rows
            if row["final_label"] == "REJECT"),
        "ambiguous_event_ids": sorted(
            row["event_id"] for row in rows
            if row["final_label"] == "AMBIGUOUS"),
        "human_labels_created_by_validator": 0,
        "original_review_rewritten": False,
        "causal_gate_authorized_event_count": counts["ACCEPT"],
        "training_authorized": False,
    }
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "counts": output["label_counts"],
        "causal_gate_events": output["causal_gate_authorized_event_count"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
