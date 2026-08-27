#!/usr/bin/env python3
"""Validate the 16-item human confirmation of queue50 machine rejects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
REVIEW = ROOT / "artifacts/phase0/phase0c_cr5_queue50/human_review_fast"
PACKAGE = REVIEW / "CR5_QUEUE50_AUTO_REJECT_REVIEW_PACKAGE.json"
LEDGER = REVIEW / "CR5_QUEUE50_AUTO_REJECTED.json"
DEFAULT_REVIEW = REVIEW / "daiyang_auto_reject16.jsonl"
OUT = REVIEW / "CR5_QUEUE50_AUTO_REJECT_HUMAN_ACCEPTANCE.json"
EXPECTED_PACKAGE_SHA256 = (
    "468899e8dcbf084a9f353c49c3474ceac44606d28330d8c6c2cf0cd6b8005625"
)
EXPECTED_LEDGER_SHA256 = (
    "14f549c8d0c73628335fa673b433593f7152fb6b6dd8a0abd074134b7c218403"
)
LABELS = {"CONFIRM_REJECT", "SUSPECT_FALSE_REJECT", "AMBIGUOUS"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review", nargs="?", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    review_path = args.review.resolve()
    if (ROOT.resolve() not in review_path.parents or not review_path.is_file()
            or review_path.is_symlink()):
        raise SystemExit("review must be a regular project-local file")
    for path, expected in ((PACKAGE, EXPECTED_PACKAGE_SHA256),
                           (LEDGER, EXPECTED_LEDGER_SHA256)):
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected):
            raise SystemExit("pinned review evidence drift: " + str(path))

    ledger = json.loads(LEDGER.read_text())
    expected = {row["event_id"]: row for row in ledger["events"]}
    failures = []
    rows = []
    for line_number, line in enumerate(
            review_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append("line %d invalid JSON: %s" % (line_number, exc))
            continue
        rows.append(row)
    ids = [row.get("event_id") for row in rows]
    if len(rows) != 16 or len(set(ids)) != 16 or set(ids) != set(expected):
        failures.append("review must contain exactly the 16 expected events")

    for row in rows:
        event_id = row.get("event_id")
        if event_id not in expected:
            continue
        label = row.get("final_label")
        if set(row) != {
                "reviewer_id", "reviewer_type", "event_id",
                "machine_reject_confirmed", "final_label", "reason_codes",
                "comment_zh"}:
            failures.append(event_id + ": schema keys")
            continue
        if (row.get("reviewer_type") != "HUMAN"
                or not isinstance(row.get("reviewer_id"), str)
                or not row["reviewer_id"].strip()):
            failures.append(event_id + ": human reviewer identity")
        if label not in LABELS:
            failures.append(event_id + ": final_label")
        reasons = row.get("reason_codes")
        if (not isinstance(reasons, list)
                or len(reasons) != len(set(reasons))
                or not all(isinstance(value, str) for value in reasons)):
            failures.append(event_id + ": reason_codes")
            continue
        if label == "CONFIRM_REJECT" and (
                row.get("machine_reject_confirmed") is not True
                or set(reasons) != set(expected[event_id][
                    "automatic_reject_reasons"])):
            failures.append(event_id + ": confirmed-reject consistency")
        if label == "SUSPECT_FALSE_REJECT" and (
                row.get("machine_reject_confirmed") is not False
                or reasons != ["SUSPECT_FALSE_REJECT"]):
            failures.append(event_id + ": false-reject consistency")
        if label == "AMBIGUOUS" and (
                row.get("machine_reject_confirmed") is not None
                or reasons != ["INSUFFICIENT_EVIDENCE"]):
            failures.append(event_id + ": ambiguous consistency")
        if not isinstance(row.get("comment_zh"), str):
            failures.append(event_id + ": comment_zh")

    counts = {label: sum(row.get("final_label") == label for row in rows)
              for label in sorted(LABELS)}
    complete = not failures and counts["CONFIRM_REJECT"] == 16
    output = {
        "manifest": "MF2-CR5 queue50 machine-reject human acceptance",
        "revision": "cr5-queue50-auto-reject-human-acceptance/1",
        "status": (
            "PASS_16_MACHINE_REJECTIONS_HUMAN_CONFIRMED"
            if complete else "HOLD_REVIEW_OR_RERUN_REQUIRED"
        ),
        "sources": {
            str(PACKAGE.relative_to(ROOT)): EXPECTED_PACKAGE_SHA256,
            str(LEDGER.relative_to(ROOT)): EXPECTED_LEDGER_SHA256,
            str(review_path.relative_to(ROOT)): sha256_file(review_path),
        },
        "review_count": len(rows),
        "unique_event_count": len(set(ids)),
        "label_counts": counts,
        "suspect_false_reject_event_ids": sorted(
            row["event_id"] for row in rows
            if row.get("final_label") == "SUSPECT_FALSE_REJECT"),
        "ambiguous_event_ids": sorted(
            row["event_id"] for row in rows
            if row.get("final_label") == "AMBIGUOUS"),
        "frozen_50_item_human_protocol_can_be_recomputed": complete,
        "original_review_rewritten": False,
        "training_authorized": False,
        "failures": failures,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "counts": counts,
        "failures": failures,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
