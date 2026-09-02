#!/usr/bin/env python3
"""Apply the fixed Q1 atom audit and its pre-registered gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from revealnav_mf3.mf3zv_protocol import atom_gate, validate_protocol
from revealnav_mf3.progress_schema import AtomReviewStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    selection = json.loads(args.selection.read_text())
    rows = []
    for selected in selection["events"]:
        status = selected["mechanical_review_status"]
        if status not in {item.value for item in AtomReviewStatus}:
            raise ValueError(f"unknown atom review status: {status}")
        rows.append(
            {
                "dataset": selected["dataset"],
                "episode_id": selected["episode_id"],
                "scene_id": selected["scene_id"],
                "instruction": selected["instruction"],
                "atom": selected["atom"],
                "review_status": status,
                "review_reason": selected["mechanical_reason"],
                "review_source": "AI_ASSISTED_FIXED_RULE_REVIEW_NOT_HUMAN_GOLD",
                "outcome_payload_read": False,
            }
        )
    valid = sum(row["review_status"] == "VALID_PROGRESS_ATOM" for row in rows)
    reviewed = len(rows)
    passed = atom_gate(valid, reviewed)
    counts = Counter(row["review_status"] for row in rows)
    family_counts = Counter(row["atom"]["family"] for row in rows if row["review_status"] == "VALID_PROGRESS_ATOM")
    args.atoms.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    audit = {
        "schema_version": "revealnav-mf3zv-atom-audit/1",
        "revision": "mf3zv_minimal_progress_support_v1",
        "status": "MF3ZV_ATOM_GATE_PASS" if passed else "MF3ZV_PROGRESS_ATOM_SUPPORT_FAIL",
        "reviewed_count": reviewed,
        "valid_progress_atoms": valid,
        "atom_coverage": valid / reviewed if reviewed else 0.0,
        "review_status_counts": dict(counts),
        "valid_family_counts": dict(family_counts),
        "valid_scene_count": len({row["scene_id"] for row in rows if row["review_status"] == "VALID_PROGRESS_ATOM"}),
        "outcome_payload_read": False,
        "training_run": False,
        "navigation_run": False,
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

