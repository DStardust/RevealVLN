#!/usr/bin/env python3
"""Validate first review or score the pre-sealed blind test-retest scout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.single_expert_dec_scout import (  # noqa: E402
    RESULT_PATH,
    atomic_json,
    load_completed_review,
    score_completed_reviews,
    validate_first_review,
    verify_scout_protocol,
)
def final_audit(first_path: Path, retest_path: Path) -> dict[str, object]:
    verify_scout_protocol()
    first = load_completed_review(first_path, mode="first")
    retest = load_completed_review(retest_path, mode="retest")
    first_reviewers = {str(row["reviewer_id"]) for row in first}
    retest_reviewers = {str(row["reviewer_id"]) for row in retest}
    if len(first_reviewers) != 1 or retest_reviewers != first_reviewers:
        raise ScoutError("first and retest must use one stable expert identity")
    result = score_completed_reviews(first, retest)
    result.update({
        "schema_version": "revealnav-mf3zp-single-expert-dec-scout-result/1",
        "revision": "mf3zp_single_expert_dec_scout_v1",
        "single_expert_only": True,
        "human_gold": False,
        "formal_label_validity_pass": False,
    })
    atomic_json(RESULT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-first", "final"))
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--retest", type=Path)
    args = parser.parse_args()
    if args.command == "validate-first":
        result = validate_first_review(args.first)
    else:
        if args.retest is None:
            parser.error("--retest is required for final")
        result = final_audit(args.first, args.retest)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
