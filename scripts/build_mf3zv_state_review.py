#!/usr/bin/env python3
"""Materialize the Q2 review ledger and fail closed when its support bound is insufficient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from revealnav_mf3.mf3zv_protocol import GATES, validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_protocol(json.loads(args.protocol.read_text()))
    selection = json.loads(args.selection.read_text())
    valid = {
        (row["dataset"], row["episode_id"])
        for row in (json.loads(line) for line in args.atoms.open() if line.strip())
        if row["review_status"] == "VALID_PROGRESS_ATOM"
    }
    chosen = {
        (row["dataset"], row["episode_id"]): row
        for row in selection["events"]
        if (row["dataset"], row["episode_id"]) in valid
    }
    potential = []
    for key, row in chosen.items():
        support = row["causal_support"]
        has_window = (
            support["causal_support_kind"] == "VISUAL_CAUSAL_PREFIX"
            and int(support["prefix_end"]) >= int(support["prefix_start"]) + 1
        )
        if has_window:
            potential.append(key)
    maximum = len(potential)
    bound_passes = (
        maximum >= GATES["minimum_state_supported_episodes"]
        and maximum / len(valid) >= GATES["state_coverage_minimum"]
    )
    if bound_passes:
        raise RuntimeError(
            "causal-window upper bound reaches Q2; explicit visual review is required instead"
        )
    rows = []
    for key in sorted(valid):
        selected = chosen[key]
        support = selected["causal_support"]
        has_window = key in potential
        if support["causal_support_kind"] != "VISUAL_CAUSAL_PREFIX":
            reason = "NO_CAUSAL_VISUAL_SEMANTIC_EVIDENCE"
        elif not has_window:
            reason = "NO_TWO_PREFIX_CAUSAL_WINDOW"
        else:
            reason = "NOT_REVIEWED_AFTER_Q2_SUPPORT_UPPER_BOUND_FAILURE"
        rows.append(
            {
                "dataset": key[0],
                "episode_id": key[1],
                "scene_id": selected["scene_id"],
                "atom_id": selected["atom"]["atom_id"],
                "status": "UNKNOWN",
                "reason": reason,
                "potential_transition_window": has_window,
                "review_executed": False,
                "outcome_payload_read": False,
            }
        )
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

