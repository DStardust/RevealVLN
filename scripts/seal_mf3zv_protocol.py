#!/usr/bin/env python3
"""Seal the immutable MF3ZV protocol before support results."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from revealnav_mf3.mf3zv_protocol import (
    GATES,
    PUBLIC_CLOSED,
    REVISION,
    STATUS_SEALED,
    inventory,
    validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--r2r", type=Path, required=True)
    parser.add_argument("--rxr", type=Path, required=True)
    parser.add_argument("--causal-events", type=Path, required=True)
    parser.add_argument("--rxr-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    selection = json.loads(args.selection.read_text())
    if selection.get("outcome_payload_read") is not False:
        raise ValueError("selection must be outcome blind")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    implementation = [
        root / "revealnav_mf3/progress_schema.py",
        root / "revealnav_mf3/progress_language_filter.py",
        root / "revealnav_mf3/progress_state_audit.py",
        root / "revealnav_mf3/progress_target_support.py",
        root / "revealnav_mf3/mf3zv_protocol.py",
    ]
    protocol = {
        "schema_version": "revealnav-mf3zv-progress-support-protocol/1",
        "revision": REVISION,
        "status": STATUS_SEALED,
        "source_commit": source_commit,
        "allowed_datasets": ["R2R", "RxR"],
        "allowed_split": "train-development",
        "progress_families": ["ORDINAL", "PASSED_LANDMARK"],
        "review": {"maximum": 100, "R2R_nominal_maximum": 50, "RxR_nominal_maximum": 50},
        "parser": {
            "fixed_lexical_rules": True,
            "earliest_unambiguous_atom_only": True,
            "no_post_result_parser_change": True,
        },
        "gates": GATES,
        "stage_order": ["LANGUAGE", "ATOM", "STATE", "LOCAL_TARGET", "FINAL_SUPPORT"],
        "source_files": {
            "R2R_train": inventory(args.r2r.resolve(), root),
            "RxR_train_guide": inventory(args.rxr.resolve(), root),
            "causal_event_inventory": inventory(args.causal_events.resolve(), root),
            "RxR_causal_decision_inventory": inventory(args.rxr_decisions.resolve(), root),
            "review_selection": inventory(args.selection.resolve(), root),
        },
        "implementation_inventory": [inventory(path, root) for path in implementation],
        "outcome_payload_read": False,
        "public_split_access": PUBLIC_CLOSED,
        "training_run": False,
        "navigation_run": False,
        "checkpoint_generated": False,
        "qwen_api_calls": 0,
        "review_provenance": "AI_ASSISTED_REVIEW_NOT_HUMAN_GOLD",
    }
    validate_protocol(protocol)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite sealed protocol: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
