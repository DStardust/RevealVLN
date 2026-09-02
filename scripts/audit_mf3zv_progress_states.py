#!/usr/bin/env python3
"""Audit Q2 from explicit causal visual-review records; UNKNOWN is never guessed."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from revealnav_mf3.mf3zv_protocol import state_gate, validate_protocol
from revealnav_mf3.progress_state_audit import transition_from_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--atom-audit", type=Path, required=True)
    parser.add_argument("--atoms", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> int:
    args = parse_args()
    validate_protocol(json.loads(args.protocol.read_text()))
    atom_audit = json.loads(args.atom_audit.read_text())
    if atom_audit["status"] != "MF3ZV_ATOM_GATE_PASS":
        raise ValueError("Q2 is forbidden because Q1 did not pass")
    valid_atoms = {
        (row["dataset"], row["episode_id"]): row
        for row in _jsonl(args.atoms)
        if row["review_status"] == "VALID_PROGRESS_ATOM"
    }
    reviews = _jsonl(args.reviews)
    if {(row["dataset"], row["episode_id"]) for row in reviews} != set(valid_atoms):
        raise ValueError("state review must cover every and only Q1-valid episode")
    transitions = []
    unknown_reasons = Counter()
    maximum_possible_supported = sum(
        bool(row.get("potential_transition_window")) for row in reviews
    )
    reviews_executed = sum(bool(row.get("review_executed")) for row in reviews)
    for row in reviews:
        if row.get("status") == "PROGRESS_STATE_SUPPORTED":
            transitions.append(transition_from_review(row, args.root.resolve()))
        elif row.get("status") == "UNKNOWN":
            unknown_reasons[str(row.get("reason", "UNSPECIFIED"))] += 1
        else:
            raise ValueError("state review status must be PROGRESS_STATE_SUPPORTED or UNKNOWN")
    payload = "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in transitions)
    args.transitions.write_text(payload)
    scenes = len({item.scene_id for item in transitions})
    passed = state_gate(len(transitions), len(valid_atoms), scenes)
    families = Counter(valid_atoms[(item.dataset, item.episode_id)]["atom"]["family"] for item in transitions)
    audit = {
        "schema_version": "revealnav-mf3zv-state-audit/1",
        "revision": "mf3zv_minimal_progress_support_v1",
        "status": "MF3ZV_STATE_GATE_PASS" if passed else "MF3ZV_PROGRESS_STATE_SUPPORT_FAIL",
        "valid_progress_atoms": len(valid_atoms),
        "progress_state_supported": len(transitions),
        "state_coverage": len(transitions) / len(valid_atoms) if valid_atoms else 0.0,
        "state_scene_count": scenes,
        "supported_family_counts": dict(families),
        "unknown_count": len(valid_atoms) - len(transitions),
        "unknown_reason_counts": dict(unknown_reasons),
        "reviewed_transition_windows": reviews_executed,
        "maximum_possible_supported_from_available_causal_windows": maximum_possible_supported,
        "maximum_possible_state_coverage": (
            maximum_possible_supported / len(valid_atoms) if valid_atoms else 0.0
        ),
        "signal_status": (
            "FAIL_AT_CAUSAL_WINDOW_SUPPORT_UPPER_BOUND"
            if maximum_possible_supported < 40
            or (valid_atoms and maximum_possible_supported / len(valid_atoms) < 0.70)
            else "SCIENTIFIC_STATE_REVIEW_EXECUTED"
        ),
        "outcome_payload_read": False,
        "training_run": False,
        "navigation_run": False,
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
