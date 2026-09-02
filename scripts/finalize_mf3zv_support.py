#!/usr/bin/env python3
"""Finalize the immutable MF3ZV support-only result."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from revealnav_mf3.mf3zv_protocol import eligible_domains, final_status, validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--atom-audit", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    discovery = json.loads(args.discovery.read_text())
    atom = json.loads(args.atom_audit.read_text())
    state = json.loads(args.state_audit.read_text()) if args.state_audit and args.state_audit.exists() else None
    targets = _rows(args.targets)
    if atom["status"] != "MF3ZV_ATOM_GATE_PASS":
        status = "MF3ZV_PROGRESS_SUPPORT_FAIL"
        first_failure = "MF3ZV_PROGRESS_ATOM_SUPPORT_FAIL"
        domains = []
    elif state is None or state["status"] != "MF3ZV_STATE_GATE_PASS":
        status = "MF3ZV_PROGRESS_SUPPORT_FAIL"
        first_failure = "MF3ZV_PROGRESS_STATE_SUPPORT_FAIL"
        domains = []
    else:
        domains = eligible_domains(targets)
        status = final_status(domains)
        first_failure = None if domains else "MF3ZV_LOCAL_TARGET_SUPPORT_FAIL"
    target_counts = Counter(row["dataset"] for row in targets)
    target_scenes = {
        dataset: len({row["scene_id"] for row in targets if row["dataset"] == dataset})
        for dataset in ("R2R", "RxR")
    }
    audit = {
        "schema_version": "revealnav-mf3zv-progress-support-audit/1",
        "revision": "mf3zv_minimal_progress_support_v1",
        "language": {
            "candidate_instruction_count": discovery["candidate_instruction_count"],
            "family_episode_counts": discovery["family_episode_counts"],
        },
        "atom": atom,
        "state": state,
        "local_target": {
            "supported_total": len(targets),
            "domain_counts": dict(target_counts),
            "domain_scene_counts": target_scenes,
            "eligible_domains": domains,
        },
        "first_failure": first_failure,
        "outcome_payload_read": False,
    }
    result = {
        "schema_version": "revealnav-mf3zv-progress-support-result/1",
        "revision": "mf3zv_minimal_progress_support_v1",
        "status": status,
        "first_failure": first_failure,
        "training_run": False,
        "navigation_run": False,
        "checkpoint_generated": False,
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "outcome_payload_read": False,
        "eligible_future_probe_domains": domains,
        "mf3zw_allowed": bool(domains),
        "progress_memory_direction_stopped": not bool(domains),
    }
    if args.audit.exists() or args.result.exists():
        raise FileExistsError("refusing to overwrite an MF3ZV final artifact")
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

