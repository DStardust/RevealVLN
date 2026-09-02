#!/usr/bin/env python3
"""Select a scene-balanced MF3ZV review cohort from causal-observation-supported rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from revealnav_mf3.progress_language_filter import AtomProposal, deterministic_scene_round_robin
from revealnav_mf3.progress_schema import ProgressAtom, reject_forbidden_progress_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--causal-events", type=Path, required=True)
    parser.add_argument("--rxr-decisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _proposal(row: dict) -> AtomProposal:
    atom = ProgressAtom(**row["atom"])
    return AtomProposal(
        dataset=row["dataset"],
        episode_id=row["episode_id"],
        scene_id=row["scene_id"],
        instruction=row["instruction"],
        language=row.get("language"),
        atom=atom,
        span_start=int(row["span_start"]),
        span_end=int(row["span_end"]),
        mechanical_review_status=row["mechanical_review_status"],
        mechanical_reason=row["mechanical_reason"],
    )


def main() -> int:
    args = parse_args()
    discovery = json.loads(args.discovery.read_text())
    if discovery.get("outcome_payload_read") is not False:
        raise ValueError("language discovery was not outcome blind")
    available: dict[tuple[str, str], dict] = {}
    with args.causal_events.open() as handle:
        for line in handle:
            row = json.loads(line)
            reject_forbidden_progress_payload(row)
            key = (str(row["dataset"]), str(row["episode_id"]))
            normalized = dict(row)
            normalized["causal_support_kind"] = "VISUAL_CAUSAL_PREFIX"
            normalized["support_quality"] = 2
            prior = available.get(key)
            order = (2, int(row["prefix_end"]), str(row["event_id"]))
            if prior is None or order > (
                int(prior["support_quality"]),
                int(prior["prefix_end"]),
                str(prior["event_id"]),
            ):
                available[key] = normalized
    if args.rxr_decisions is not None:
        with args.rxr_decisions.open() as handle:
            for line in handle:
                raw = json.loads(line)
                safe = {
                    "dataset": raw["dataset"],
                    "episode_id": str(raw["episode_id"]),
                    "scene_id": raw["scene_id"],
                    "event_id": raw["event_id"],
                    "prefix_start": 0,
                    "prefix_end": int(raw["decision_step"]),
                    "causal_prefix_sha256": raw["source_native_trace_sha256"],
                    "observation_path": raw["source_native_trace_path"],
                    "current_panorama_path": None,
                    "source_image_inventory": {},
                    "trigger_types": ["EXACT_CAUSAL_DECISION_TRACE"],
                    "causal_support_kind": "SCALAR_CANDIDATE_TRACE",
                    "support_quality": 1,
                }
                reject_forbidden_progress_payload(safe)
                key = ("RxR", safe["episode_id"])
                prior = available.get(key)
                order = (1, safe["prefix_end"], safe["event_id"])
                if prior is None or order > (
                    int(prior["support_quality"]),
                    int(prior["prefix_end"]),
                    str(prior["event_id"]),
                ):
                    available[key] = safe
    eligible = []
    for row in discovery["candidates"]:
        key = (row["dataset"], row["episode_id"])
        if key in available:
            eligible.append(_proposal(row))
    selected = []
    for dataset in ("R2R", "RxR"):
        domain = [row for row in eligible if row.dataset == dataset]
        selected.extend(deterministic_scene_round_robin(domain, 50))
    event_rows = []
    for proposal in selected:
        data = proposal.to_dict()
        event = available[(proposal.dataset, proposal.episode_id)]
        data["causal_support"] = {
            "event_id": event["event_id"],
            "prefix_start": int(event["prefix_start"]),
            "prefix_end": int(event["prefix_end"]),
            "causal_prefix_sha256": event["causal_prefix_sha256"],
            "observation_path": event["observation_path"],
            "current_panorama_path": event["current_panorama_path"],
            "source_image_inventory": event["source_image_inventory"],
            "trigger_types": event.get("trigger_types", []),
            "causal_support_kind": event["causal_support_kind"],
        }
        event_rows.append(data)
    payload = {
        "schema_version": "revealnav-mf3zv-review-selection/1",
        "revision": "mf3zv_minimal_progress_support_v1",
        "status": "OUTCOME_BLIND_REVIEW_SELECTION_COMPLETE",
        "selection_rule": (
            "mechanical lexical candidates with pre-existing strictly causal observation support; "
            "deterministic scene round-robin; maximum 50 per domain"
        ),
        "selection_inputs": [
            "dataset",
            "episode_id",
            "raw_scene_id",
            "instruction_text",
            "causal_observation_availability",
        ],
        "selection_forbidden_inputs": [
            "success",
            "reward",
            "SPL",
            "nDTW",
            "SDTW",
            "utility",
            "delta_utility",
            "model_failure",
        ],
        "outcome_payload_read": False,
        "eligible_with_causal_observations": len(eligible),
        "causal_support_preference": ["VISUAL_CAUSAL_PREFIX", "SCALAR_CANDIDATE_TRACE"],
        "reviewed_count": len(event_rows),
        "domain_counts": dict(Counter(row["dataset"] for row in event_rows)),
        "scene_count": len({row["scene_id"] for row in event_rows}),
        "events": event_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
