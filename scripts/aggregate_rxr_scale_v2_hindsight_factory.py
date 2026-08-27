#!/usr/bin/env python3
"""Aggregate the frozen scale-v2 route census into event candidates."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path("/mnt/daiyang/vla/scripts")))
import aggregate_phase0c_cr5_hindsight_preflight as merge
import run_rxr_scale_v2_hindsight_factory as factory


ROOT = Path("/mnt/daiyang/vla")
QUEUE = factory.QUEUE
RUN_DIR = factory.base.RUN_DIR
OUT = ROOT / (
    "artifacts/phase1/rxr_train_expansion/scale_v2/hindsight_factory/"
    "RXR_SCALE_V2_HINDSIGHT_EVENT_CANDIDATES.json"
)
SHARDS = 24
COUNT = factory.COUNT


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def main() -> int:
    if (
        not QUEUE.is_file()
        or QUEUE.is_symlink()
        or factory.base.sha256_file(QUEUE) != factory.EXPECTED[QUEUE]
    ):
        raise RuntimeError("scale-v2 route census drift")
    queue = json.loads(QUEUE.read_text())
    queue_rows = {row["expansion_order"]: row for row in queue["candidates"]}
    if len(queue_rows) != COUNT:
        raise RuntimeError("scale-v2 route count drift")

    accepted = {}
    run_sources = []
    for shard_index in range(SHARDS):
        path = RUN_DIR / f"shard_{shard_index:02d}.json"
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("missing scale-v2 shard: " + str(path))
        run = json.loads(path.read_text())
        expected_orders = {
            row["expansion_order"]
            for row in queue_rows.values()
            if row["scale_v2_order"] % SHARDS == shard_index
        }
        observed_orders = {row["expansion_order"] for row in run["results"]}
        if not (
            run.get("status") in {"PASS", "PASS_WITH_FAIL_CLOSED_FAILURES"}
            and run.get("queue_sha256") == factory.EXPECTED[QUEUE]
            and run.get("selection_commitment_sha256") == factory.COMMITMENT
            and run.get("shard_index") == shard_index
            and run.get("shard_count") == SHARDS
            and run.get("enable_thinking") is False
            and run.get("reasoning_effort") == "none"
            and observed_orders == expected_orders
        ):
            raise RuntimeError(f"scale-v2 shard contract failure: {shard_index}")
        for row in run["results"]:
            result_path = ROOT / row["path"]
            if (
                not result_path.is_file()
                or result_path.is_symlink()
                or factory.base.sha256_file(result_path) != row["sha256"]
                or row["status"]
                not in {
                    "VALID_MLLM_PROPOSAL",
                    "FACTORY_INPUT_FAILURE",
                    "INVALID_MLLM_PROPOSAL",
                    "REQUEST_OR_VALIDATION_FAILURE",
                }
            ):
                raise RuntimeError("scale-v2 accepted result drift")
            accepted[row["expansion_order"]] = (result_path, row["sha256"])
        run_sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": factory.base.sha256_file(path),
                "job_count": run["job_count"],
            }
        )
    if set(accepted) != set(queue_rows):
        raise RuntimeError("scale-v2 result population is incomplete")

    candidates = []
    plans = []
    input_failures = []
    assessment_counts = Counter()
    usage = Counter()
    for expansion_order in sorted(queue_rows):
        queue_row = queue_rows[expansion_order]
        result_path, result_sha = accepted[expansion_order]
        result = json.loads(result_path.read_text())
        if result["status"] != "VALID_MLLM_PROPOSAL":
            input_failures.append(
                {
                    "scale_v2_order": queue_row["scale_v2_order"],
                    "expansion_order": expansion_order,
                    "episode_id": queue_row["episode_id"],
                    "scene_id": queue_row["scene_id"],
                    "failure_stage": (
                        result.get("failure_stage")
                        or "HINDSIGHT_PROVIDER_OR_SCHEMA_FAIL_CLOSED"
                    ),
                    "error_type": result.get("error_type") or result["status"],
                    "error": result.get("error") or "; ".join(
                        result.get("validation_errors", [])
                    ),
                    "path": str(result_path.relative_to(ROOT)),
                    "sha256": result_sha,
                    "replacement_sample_created": False,
                }
            )
            plans.append(
                {
                    "scale_v2_order": queue_row["scale_v2_order"],
                    "expansion_order": expansion_order,
                    "episode_id": queue_row["episode_id"],
                    "automatic_candidate_count": 0,
                    "eligible_candidate_count": 0,
                    "processing_status": "FACTORY_INPUT_FAILURE_NO_REPLACEMENT",
                }
            )
            continue

        request = result["request_evidence"]
        record = {
            "trajectory_id": request["trajectory_id"],
            "timeline_frame_ids": request["timeline_frame_ids"],
            "deterministic_segments": request["deterministic_segments"],
        }
        proposal = result["normalized_proposal"]
        if not (
            result.get("provider_model") == factory.base.MODEL
            and result.get("enable_thinking") is False
            and request.get("selection_commitment_sha256") == factory.COMMITMENT
            and not factory.base.validate(proposal, record)
        ):
            raise RuntimeError("invalid scale-v2 accepted proposal")
        assessment_counts[proposal["trajectory_assessment"]] += 1
        for key, value in (result.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] += value

        raw = [
            {
                "proposal": row,
                "start": merge.prefix(row["start_frame_id"]),
                "center": merge.prefix(row["center_frame_id"]),
                "end": merge.prefix(row["end_frame_id"]),
            }
            for row in proposal["candidate_intervals"]
        ]
        groups = merge.components(
            sorted(raw, key=lambda row: (row["start"], row["center"]))
        )
        episode_candidates = []
        for local_index, group in enumerate(groups, 1):
            representative = max(
                group,
                key=lambda row: (
                    float(row["proposal"]["confidence"]),
                    merge.KIND_PRIORITY[row["proposal"]["candidate_kind"]],
                    -abs(row["end"] - row["start"]),
                    -row["center"],
                ),
            )
            kinds = sorted(
                {row["proposal"]["candidate_kind"] for row in group},
                key=lambda value: -merge.KIND_PRIORITY[value],
            )
            event_id = (
                f"v2x{queue_row['scale_v2_order']:04d}_"
                f"ep{queue_row['episode_id']}_hv{local_index:02d}"
            )
            candidate = {
                "hindsight_candidate_id": event_id,
                "scale_v2_order": queue_row["scale_v2_order"],
                "expansion_order": expansion_order,
                "episode_id": queue_row["episode_id"],
                "trajectory_id": queue_row["trajectory_id"],
                "scene_id": queue_row["scene_id"],
                "scene_split": queue_row["scene_split"],
                "instruction_id": queue_row["instruction_id"],
                "interval": {
                    "start_frame_id": f"P{min(row['start'] for row in group):04d}",
                    "representative_center_frame_id": representative["proposal"][
                        "center_frame_id"
                    ],
                    "end_frame_id": f"P{max(row['end'] for row in group):04d}",
                    "supporting_frame_ids": merge.ordered_ids(
                        item
                        for row in group
                        for item in row["proposal"]["supporting_frame_ids"]
                    ),
                },
                "candidate_kind": kinds[0],
                "candidate_kind_votes": kinds,
                "conflicting_kind_votes": (
                    "LIKELY_NO_CHOICE_HARD_NEGATIVE" in kinds
                    and any(value != "LIKELY_NO_CHOICE_HARD_NEGATIVE" for value in kinds)
                ),
                "scene_pattern_votes": sorted(
                    {row["proposal"]["scene_pattern"] for row in group}
                ),
                "reveal_clause_ids": merge.ordered_ids(
                    item
                    for row in group
                    for item in row["proposal"]["reveal_clause_ids"]
                ),
                "action_clause_ids": merge.ordered_ids(
                    item
                    for row in group
                    for item in row["proposal"]["action_clause_ids"]
                ),
                "representative_summary": representative["proposal"][
                    "reference_route_choice_summary"
                ],
                "representative_rationale": representative["proposal"]["rationale"],
                "representative_confidence": representative["proposal"]["confidence"],
                "source_count": len(group),
                "source": {
                    "proposal_path": str(result_path.relative_to(ROOT)),
                    "proposal_sha256": result_sha,
                    "proposal_ids": sorted(
                        row["proposal"]["proposal_id"] for row in group
                    ),
                },
                "geometry_verified": False,
                "online_causality_verified": False,
                "human_reviewed": False,
                "training_label": False,
            }
            candidates.append(candidate)
            episode_candidates.append(candidate)
        eligible = [
            row
            for row in episode_candidates
            if row["candidate_kind"] != "LIKELY_NO_CHOICE_HARD_NEGATIVE"
            and not row["conflicting_kind_votes"]
        ]
        plans.append(
            {
                "scale_v2_order": queue_row["scale_v2_order"],
                "expansion_order": expansion_order,
                "episode_id": queue_row["episode_id"],
                "automatic_candidate_count": len(episode_candidates),
                "eligible_candidate_count": len(eligible),
                "processing_status": (
                    "PENDING_MULTIVIEW_AND_3D_GATES"
                    if eligible
                    else "NO_ELIGIBLE_HINDSIGHT_CANDIDATE"
                ),
            }
        )

    eligible_count = sum(
        row["candidate_kind"] != "LIKELY_NO_CHOICE_HARD_NEGATIVE"
        and not row["conflicting_kind_votes"]
        for row in candidates
    )
    output = {
        "schema_version": "revealnav-rxr-scale-v2-hindsight-candidates/1",
        "status": "PASS_PENDING_MULTIVIEW_AND_3D_GATES",
        "sources": {
            "queue": {
                "path": str(QUEUE.relative_to(ROOT)),
                "sha256": factory.EXPECTED[QUEUE],
            },
            "shards": run_sources,
        },
        "model": factory.base.MODEL,
        "enable_thinking": False,
        "reasoning_effort": "none",
        "usage": dict(sorted(usage.items())),
        "trajectory_assessment_counts": dict(sorted(assessment_counts.items())),
        "route_count": len(plans),
        "merged_candidate_count": len(candidates),
        "eligible_candidate_count": eligible_count,
        "fail_closed_input_failure_count": len(input_failures),
        "fail_closed_input_failures": input_failures,
        "replacement_samples_created": 0,
        "candidates": candidates,
        "route_plans": plans,
        "future_frames_are_offline_annotation_only": True,
        "old_gold_payload_read": False,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "routes": len(plans),
                "merged_candidates": len(candidates),
                "eligible_candidates": eligible_count,
                "input_failures": len(input_failures),
                "output": str(OUT.relative_to(ROOT)),
                "sha256": factory.base.sha256_file(OUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
