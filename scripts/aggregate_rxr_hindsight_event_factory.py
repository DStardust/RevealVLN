#!/usr/bin/env python3
"""Aggregate the sealed 2,303-route hindsight factory into a blind shortlist."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import aggregate_phase0c_cr5_hindsight_preflight as merge
import run_rxr_hindsight_event_factory as factory


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
QUEUE = BASE / "RXR_TRAIN_UNBIASED_EXPANSION_QUEUE.json"
RUN_DIR = BASE / "hindsight_factory/runs"
OUT = BASE / "hindsight_factory/RXR_HINDSIGHT_EVENT_CANDIDATES.json"
EXPECTED_QUEUE = (
    "7b3578afae71dc35327c9ad31b4a97df1a3ccd4960109a2e1fd78f4fa4facbab")
SHARDS = 28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if (not QUEUE.is_file() or QUEUE.is_symlink()
            or sha256_file(QUEUE) != EXPECTED_QUEUE):
        raise SystemExit("frozen expansion queue drift")
    queue = json.loads(QUEUE.read_text())
    queue_rows = {row["expansion_order"]: row
                  for row in queue["candidates"]}
    accepted = {}
    run_sources = []
    for index in range(SHARDS):
        path = RUN_DIR / ("shard_%02d.json" % index)
        if not path.is_file() or path.is_symlink():
            raise SystemExit("missing shard summary: " + str(path))
        run = json.loads(path.read_text())
        expected_orders = {order for order in queue_rows
                           if order % SHARDS == index}
        observed_orders = {row["expansion_order"] for row in run["results"]}
        if (run["status"] not in {
                    "PASS", "PASS_WITH_FAIL_CLOSED_INPUT_FAILURES"}
                or run["shard_index"] != index
                or run["shard_count"] != SHARDS
                or run["enable_thinking"] is not False
                or run["reasoning_effort"] != "none"
                or observed_orders != expected_orders):
            raise SystemExit("shard summary contract failure: " + str(index))
        for row in run["results"]:
            result_path = ROOT / row["path"]
            if (not result_path.is_file() or result_path.is_symlink()
                    or sha256_file(result_path) != row["sha256"]
                    or row["status"] not in {
                        "VALID_MLLM_PROPOSAL", "FACTORY_INPUT_FAILURE"}):
                raise SystemExit("accepted result drift")
            accepted[row["expansion_order"]] = (result_path, row["sha256"])
        run_sources.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "job_count": run["job_count"],
        })
    if set(accepted) != set(queue_rows) or len(accepted) != 2303:
        raise SystemExit("2,303-result exact closure failure")

    candidates = []
    plans = []
    usage = Counter()
    assessment_counts = Counter()
    input_failures = []
    for order in range(2303):
        queue_row = queue_rows[order]
        path, path_sha = accepted[order]
        result = json.loads(path.read_text())
        if result["status"] == "FACTORY_INPUT_FAILURE":
            input_failures.append({
                "expansion_order": order,
                "episode_id": queue_row["episode_id"],
                "scene_id": queue_row["scene_id"],
                "failure_stage": result["failure_stage"],
                "error_type": result["error_type"],
                "error": result["error"],
                "path": str(path.relative_to(ROOT)),
                "sha256": path_sha,
                "replacement_sample_created": False,
            })
            plans.append({
                "expansion_order": order,
                "episode_id": queue_row["episode_id"],
                "automatic_candidate_count": 0,
                "eligible_candidate_count": 0,
                "primary_candidate_id": None,
                "secondary_candidate_id": None,
                "selection_used_geometry_or_human_labels": False,
                "processing_status": "FACTORY_INPUT_FAILURE_NO_REPLACEMENT",
            })
            continue
        request = result["request_evidence"]
        record = {
            "trajectory_id": request["trajectory_id"],
            "timeline_frame_ids": request["timeline_frame_ids"],
            "deterministic_segments": request["deterministic_segments"],
        }
        proposal = result["normalized_proposal"]
        if (result["status"] != "VALID_MLLM_PROPOSAL"
                or result["provider_model"] != "qwen3.8-max"
                or result["enable_thinking"] is not False
                or result["reasoning_effort"] != "none"
                or factory.validate(proposal, record)):
            raise SystemExit("invalid accepted result: " + str(order))
        assessment_counts[proposal["trajectory_assessment"]] += 1
        for key, value in result.get("usage", {}).items():
            if isinstance(value, int):
                usage[key] += value
        raw = []
        for row in proposal["candidate_intervals"]:
            raw.append({
                "proposal": row,
                "start": merge.prefix(row["start_frame_id"]),
                "center": merge.prefix(row["center_frame_id"]),
                "end": merge.prefix(row["end_frame_id"]),
            })
        groups = merge.components(sorted(
            raw, key=lambda row: (row["start"], row["center"])))
        episode_candidates = []
        for local_index, group in enumerate(groups, 1):
            representative = max(group, key=lambda row: (
                float(row["proposal"]["confidence"]),
                merge.KIND_PRIORITY[row["proposal"]["candidate_kind"]],
                -abs(row["end"] - row["start"]), -row["center"]))
            kinds = sorted({row["proposal"]["candidate_kind"] for row in group},
                           key=lambda value: -merge.KIND_PRIORITY[value])
            event_id = "x%04d_ep%s_hv%02d" % (
                order, queue_row["episode_id"], local_index)
            candidate = {
                "hindsight_candidate_id": event_id,
                "expansion_order": order,
                "episode_id": queue_row["episode_id"],
                "trajectory_id": queue_row["trajectory_id"],
                "scene_id": queue_row["scene_id"],
                "instruction_id": queue_row["instruction_id"],
                "interval": {
                    "start_frame_id": "P%04d" % min(
                        row["start"] for row in group),
                    "representative_center_frame_id":
                        representative["proposal"]["center_frame_id"],
                    "end_frame_id": "P%04d" % max(
                        row["end"] for row in group),
                    "supporting_frame_ids": merge.ordered_ids(
                        item for row in group for item in
                        row["proposal"]["supporting_frame_ids"]),
                },
                "candidate_kind": kinds[0],
                "candidate_kind_votes": kinds,
                "conflicting_kind_votes": (
                    "LIKELY_NO_CHOICE_HARD_NEGATIVE" in kinds
                    and any(value != "LIKELY_NO_CHOICE_HARD_NEGATIVE"
                            for value in kinds)),
                "scene_pattern_votes": sorted({
                    row["proposal"]["scene_pattern"] for row in group}),
                "reveal_clause_ids": merge.ordered_ids(
                    item for row in group for item in
                    row["proposal"]["reveal_clause_ids"]),
                "action_clause_ids": merge.ordered_ids(
                    item for row in group for item in
                    row["proposal"]["action_clause_ids"]),
                "representative_summary": representative["proposal"][
                    "reference_route_choice_summary"],
                "representative_rationale": representative["proposal"][
                    "rationale"],
                "representative_confidence": representative["proposal"][
                    "confidence"],
                "source_count": len(group),
                "source": {
                    "proposal_path": str(path.relative_to(ROOT)),
                    "proposal_sha256": path_sha,
                    "proposal_ids": sorted(
                        row["proposal"]["proposal_id"] for row in group),
                },
                "geometry_verified": False,
                "online_causality_verified": False,
                "human_reviewed": False,
                "training_label": False,
            }
            candidates.append(candidate)
            episode_candidates.append(candidate)
        eligible = [row for row in episode_candidates
                    if row["candidate_kind"] !=
                    "LIKELY_NO_CHOICE_HARD_NEGATIVE"
                    and not row["conflicting_kind_votes"]]
        eligible.sort(key=lambda row: (
            -merge.KIND_PRIORITY[row["candidate_kind"]],
            -float(row["representative_confidence"]),
            -int(row["source_count"]),
            merge.prefix(row["interval"]["end_frame_id"])
            - merge.prefix(row["interval"]["start_frame_id"]),
            merge.prefix(row["interval"]["representative_center_frame_id"])))
        plans.append({
            "expansion_order": order,
            "episode_id": queue_row["episode_id"],
            "automatic_candidate_count": len(episode_candidates),
            "eligible_candidate_count": len(eligible),
            "primary_candidate_id": eligible[0]["hindsight_candidate_id"]
                if eligible else None,
            "secondary_candidate_id": eligible[1]["hindsight_candidate_id"]
                if len(eligible) > 1 else None,
            "selection_used_geometry_or_human_labels": False,
            "processing_status": "PENDING_MULTIVIEW_AND_3D_GATES"
                if eligible else "NO_ELIGIBLE_HINDSIGHT_CANDIDATE",
        })

    primary_count = sum(row["primary_candidate_id"] is not None
                        for row in plans)
    output = {
        "manifest": "RevealNav RxR expansion hindsight event candidates",
        "revision": "rxr-hindsight-event-candidates/1",
        "status": "PASS_PENDING_MULTIVIEW_AND_3D_GATES",
        "sources": {
            "queue": {"path": str(QUEUE.relative_to(ROOT)),
                      "sha256": EXPECTED_QUEUE},
            "shards": run_sources,
        },
        "requested_and_provider_model": "qwen3.8-max",
        "enable_thinking": False,
        "reasoning_effort": "none",
        "usage": dict(sorted(usage.items())),
        "trajectory_assessment_counts": dict(sorted(
            assessment_counts.items())),
        "fail_closed_input_failure_count": len(input_failures),
        "fail_closed_input_failures": input_failures,
        "replacement_samples_created": 0,
        "merge_rule": {
            "same_episode": True,
            "strict_positive_prefix_interval_overlap": True,
            "shared_boundary_only_does_not_merge": True,
            "semantic_geometry_or_human_labels_used": False,
        },
        "shortlist_rule": {
            "exclude_hard_negative_representative": True,
            "exclude_conflicting_kind_votes": True,
            "then_kind_confidence_source_count_compactness_time": True,
            "maximum_initial_card_per_trajectory": 1,
            "secondary_is_conditional_on_primary_rejection": True,
        },
        "trajectory_count": len(plans),
        "raw_candidate_count": sum(
            row["automatic_candidate_count"] for row in plans),
        "merged_candidate_count": len(candidates),
        "trajectory_with_primary_count": primary_count,
        "trajectory_without_primary_count": len(plans) - primary_count,
        "conditional_secondary_count": sum(
            row["secondary_candidate_id"] is not None for row in plans),
        "candidates": candidates,
        "cascade_review_plan": plans,
        "future_frames_are_offline_annotation_only": True,
        "online_causal_labels_created": 0,
        "geometry_verified_candidates": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "trajectories": output["trajectory_count"],
        "merged_candidates": output["merged_candidate_count"],
        "with_primary": output["trajectory_with_primary_count"],
        "without_primary": output["trajectory_without_primary_count"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
