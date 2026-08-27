#!/usr/bin/env python3
"""Merge queue50 hindsight proposals and build a cascade review shortlist."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_phase0c_cr5_hindsight_preflight as base  # noqa: E402
import run_phase0c_cr5_hindsight_locator as validator  # noqa: E402


OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator"
INPUT = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_INPUTS.json"
RUN = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_ACCEPTED_RUN.json"
OUT = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_AGGREGATED.json"
EXPECTED_INPUT_SHA = (
    "8e00000ee306369e305c53d580444e1ac3228a6e94c3c424d84f9db5d16ea151"
)
EXPECTED_RUN_SHA = (
    "7bc014b084324152083dde4467fb7f849455c830a855cd6826cca35ff5b00692"
)


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)
                         + "\n")
    os.replace(temporary, path)


def main():
    if (base.sha256_file(INPUT) != EXPECTED_INPUT_SHA
            or base.sha256_file(RUN) != EXPECTED_RUN_SHA):
        raise SystemExit("queue50 aggregate source drift")
    manifest = json.loads(INPUT.read_text())
    run = json.loads(RUN.read_text())
    if (run.get("status") != "PASS_WITH_FAIL_CLOSED_BLOCK_REJECTIONS"
            or run.get("valid_count") != 136
            or run.get("fail_closed_rejected_block_count") != 2):
        raise SystemExit("accepted run contract failed")
    episode_meta = {row["episode_id"]: row
                    for row in manifest["episodes"]}
    chunk_meta = {(row["episode_id"], chunk["chunk_id"]): chunk
                  for row in manifest["episodes"] for chunk in row["chunks"]}
    raw_by_episode = {key: [] for key in episode_meta}
    for result in run["results"]:
        path = ROOT / result["path"]
        if (not path.is_file() or path.is_symlink()
                or base.sha256_file(path) != result["sha256"]):
            raise SystemExit("accepted proposal drift: " + result["path"])
        payload = json.loads(path.read_text())
        episode = episode_meta[payload["episode_id"]]
        chunk = chunk_meta[(payload["episode_id"], payload["chunk_id"])]
        errors = validator.validate_proposal(
            payload["normalized_proposal"], episode, chunk)
        if (payload.get("status") != "VALID_MLLM_PROPOSAL"
                or payload.get("provider_model") != "qwen3.8-max"
                or errors):
            raise SystemExit("invalid accepted proposal")
        for row in payload["normalized_proposal"]["candidate_intervals"]:
            raw_by_episode[payload["episode_id"]].append({
                "episode_id": payload["episode_id"],
                "trajectory_id": payload["trajectory_id"],
                "chunk_id": payload["chunk_id"],
                "proposal_path": result["path"],
                "proposal_file_sha256": result["sha256"],
                "proposal": row,
                "start": base.prefix(row["start_frame_id"]),
                "center": base.prefix(row["center_frame_id"]),
                "end": base.prefix(row["end_frame_id"]),
            })

    merged = []
    counts = {}
    review_plan = []
    for episode in manifest["episodes"]:
        episode_id = episode["episode_id"]
        groups = base.components(sorted(
            raw_by_episode[episode_id],
            key=lambda row: (row["start"], row["center"])))
        episode_candidates = []
        counts[episode_id] = {"raw": len(raw_by_episode[episode_id]),
                              "merged": len(groups)}
        for local_index, group in enumerate(groups, 1):
            representative = max(group, key=lambda row: (
                float(row["proposal"]["confidence"]),
                base.KIND_PRIORITY[row["proposal"]["candidate_kind"]],
                -abs(row["end"] - row["start"]), -row["center"]))
            kinds = sorted({row["proposal"]["candidate_kind"]
                            for row in group},
                           key=lambda value: -base.KIND_PRIORITY[value])
            patterns = sorted({row["proposal"]["scene_pattern"]
                               for row in group})
            conflict = ("LIKELY_NO_CHOICE_HARD_NEGATIVE" in kinds
                        and any(value != "LIKELY_NO_CHOICE_HARD_NEGATIVE"
                                for value in kinds))
            event_id = "q%02d_ep%s_hv%02d" % (
                episode["queue_order"], episode_id, local_index)
            candidate = {
                "hindsight_candidate_id": event_id,
                "queue_order": episode["queue_order"],
                "episode_id": episode_id,
                "trajectory_id": episode["trajectory_id"],
                "scene_id": episode["scene_id"],
                "instruction_id": episode["instruction_id"],
                "interval": {
                    "start_frame_id": "P%04d" % min(
                        row["start"] for row in group),
                    "representative_center_frame_id":
                        representative["proposal"]["center_frame_id"],
                    "end_frame_id": "P%04d" % max(
                        row["end"] for row in group),
                    "supporting_frame_ids": base.ordered_ids(
                        value for row in group
                        for value in row["proposal"][
                            "supporting_frame_ids"]),
                },
                "candidate_kind": kinds[0],
                "candidate_kind_votes": kinds,
                "conflicting_kind_votes": conflict,
                "scene_pattern_votes": patterns,
                "reveal_clause_ids": base.ordered_ids(
                    value for row in group for value in
                    row["proposal"]["reveal_clause_ids"]),
                "action_clause_ids": base.ordered_ids(
                    value for row in group for value in
                    row["proposal"]["action_clause_ids"]),
                "representative_summary": representative["proposal"][
                    "reference_route_choice_summary"],
                "representative_rationale": representative["proposal"][
                    "rationale"],
                "representative_confidence": representative["proposal"][
                    "confidence"],
                "source_count": len(group),
                "sources": [{
                    "chunk_id": row["chunk_id"],
                    "proposal_id": row["proposal"]["proposal_id"],
                    "proposal_path": row["proposal_path"],
                    "proposal_file_sha256": row["proposal_file_sha256"],
                    "interval": [row["proposal"]["start_frame_id"],
                                 row["proposal"]["center_frame_id"],
                                 row["proposal"]["end_frame_id"]],
                    "kind": row["proposal"]["candidate_kind"],
                    "confidence": row["proposal"]["confidence"],
                } for row in sorted(group, key=lambda value: (
                    value["chunk_id"], value["proposal"]["proposal_id"]))],
                "geometry_verified": False,
                "online_causality_verified": False,
                "human_reviewed": False,
                "training_label": False,
            }
            merged.append(candidate)
            episode_candidates.append(candidate)

        eligible = [row for row in episode_candidates
                    if row["candidate_kind"] !=
                    "LIKELY_NO_CHOICE_HARD_NEGATIVE"
                    and not row["conflicting_kind_votes"]]
        eligible.sort(key=lambda row: (
            -base.KIND_PRIORITY[row["candidate_kind"]],
            -float(row["representative_confidence"]),
            -int(row["source_count"]),
            base.prefix(row["interval"]["end_frame_id"])
            - base.prefix(row["interval"]["start_frame_id"]),
            base.prefix(row["interval"]["representative_center_frame_id"]),
        ))
        review_plan.append({
            "queue_order": episode["queue_order"],
            "episode_id": episode_id,
            "trajectory_id": episode["trajectory_id"],
            "automatic_candidate_count": len(episode_candidates),
            "eligible_nonnegative_nonconflicting_count": len(eligible),
            "primary_candidate_id": (eligible[0]["hindsight_candidate_id"]
                                     if eligible else None),
            "secondary_candidate_id": (eligible[1]["hindsight_candidate_id"]
                                       if len(eligible) > 1 else None),
            "cascade_rule": (
                "review primary first; reveal secondary only if primary is "
                "human-rejected; no model output counts as human review"
            ),
            "human_review_status": "PENDING",
        })

    primary_count = sum(row["primary_candidate_id"] is not None
                        for row in review_plan)
    output = {
        "manifest": "MF2-CR5 queue50 hindsight candidates and cascade plan",
        "revision": "cr5-queue50-hindsight-aggregate/1",
        "status": "PASS_PENDING_PRIMARY_MULTIVIEW_GATE",
        "input_manifest_sha256": EXPECTED_INPUT_SHA,
        "accepted_run_sha256": EXPECTED_RUN_SHA,
        "requested_and_provider_model": "qwen3.8-max",
        "usage_including_retries": run["usage_including_retries"],
        "merge_rule": {
            "same_episode": True,
            "strict_positive_prefix_interval_overlap": True,
            "shared_boundary_only_does_not_merge": True,
            "semantic_or_geometry_labels_used": False,
        },
        "shortlist_rule": {
            "candidate_kind_priority": base.KIND_PRIORITY,
            "exclude_hard_negative_representative": True,
            "exclude_conflicting_kind_votes": True,
            "then_confidence_source_count_compactness_time": True,
            "maximum_initial_human_cards_per_trajectory": 1,
            "secondary_is_conditional_on_primary_rejection": True,
        },
        "counts_by_episode": counts,
        "raw_candidate_count": sum(value["raw"]
                                   for value in counts.values()),
        "merged_candidate_count": len(merged),
        "trajectory_count": 50,
        "trajectory_with_primary_count": primary_count,
        "trajectory_without_primary_count": 50 - primary_count,
        "initial_primary_review_card_count": primary_count,
        "conditional_secondary_card_count": sum(
            row["secondary_candidate_id"] is not None
            for row in review_plan),
        "candidates": merged,
        "cascade_review_plan": review_plan,
        "fail_closed_rejected_blocks": run["unresolved"],
        "future_frames_are_offline_annotation_only": True,
        "online_causal_labels_created": 0,
        "geometry_verified_candidates": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "raw_candidates": output["raw_candidate_count"],
        "merged_candidates": output["merged_candidate_count"],
        "trajectory_with_primary": primary_count,
        "trajectory_without_primary": 50 - primary_count,
        "initial_primary_cards": primary_count,
        "conditional_secondary_cards": output[
            "conditional_secondary_card_count"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": base.sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
