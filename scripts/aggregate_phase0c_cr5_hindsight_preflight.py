#!/usr/bin/env python3
"""Merge overlapping CR5 hindsight chunk proposals by stable prefix time."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
INPUT = ROOT / (
    "artifacts/phase0/phase0c_cr5_preflight/hindsight_locator/"
    "CR5_HINDSIGHT_PREFLIGHT_INPUTS_V2.json"
)
RUN = INPUT.with_name("CR5_HINDSIGHT_PREFLIGHT_RUN_V2.json")
OUT = INPUT.with_name("CR5_HINDSIGHT_PREFLIGHT_AGGREGATED.json")
LEGACY_PROBE = ROOT / (
    "artifacts/runtime/phase0_correctness/PHASE0C_ORACLE_LOWLEVEL_PROBE.json"
)
EXPECTED_INPUT_SHA = "939945e2a21fb571aeec7c7f8914be6873bf73ef08b7e7b12d3e2d94ac9d999d"
EXPECTED_RUN_SHA = "b58cb83ccce25730cb780d8727b8590e4ff54ec13a2865d5ad14b331d876c03f"
CALIBRATION_IDS = {
    "ep41233_turn01", "ep34121_turn02", "ep46758_turn03",
    "ep43805_turn02", "ep7619_turn05", "ep56443_turn01",
}
KIND_PRIORITY = {
    "LIKELY_DECISION": 2,
    "POSSIBLE_DECISION": 1,
    "LIKELY_NO_CHOICE_HARD_NEGATIVE": 0,
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def prefix(value: str) -> int:
    match = re.fullmatch(r"P([0-9]{4,6})", value)
    if not match:
        raise ValueError("invalid frame ID: " + value)
    return int(match.group(1))


def ordered_ids(values):
    return sorted(set(values), key=lambda value: int(value[1:]))


def overlaps(left, right) -> bool:
    # A shared boundary frame is not enough: adjacent decisions must survive.
    return max(left["start"], right["start"]) < min(left["end"], right["end"])


def components(records):
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if overlaps(records[i], records[j]):
                union(i, j)
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    return sorted(groups.values(), key=lambda group: (
        min(row["start"] for row in group),
        min(row["center"] for row in group),
    ))


def main() -> int:
    if sha256_file(INPUT) != EXPECTED_INPUT_SHA:
        raise SystemExit("input SHA drift")
    if sha256_file(RUN) != EXPECTED_RUN_SHA:
        raise SystemExit("run SHA drift")
    manifest = json.loads(INPUT.read_text())
    run = json.loads(RUN.read_text())
    if run.get("status") != "PASS" or run.get("valid_count") != 19:
        raise SystemExit("hindsight run not complete")
    episode_meta = {row["episode_id"]: row for row in manifest["episodes"]}
    raw_by_episode = {key: [] for key in episode_meta}
    for result in run["results"]:
        path = ROOT / result["path"]
        if sha256_file(path) != result["sha256"]:
            raise SystemExit("proposal SHA drift: " + result["path"])
        payload = json.loads(path.read_text())
        if (payload.get("status") != "VALID_MLLM_PROPOSAL"
                or payload.get("provider_model") != "qwen3.8-max"
                or payload.get("validation_errors")):
            raise SystemExit("invalid proposal in accepted run")
        for row in payload["normalized_proposal"]["candidate_intervals"]:
            raw_by_episode[payload["episode_id"]].append({
                "episode_id": payload["episode_id"],
                "trajectory_id": payload["trajectory_id"],
                "chunk_id": payload["chunk_id"],
                "proposal_path": result["path"],
                "proposal_file_sha256": result["sha256"],
                "proposal": row,
                "start": prefix(row["start_frame_id"]),
                "center": prefix(row["center_frame_id"]),
                "end": prefix(row["end_frame_id"]),
            })

    merged = []
    counts = {}
    for episode_id in sorted(raw_by_episode, key=int):
        groups = components(sorted(raw_by_episode[episode_id],
                                   key=lambda row: (row["start"],
                                                    row["center"])))
        counts[episode_id] = {"raw": len(raw_by_episode[episode_id]),
                              "merged": len(groups)}
        for local_index, group in enumerate(groups, 1):
            representative = max(group, key=lambda row: (
                float(row["proposal"]["confidence"]),
                KIND_PRIORITY[row["proposal"]["candidate_kind"]],
                -abs(row["end"] - row["start"]),
                -row["center"],
            ))
            kinds = sorted({row["proposal"]["candidate_kind"]
                            for row in group},
                           key=lambda value: -KIND_PRIORITY[value])
            patterns = sorted({row["proposal"]["scene_pattern"]
                               for row in group})
            event_id = "ep%s_hv%02d" % (episode_id, local_index)
            merged.append({
                "hindsight_candidate_id": event_id,
                "episode_id": episode_id,
                "trajectory_id": episode_meta[episode_id]["trajectory_id"],
                "scene_id": episode_meta[episode_id]["scene_id"],
                "instruction_id": episode_meta[episode_id]["instruction_id"],
                "interval": {
                    "start_frame_id": "P%04d" % min(row["start"]
                                                      for row in group),
                    "representative_center_frame_id":
                        representative["proposal"]["center_frame_id"],
                    "end_frame_id": "P%04d" % max(row["end"]
                                                    for row in group),
                    "supporting_frame_ids": ordered_ids(
                        value for row in group
                        for value in row["proposal"][
                            "supporting_frame_ids"]),
                },
                "candidate_kind": kinds[0],
                "candidate_kind_votes": kinds,
                "conflicting_kind_votes":
                    ("LIKELY_NO_CHOICE_HARD_NEGATIVE" in kinds
                     and any(value != "LIKELY_NO_CHOICE_HARD_NEGATIVE"
                             for value in kinds)),
                "scene_pattern_votes": patterns,
                "reveal_clause_ids": ordered_ids(
                    value for row in group
                    for value in row["proposal"]["reveal_clause_ids"]),
                "action_clause_ids": ordered_ids(
                    value for row in group
                    for value in row["proposal"]["action_clause_ids"]),
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
                "training_label": False,
            })

    probe = json.loads(LEGACY_PROBE.read_text())
    legacy = []
    for row in probe["events"]:
        if row["provisional_event_id"] not in CALIBRATION_IDS:
            continue
        reveal = int(row["candidate_reveal_prefix"])
        eligible = [value for value in merged
                    if value["episode_id"] == str(row["episode_id"])]

        def interval_distance(value):
            low = prefix(value["interval"]["start_frame_id"])
            high = prefix(value["interval"]["end_frame_id"])
            return 0 if low <= reveal <= high else min(abs(reveal - low),
                                                       abs(reveal - high))

        nearest = min(eligible, key=lambda value: (
            interval_distance(value),
            abs(reveal - prefix(value["interval"][
                "representative_center_frame_id"]))))
        legacy.append({
            "legacy_event_id": row["provisional_event_id"],
            "legacy_candidate_reveal_frame_id": "P%04d" % reveal,
            "nearest_hindsight_candidate_id": nearest[
                "hindsight_candidate_id"],
            "distance_to_interval_prefixes": interval_distance(nearest),
            "legacy_bt_used_for_mllm": False,
        })
    if len(legacy) != len(CALIBRATION_IDS):
        raise SystemExit("calibration closure failure")

    output = {
        "manifest": "MF2-CR5 hindsight locator merged preflight candidates",
        "revision": "cr5-hindsight-aggregate/1-strict-overlap",
        "status": "PASS_PENDING_MULTIVIEW_PROPOSER",
        "input_manifest_sha256": EXPECTED_INPUT_SHA,
        "run_sha256": EXPECTED_RUN_SHA,
        "requested_and_provider_model": "qwen3.8-max",
        "usage": run["usage"],
        "merge_rule": {
            "same_episode": True,
            "strict_positive_prefix_interval_overlap": True,
            "shared_boundary_only_does_not_merge": True,
            "semantic_or_geometry_labels_used": False,
        },
        "counts_by_episode": counts,
        "raw_candidate_count": sum(value["raw"] for value in counts.values()),
        "merged_candidate_count": len(merged),
        "candidates": merged,
        "legacy_calibration_correspondence": sorted(
            legacy, key=lambda value: value["legacy_event_id"]),
        "future_frames_are_offline_annotation_only": True,
        "online_causal_labels_created": 0,
        "geometry_verified_candidates": 0,
        "training_authorized": False,
    }
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "raw_candidates": output["raw_candidate_count"],
        "merged_candidates": output["merged_candidate_count"],
        "counts_by_episode": counts,
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
