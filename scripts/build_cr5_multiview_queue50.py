#!/usr/bin/env python3
"""Render sharded 3x12 multi-view evidence for queue50 primary events."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

import cv2


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
HABSIM = ROOT / "third_party/habitat-sim"
for value in (str(SCRIPTS), str(HABSIM)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_phase0c_cr5_multiview_preflight as base  # noqa: E402


AGGREGATED = ROOT / (
    "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/"
    "CR5_QUEUE50_HINDSIGHT_AGGREGATED.json"
)
LOCATOR_INPUT = AGGREGATED.with_name("CR5_QUEUE50_HINDSIGHT_INPUTS.json")
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
MEDIA_DIR = OUT_DIR / "private_media"
SHARD_DIR = OUT_DIR / "shards"
OUT = OUT_DIR / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
EXPECTED_AGGREGATED_SHA = (
    "46f4e592b8b15e16df9641351b5d08a0d1a9fe6b59def4959b34775b8b469612"
)
EXPECTED_LOCATOR_INPUT_SHA = (
    "8e00000ee306369e305c53d580444e1ac3228a6e94c3c424d84f9db5d16ea151"
)
EXPECTED_PROMPT_SHA = (
    "5a3637109eaabc5f0464f5b808c04d42e6433b9cdb5ae106ee0fb0873f9eba85"
)
EXPECTED_SCHEMA_SHA = (
    "d3c76ee4c26b47f9f9b3d03d9a1244d2dad6565331e44e014ff823514f8e5f33"
)
NUM_SHARDS = 4


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)
                         + "\n")
    os.replace(temporary, path)


def write_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.jpg")
    if not cv2.imwrite(str(temporary), image,
                       [int(cv2.IMWRITE_JPEG_QUALITY), base.JPEG_QUALITY]):
        raise RuntimeError("failed to write image")
    os.replace(temporary, path)


def sources():
    for path, expected in ((AGGREGATED, EXPECTED_AGGREGATED_SHA),
                           (LOCATOR_INPUT, EXPECTED_LOCATOR_INPUT_SHA),
                           (base.PROMPT, EXPECTED_PROMPT_SHA),
                           (base.SCHEMA, EXPECTED_SCHEMA_SHA)):
        if (not path.is_file() or path.is_symlink()
                or base.sha256_file(path) != expected):
            raise RuntimeError("multiview source drift: " + str(path))
    aggregate = json.loads(AGGREGATED.read_text())
    locator = json.loads(LOCATOR_INPUT.read_text())
    candidate_map = {row["hindsight_candidate_id"]: row
                     for row in aggregate["candidates"]}
    primaries = []
    for row in aggregate["cascade_review_plan"]:
        event_id = row["primary_candidate_id"]
        if event_id is None or event_id not in candidate_map:
            raise RuntimeError("queue trajectory lacks primary")
        primaries.append(candidate_map[event_id])
    if (len(primaries) != 50
            or [row["queue_order"] for row in primaries] != list(range(50))
            or len({row["episode_id"] for row in primaries}) != 50):
        raise RuntimeError("primary queue closure failed")
    return aggregate, locator, primaries


def build_event(candidate, locator_episode, locator_media, episode, sim):
    trace = base.build_lowlevel_trace(sim.pathfinder, episode)
    center = base.prefix_number(candidate["interval"][
        "representative_center_frame_id"])
    if not 0 <= center < len(trace):
        raise RuntimeError("candidate center outside trace")
    a_prefix, a_distance = base.position_prefix(
        trace, center, -1, base.POSITION_DISTANCE_M)
    d_prefix, d_distance = base.position_prefix(
        trace, center, 1, base.POSITION_DISTANCE_M)
    roles = [("A", a_prefix, a_distance), ("Q", center, 0.0),
             ("D", d_prefix, d_distance)]
    event_dir = MEDIA_DIR / candidate["hindsight_candidate_id"]
    role_records, media = {}, []
    for role, trace_prefix, achieved in roles:
        base_heading = base.route_forward_heading(trace, trace_prefix)
        view_records, view_paths, view_labels = [], [], []
        for view_index in range(base.PANORAMA_HEADINGS):
            view_id = "%s_V%02d" % (role, view_index)
            offset = base.relative_yaw(view_index)
            heading = (base_heading + math.radians(offset)) % (2 * math.pi)
            direction = "LEFT" if offset > 0 else (
                "RIGHT" if offset < 0 else "FORWARD")
            label = "%s | %+d deg %s" % (view_id, int(offset), direction)
            path = event_dir / (view_id + ".jpg")
            write_image(path, base.render(
                sim, trace[trace_prefix]["position"], heading, label))
            record = base.media_record(
                path, "panorama_view",
                event_id=candidate["hindsight_candidate_id"], role=role,
                view_id=view_id, trace_prefix=trace_prefix,
                relative_yaw_deg=offset,
                habitat_heading_rad=round(heading, 6),
                position_q=base.q(trace[trace_prefix]["position"]),
                pixels=[base.FRAME_SIZE, base.FRAME_SIZE + base.HEADER_PX])
            media.append(record)
            view_records.append(record)
            view_paths.append(path)
            view_labels.append(view_id)
        board_path = event_dir / (role + "_PANORAMA.jpg")
        board = base.contact_sheet(view_paths, view_labels)
        write_image(board_path, board)
        board_record = base.media_record(
            board_path, "panorama_contact_sheet",
            event_id=candidate["hindsight_candidate_id"], role=role,
            view_ids=[row["view_id"] for row in view_records],
            pixels=[board.shape[1], board.shape[0]])
        media.append(board_record)
        role_records[role] = {
            "trace_prefix": trace_prefix,
            "frame_id": base.prefix_id(trace_prefix),
            "requested_route_distance_m": base.POSITION_DISTANCE_M
                if role != "Q" else 0.0,
            "achieved_route_distance_m": round(achieved, 6),
            "position_q": base.q(trace[trace_prefix]["position"]),
            "route_forward_heading_rad": round(base_heading, 6),
            "views": view_records,
            "contact_sheet": board_record,
        }
    timeline = locator_episode["timeline_prefix_indices"]
    nearest_offset = min(range(len(timeline)),
                         key=lambda index: abs(timeline[index] - center))
    low = max(0, nearest_offset - base.CONTEXT_RADIUS)
    high = min(len(timeline), nearest_offset + base.CONTEXT_RADIUS + 1)
    context_prefixes = timeline[low:high]
    context_prefixes = sorted(set(context_prefixes + [
        base.prefix_number(value) for value in
        candidate["interval"]["supporting_frame_ids"]]))
    context_records = []
    for value in context_prefixes:
        key = base.prefix_id(value) + "@" + candidate["episode_id"]
        if key not in locator_media:
            raise RuntimeError("missing locator context: " + key)
        context_records.append(locator_media[key])
    event = {
        "event_id": candidate["hindsight_candidate_id"],
        "queue_order": candidate["queue_order"],
        "episode_id": candidate["episode_id"],
        "trajectory_id": candidate["trajectory_id"],
        "scene_id": candidate["scene_id"],
        "instruction_id": candidate["instruction_id"],
        "language": locator_episode["language"],
        "instruction_text": locator_episode["instruction_text"],
        "instruction_sha256": locator_episode["instruction_sha256"],
        "deterministic_segments": locator_episode["deterministic_segments"],
        "candidate_interval": candidate["interval"],
        "candidate_center_source": "queue50_full_trajectory_hindsight_"
                                   "primary_shortlist",
        "locator_free_text_in_model_input": False,
        "legacy_bt_in_model_input": False,
        "positions": role_records,
        "chronological_context_frames": context_records,
        "mllm_branch_proposal": None,
        "geometry_verified": False,
        "human_reviewed": False,
        "training_label": False,
    }
    return event, media


def run_shard(shard_index: int, gpu_device: int):
    if not 0 <= shard_index < NUM_SHARDS:
        raise RuntimeError("invalid shard")
    base.GPU_DEVICE = gpu_device
    _, locator, primaries = sources()
    selected = [row for row in primaries
                if row["queue_order"] % NUM_SHARDS == shard_index]
    locator_episodes = {row["episode_id"]: row
                        for row in locator["episodes"]}
    locator_media = {row["frame_id"] + "@" + row["episode_id"]: row
                     for row in locator["media_manifest"]
                     if row["kind"] == "chronological_route_frame"}
    wanted = {row["episode_id"] for row in selected}
    with gzip.open(base.RXR_TRAIN, "rt") as handle:
        episodes = {str(row["episode_id"]): row
                    for row in json.load(handle)["episodes"]
                    if str(row["episode_id"]) in wanted}
    if set(episodes) != wanted:
        raise RuntimeError("RxR primary episode closure failed")
    by_scene = {}
    for candidate in selected:
        by_scene.setdefault(candidate["scene_id"], []).append(candidate)
    events, media = [], []
    for scene, candidates in sorted(by_scene.items()):
        sim = base.build_sim(scene)
        try:
            for candidate in sorted(candidates,
                                    key=lambda row: row["queue_order"]):
                event, records = build_event(
                    candidate, locator_episodes[candidate["episode_id"]],
                    locator_media, episodes[candidate["episode_id"]], sim)
                events.append(event)
                media.extend(records)
                print("shard", shard_index, "order", event["queue_order"],
                      event["event_id"], flush=True)
        finally:
            sim.close()
    events.sort(key=lambda row: row["queue_order"])
    output = {
        "revision": "cr5-queue50-primary-multiview-shard/1",
        "shard_index": shard_index, "num_shards": NUM_SHARDS,
        "gpu_device_id": gpu_device,
        "aggregated_sha256": EXPECTED_AGGREGATED_SHA,
        "locator_input_sha256": EXPECTED_LOCATOR_INPUT_SHA,
        "events": events,
        "media_manifest": sorted(media, key=lambda row: row["path"]),
        "event_count": len(events), "network_calls_made": 0,
        "branch_labels_created": 0, "human_labels_created": 0,
        "training_authorized": False,
    }
    path = SHARD_DIR / ("shard_%02d.json" % shard_index)
    atomic_json(path, output)
    print(json.dumps({"status": "SHARD_PASS", "events": len(events),
                      "media": len(media), "path": str(path.relative_to(ROOT)),
                      "sha256": base.sha256_file(path)}))


def aggregate():
    _, _, primaries = sources()
    events, media, shards = [], [], []
    for index in range(NUM_SHARDS):
        path = SHARD_DIR / ("shard_%02d.json" % index)
        value = json.loads(path.read_text())
        if (value["shard_index"] != index
                or value["num_shards"] != NUM_SHARDS
                or value["network_calls_made"] != 0
                or value["branch_labels_created"] != 0
                or value["human_labels_created"] != 0
                or value["training_authorized"] is not False):
            raise RuntimeError("multiview shard contract failed")
        events.extend(value["events"])
        media.extend(value["media_manifest"])
        shards.append({"path": str(path.relative_to(ROOT)),
                       "sha256": base.sha256_file(path),
                       "event_count": value["event_count"]})
    events.sort(key=lambda row: row["queue_order"])
    if ([row["queue_order"] for row in events] != list(range(50))
            or [row["event_id"] for row in events] != [
                row["hindsight_candidate_id"] for row in primaries]):
        raise RuntimeError("multiview aggregate closure failed")
    if len(media) != 50 * 39 or len({row["path"] for row in media}) != len(media):
        raise RuntimeError("multiview media count/uniqueness failed")
    for record in media:
        path = ROOT / record["path"]
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != record["bytes"]
                or base.sha256_file(path) != record["sha256"]):
            raise RuntimeError("multiview media integrity failed")
    output = {
        "manifest": "MF2-CR5 queue50 primary multi-view inputs",
        "revision": "cr5-queue50-primary-multiview-inputs/1",
        "status": "READY_FOR_BRANCH_PROPOSER_DRY_RUN",
        "source_scope": "RxR-CE-en train only",
        "aggregated_source": {"path": str(AGGREGATED.relative_to(ROOT)),
                              "sha256": EXPECTED_AGGREGATED_SHA},
        "locator_input_source": {
            "path": str(LOCATOR_INPUT.relative_to(ROOT)),
            "sha256": EXPECTED_LOCATOR_INPUT_SHA},
        "contract": {"prompt_path": str(base.PROMPT.relative_to(ROOT)),
                     "prompt_sha256": EXPECTED_PROMPT_SHA,
                     "schema_path": str(base.SCHEMA.relative_to(ROOT)),
                     "schema_sha256": EXPECTED_SCHEMA_SHA},
        "rendering": {
            "gpu_shards": shards, "positions": ["A", "Q", "D"],
            "requested_position_spacing_m": base.POSITION_DISTANCE_M,
            "headings_per_position": base.PANORAMA_HEADINGS,
            "heading_step_deg": base.PANORAMA_STEP_DEG,
            "positive_habitat_yaw_is_left": True,
            "v00_is_local_reference_route_forward": True,
            "rgb_pixels": [base.FRAME_SIZE, base.FRAME_SIZE],
            "header_pixels": base.HEADER_PX, "hfov_deg": 63.0,
            "sensor_height_m": 0.88, "jpeg_quality": base.JPEG_QUALITY},
        "event_count": len(events), "events": events,
        "media_manifest": sorted(media, key=lambda row: row["path"]),
        "media_file_count": len(media),
        "media_total_bytes": sum(row["bytes"] for row in media),
        "network_calls_made": 0, "branch_labels_created": 0,
        "geometry_verified_candidates": 0, "human_labels_created": 0,
        "future_frames_are_offline_annotation_only": True,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({"status": output["status"], "events": len(events),
                      "panorama_views": 50 * 36,
                      "media_files": len(media),
                      "media_total_bytes": output["media_total_bytes"],
                      "output": str(OUT.relative_to(ROOT)),
                      "sha256": base.sha256_file(OUT)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--gpu-device", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.shard_index is not None and args.gpu_device is not None:
        run_shard(args.shard_index, args.gpu_device)
    else:
        raise SystemExit("choose shard or aggregate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
