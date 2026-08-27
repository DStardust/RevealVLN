#!/usr/bin/env python3
"""Build sharded full-trajectory CR5 hindsight inputs for the frozen queue50."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import sys
from pathlib import Path

import cv2


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
HABSIM = ROOT / "third_party/habitat-sim"
for value in (str(SCRIPTS), str(HABSIM)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_phase0c_cr5_hindsight_preflight as base  # noqa: E402


QUEUE = ROOT / "artifacts/phase0/REVEAL_QUEUE_50_MAPPING.json"
RXR_TRAIN = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator"
MEDIA_DIR = OUT_DIR / "private_media"
SHARD_DIR = OUT_DIR / "shards"
OUT = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_INPUTS.json"
EXPECTED = {
    QUEUE: "fe8dfd9c3af01a67a28035787fdfbe4844ca68c47ca1c7c9f363361c6c331ece",
    RXR_TRAIN: "f06b2ef4dc947ca15d6c4a5a3d629c9212328f4cbdd38a13bed9c5c1fc224a94",
    base.PROMPT: "156d19602960f75ace9bef444ae13cbc8c06a5b9916feeec5fc0ad5200a68597",
    base.SCHEMA: "83b8c7ca2b1f4b8bddd9febbb80d99b634afc288bbcd1bc0b10e878a21f85848",
}
NUM_SHARDS = 4
NETWORK_ATTEMPTS = []


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo",
                 "socket.gethostbyname", "socket.gethostbyaddr"}:
        NETWORK_ATTEMPTS.append({"event": event})
        raise RuntimeError("network forbidden in queue50 renderer")


def validate_sources():
    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or base.sha256_file(path) != expected):
            raise RuntimeError("pinned input drift: " + str(path))


def queue_items():
    value = json.loads(QUEUE.read_text())
    items = sorted(value["items"], key=lambda row: row["queue_order"])
    if (len(items) != 50 or [row["queue_order"] for row in items]
            != list(range(50)) or len({row["episode_id"] for row in items})
            != 50 or any(not row["mapped"] or row["split"] != "train"
                         or row["language"] not in base.ALLOWED_LANGUAGES
                         for row in items)):
        raise RuntimeError("frozen queue50 mapping contract failed")
    return items


def load_episodes(items):
    wanted = {str(row["episode_id"]): row for row in items}
    with gzip.open(RXR_TRAIN, "rt") as handle:
        payload = json.load(handle)
    episodes = {str(row["episode_id"]): row for row in payload["episodes"]
                if str(row["episode_id"]) in wanted}
    if set(episodes) != set(wanted):
        raise RuntimeError("RxR train episode closure failed")
    for episode_id, episode in episodes.items():
        mapped = wanted[episode_id]
        instruction = episode["instruction"]
        checks = {
            "instruction_id": str(instruction["instruction_id"])
            == mapped["instruction_id"],
            "trajectory_id": str(episode["trajectory_id"])
            == mapped["trajectory_id"],
            "scene_id": base.scene_name(episode) == mapped["scene_id"],
            "language": instruction["language"] == mapped["language"],
            "instruction_sha256": base.sha256_text(
                instruction["instruction_text"])
            == mapped["instruction_sha256_queue"],
        }
        if not all(checks.values()):
            raise RuntimeError("queue/payload mismatch: " + episode_id)
    return episodes


def write_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.jpg")
    if not cv2.imwrite(str(temporary), image,
                       [int(cv2.IMWRITE_JPEG_QUALITY), base.JPEG_QUALITY]):
        raise RuntimeError("failed to write image: " + str(path))
    os.replace(temporary, path)


def build_one(mapped, episode):
    episode_id = str(mapped["episode_id"])
    instruction = episode["instruction"]
    scene = base.scene_name(episode)
    sim = base.build_sim(scene)
    media = []
    try:
        trace = base.build_lowlevel_trace(sim.pathfinder, episode)
        if not trace:
            raise RuntimeError("empty low-level trace: " + episode_id)
        timeline = base.timeline_indices(trace)
        episode_dir = MEDIA_DIR / ("order%02d_ep%s" % (
            mapped["queue_order"], episode_id))
        route_dir = episode_dir / "route"
        board_dir = episode_dir / "storyboards"
        frame_records = {}
        for prefix in timeline:
            frame_id = "P%04d" % prefix
            path = route_dir / (frame_id + ".jpg")
            write_image(path, base.render(sim, trace[prefix], prefix))
            record = base.media_record(
                path, "chronological_route_frame",
                queue_order=mapped["queue_order"], episode_id=episode_id,
                frame_id=frame_id, prefix_index=prefix,
                action=trace[prefix]["action"],
                position_q=base.q(trace[prefix]["position"]),
                heading_rad=round(float(trace[prefix]["heading"]), 6),
                pixels=[base.FRAME_SIZE, base.FRAME_SIZE + base.HEADER_PX],
            )
            media.append(record)
            frame_records[frame_id] = record

        global_offsets = base.uniform_indices(len(timeline),
                                              base.GLOBAL_FRAMES)
        global_prefixes = [timeline[value] for value in global_offsets]
        global_ids = ["P%04d" % value for value in global_prefixes]
        global_path = board_dir / "GLOBAL.jpg"
        global_image = base.contact_sheet(
            [ROOT / frame_records[value]["path"] for value in global_ids],
            global_ids)
        write_image(global_path, global_image)
        global_record = base.media_record(
            global_path, "global_route_storyboard",
            queue_order=mapped["queue_order"], episode_id=episode_id,
            frame_ids=global_ids,
            pixels=[global_image.shape[1], global_image.shape[0]],
        )
        media.append(global_record)

        chunks = []
        for chunk_index, (start, end) in enumerate(
                base.chunk_ranges(len(timeline))):
            chunk_id = "C%02d" % chunk_index
            prefixes = timeline[start:end]
            frame_ids = ["P%04d" % value for value in prefixes]
            board_path = board_dir / (chunk_id + ".jpg")
            board = base.contact_sheet(
                [ROOT / frame_records[value]["path"] for value in frame_ids],
                frame_ids)
            write_image(board_path, board)
            board_record = base.media_record(
                board_path, "chronological_chunk_storyboard",
                queue_order=mapped["queue_order"], episode_id=episode_id,
                chunk_id=chunk_id, frame_ids=frame_ids,
                pixels=[board.shape[1], board.shape[0]],
            )
            media.append(board_record)
            chunks.append({
                "chunk_id": chunk_id,
                "timeline_offset_start": start,
                "timeline_offset_end_exclusive": end,
                "frame_ids": frame_ids,
                "frame_paths": [frame_records[value]["path"]
                                for value in frame_ids],
                "storyboard_path": board_record["path"],
                "storyboard_sha256": board_record["sha256"],
                "overlap_with_previous": 0 if chunk_index == 0 else
                    len(set(frame_ids) & set(chunks[-1]["frame_ids"])),
            })

        text = instruction["instruction_text"]
        event = {
            "queue_order": mapped["queue_order"],
            "trajectory_id": str(episode["trajectory_id"]),
            "episode_id": episode_id,
            "scene_id": scene,
            "instruction_id": str(instruction["instruction_id"]),
            "language": instruction["language"],
            "instruction_text": text,
            "instruction_sha256": base.sha256_text(text),
            "deterministic_segments": base.instruction_segments(text),
            "trace_length": len(trace),
            "timeline_frame_ids": ["P%04d" % value for value in timeline],
            "timeline_prefix_indices": timeline,
            "timeline_sampling": {
                "all_30_degree_turn_prefixes_retained": True,
                "move_sample_max_m": base.MOVE_SAMPLE_M,
                "first_and_last_retained": True,
            },
            "global_storyboard": {
                "path": global_record["path"],
                "sha256": global_record["sha256"],
                "frame_ids": global_ids,
            },
            "chunks": chunks,
            "trace_pose_action_sha256": base.stable_sha([{
                "prefix_index": index,
                "position": base.q(row["position"]),
                "heading_rad": round(float(row["heading"]), 6),
                "action": row["action"],
            } for index, row in enumerate(trace)]),
            "legacy_target_fields_in_model_input": False,
            "mllm_output": None,
        }
        return event, media
    finally:
        sim.close()


def run_shard(shard_index: int, gpu_device: int):
    if not 0 <= shard_index < NUM_SHARDS:
        raise RuntimeError("invalid shard index")
    base.GPU_DEVICE = gpu_device
    items = queue_items()
    episodes = load_episodes(items)
    selected = [row for row in items
                if row["queue_order"] % NUM_SHARDS == shard_index]
    event_records, media = [], []
    for mapped in selected:
        event, records = build_one(
            mapped, episodes[str(mapped["episode_id"])])
        event_records.append(event)
        media.extend(records)
        print("shard", shard_index, "order", mapped["queue_order"],
              "episode", mapped["episode_id"], "frames",
              len(event["timeline_frame_ids"]), flush=True)
    output = {
        "revision": "cr5-queue50-hindsight-shard/1",
        "shard_index": shard_index,
        "num_shards": NUM_SHARDS,
        "gpu_device_id": gpu_device,
        "queue_mapping_sha256": EXPECTED[QUEUE],
        "rxr_train_sha256": EXPECTED[RXR_TRAIN],
        "episode_count": len(event_records),
        "episodes": event_records,
        "media_manifest": sorted(media, key=lambda value: value["path"]),
        "network_attempts": len(NETWORK_ATTEMPTS),
        "future_frames_are_offline_annotation_only": True,
        "online_labels_created": 0,
        "training_authorized": False,
    }
    path = SHARD_DIR / ("shard_%02d.json" % shard_index)
    atomic_json(path, output)
    print(json.dumps({"status": "SHARD_PASS", "path": str(
        path.relative_to(ROOT)), "sha256": base.sha256_file(path),
        "episodes": len(event_records), "media": len(media)}))


def aggregate():
    items = queue_items()
    episodes, media, shards = [], [], []
    for index in range(NUM_SHARDS):
        path = SHARD_DIR / ("shard_%02d.json" % index)
        value = json.loads(path.read_text())
        if (value["shard_index"] != index
                or value["num_shards"] != NUM_SHARDS
                or value["network_attempts"] != 0
                or value["online_labels_created"] != 0
                or value["training_authorized"] is not False):
            raise RuntimeError("shard contract failed")
        episodes.extend(value["episodes"])
        media.extend(value["media_manifest"])
        shards.append({"path": str(path.relative_to(ROOT)),
                       "sha256": base.sha256_file(path),
                       "episode_count": value["episode_count"]})
    episodes.sort(key=lambda row: row["queue_order"])
    if ([row["queue_order"] for row in episodes] != list(range(50))
            or [row["episode_id"] for row in episodes]
            != [str(row["episode_id"]) for row in items]):
        raise RuntimeError("aggregate queue closure failed")
    paths = [row["path"] for row in media]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate media paths")
    for record in media:
        path = ROOT / record["path"]
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != record["bytes"]
                or base.sha256_file(path) != record["sha256"]):
            raise RuntimeError("media integrity failure: " + record["path"])
    output = {
        "manifest": "MF2-CR5 frozen queue50 full-trajectory hindsight inputs",
        "revision": "cr5-queue50-hindsight-inputs/1",
        "status": "READY_FOR_MLLM_DRY_RUN",
        "source_scope": "RxR-CE-en train only",
        "queue_mapping": {
            "path": str(QUEUE.relative_to(ROOT)),
            "sha256": EXPECTED[QUEUE],
            "sampling_seed": 20260822,
            "episode_count": 50,
        },
        "rxr_train": {"path": str(RXR_TRAIN.relative_to(ROOT)),
                      "bytes": RXR_TRAIN.stat().st_size,
                      "sha256": EXPECTED[RXR_TRAIN]},
        "contract": {
            "prompt_path": str(base.PROMPT.relative_to(ROOT)),
            "prompt_sha256": EXPECTED[base.PROMPT],
            "schema_path": str(base.SCHEMA.relative_to(ROOT)),
            "schema_sha256": EXPECTED[base.SCHEMA],
        },
        "rendering": {
            "habitat_sim_source": "project-local pinned Habitat-Sim v0.1.7",
            "gpu_shards": shards,
            "rgb_pixels": [base.FRAME_SIZE, base.FRAME_SIZE],
            "frame_header_pixels": base.HEADER_PX,
            "hfov_deg": 63.0,
            "sensor_height_m": 0.88,
            "jpeg_quality": base.JPEG_QUALITY,
        },
        "chunking": {"max_frames": base.CHUNK_FRAMES,
                     "minimum_overlap_frames": base.CHUNK_OVERLAP,
                     "global_storyboard_max_frames": base.GLOBAL_FRAMES},
        "episode_count": len(episodes),
        "episodes": episodes,
        "media_manifest": sorted(media, key=lambda value: value["path"]),
        "media_file_count": len(media),
        "media_total_bytes": sum(row["bytes"] for row in media),
        "network_calls_made": 0,
        "future_frames_are_offline_annotation_only": True,
        "online_prefix_causality_verified": False,
        "branch_labels_created": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"], "episodes": output["episode_count"],
        "chunks": sum(len(row["chunks"]) for row in episodes),
        "media_files": output["media_file_count"],
        "media_total_bytes": output["media_total_bytes"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": base.sha256_file(OUT),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--gpu-device", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    validate_sources()
    sys.addaudithook(audit)
    if args.aggregate:
        aggregate()
    elif args.shard_index is not None and args.gpu_device is not None:
        run_shard(args.shard_index, args.gpu_device)
    else:
        raise SystemExit("choose --aggregate or --shard-index + --gpu-device")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
