#!/usr/bin/env python3
"""Build six blinded, train-only CR5 full-trajectory locator inputs.

This builder does not call an MLLM and does not create branch labels.  It
renders a deterministic chronological timeline for each complete reference
execution.  The six trajectories cover the user-identified calibration cases,
but their expected judgments and legacy B/T labels are intentionally absent
from every model-facing record.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
HABSIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / "third_party/habitat-sim"))
for value in (str(SCRIPTS), str(HABSIM)):
    if value not in sys.path:
        sys.path.insert(0, value)

from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402


RXR_TRAIN = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_preflight/hindsight_locator"
MEDIA_DIR = OUT_DIR / "private_media"
OUT = OUT_DIR / "CR5_HINDSIGHT_PREFLIGHT_INPUTS_V2.json"
CONTRACT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_contract"
PROMPT = CONTRACT_DIR / "CR5_MLLM_TRAJECTORY_LOCATOR_PROMPT_V2.md"
SCHEMA = CONTRACT_DIR / "CR5_MLLM_TRAJECTORY_LOCATOR_SCHEMA.json"
EPISODE_IDS = ("41233", "34121", "46758", "43805", "7619", "56443")
ALLOWED_LANGUAGES = {"en-IN", "en-US"}
FRAME_SIZE = 448
HEADER_PX = 48
JPEG_QUALITY = 92
MOVE_SAMPLE_M = 0.5
CHUNK_FRAMES = 20
CHUNK_OVERLAP = 5
GLOBAL_FRAMES = 20
GPU_DEVICE = int(os.environ.get("CR5_PREFLIGHT_GPU", "1"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False)
    return sha256_text(raw)


def q(values, places: int = 6):
    return [round(float(value), places) for value in values]


def distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


def instruction_segments(source: str):
    boundaries = []
    for index, character in enumerate(source):
        if character in ",;:.!?":
            end = index + 1
            while end < len(source) and source[end] in "\"'\u2019\u201d":
                end += 1
            boundaries.append(end)
    if not boundaries or boundaries[-1] != len(source):
        boundaries.append(len(source))
    segments, start = [], 0
    for end in boundaries:
        left, right = start, end
        while left < right and source[left].isspace():
            left += 1
        while right > left and source[right - 1].isspace():
            right -= 1
        if left < right:
            text = source[left:right]
            segments.append({
                "segment_id": "S%02d" % (len(segments) + 1),
                "char_start": left,
                "char_end_exclusive": right,
                "text": text,
                "text_sha256": sha256_text(text),
            })
        start = end
    if not segments or any(
            item["text"] != source[item["char_start"]:
                                   item["char_end_exclusive"]]
            for item in segments):
        raise RuntimeError("exact instruction segmentation failed")
    return segments


def timeline_indices(trace):
    """Keep every 30-degree turn and at most 0.5 m between move samples."""
    selected = [0]
    accumulated = 0.0
    previous = trace[0]["position"]
    for index in range(1, len(trace)):
        row = trace[index]
        step = distance(previous, row["position"])
        accumulated += step
        previous = row["position"]
        if row["action"] == "TURN":
            selected.append(index)
        elif accumulated + 1e-9 >= MOVE_SAMPLE_M:
            selected.append(index)
            accumulated = 0.0
    if selected[-1] != len(trace) - 1:
        selected.append(len(trace) - 1)
    return sorted(set(selected))


def uniform_indices(length: int, count: int):
    if length <= 0:
        return []
    count = min(length, count)
    if count == 1:
        return [0]
    return sorted({int(round(i * (length - 1) / (count - 1)))
                   for i in range(count)})


def chunk_ranges(length: int):
    if length <= CHUNK_FRAMES:
        return [(0, length)]
    step = CHUNK_FRAMES - CHUNK_OVERLAP
    starts = list(range(0, length - CHUNK_FRAMES + 1, step))
    last = length - CHUNK_FRAMES
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + CHUNK_FRAMES) for start in starts]


def scene_name(episode) -> str:
    parts = Path(episode["scene_id"]).parts
    if len(parts) != 3 or parts[0] != "mp3d" or parts[1] != parts[2][:-4]:
        raise RuntimeError("unexpected RxR scene path")
    return parts[1]


def build_sim(scene: str):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(MP3D / scene / (scene + ".glb"))
    sim_cfg.gpu_device_id = GPU_DEVICE
    sensor = habitat_sim.SensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = [FRAME_SIZE, FRAME_SIZE]
    sensor.position = [0.0, 0.88, 0.0]
    sensor.hfov = 63.0
    agent = habitat_sim.AgentConfiguration()
    agent.height = 0.88
    agent.radius = 0.18
    agent.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))
    navmesh = MP3D / scene / (scene + ".navmesh")
    if not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError("navmesh load failed: " + scene)
    return sim


def render(sim, state, prefix: int):
    import habitat_sim
    from scipy.spatial.transform import Rotation

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.asarray(
        sim.pathfinder.snap_point(state["position"]), dtype="float32")
    agent_state.rotation = Rotation.from_rotvec(
        [0.0, float(state["heading"]), 0.0]).as_quat()
    sim.get_agent(0).set_state(agent_state, True)
    rgb = sim.get_sensor_observations()["rgb"][..., :3].copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    canvas = np.zeros((FRAME_SIZE + HEADER_PX, FRAME_SIZE, 3),
                      dtype=np.uint8)
    canvas[HEADER_PX:] = bgr
    label = "P%04d | %s" % (prefix, state["action"])
    cv2.putText(canvas, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def contact_sheet(paths, labels, columns: int = 4, tile: int = 240):
    rows = int(math.ceil(len(paths) / columns))
    sheet = np.full((rows * tile, columns * tile, 3), 24,
                    dtype=np.uint8)
    for index, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to read rendered frame")
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        y, x = divmod(index, columns)
        sheet[y * tile:(y + 1) * tile,
              x * tile:(x + 1) * tile] = image
        cv2.rectangle(sheet, (x * tile, y * tile),
                      ((x + 1) * tile - 1, (y + 1) * tile - 1),
                      (100, 100, 100), 1)
        cv2.putText(sheet, label, (x * tile + 6, y * tile + tile - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1,
                    cv2.LINE_AA)
    return sheet


def media_record(path: Path, kind: str, **extra):
    value = {
        "path": str(path.relative_to(ROOT)),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    value.update(extra)
    return value


def main() -> int:
    if not RXR_TRAIN.is_file() or RXR_TRAIN.is_symlink():
        raise SystemExit("RxR train payload is missing or unsafe")
    if not PROMPT.is_file() or not SCHEMA.is_file():
        raise SystemExit("CR5 locator contract missing")
    with gzip.open(RXR_TRAIN, "rt") as handle:
        payload = json.load(handle)
    episodes = {str(row["episode_id"]): row for row in payload["episodes"]
                if str(row["episode_id"]) in EPISODE_IDS}
    if set(episodes) != set(EPISODE_IDS):
        raise SystemExit("preflight episode closure failed")

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    event_records, media = [], []
    for episode_id in EPISODE_IDS:
        episode = episodes[episode_id]
        instruction = episode["instruction"]
        if instruction["language"] not in ALLOWED_LANGUAGES:
            raise RuntimeError("non-English preflight instruction")
        scene = scene_name(episode)
        sim = build_sim(scene)
        try:
            trace = build_lowlevel_trace(sim.pathfinder, episode)
            if not trace:
                raise RuntimeError("empty low-level trace: " + episode_id)
            timeline = timeline_indices(trace)
            episode_dir = MEDIA_DIR / ("ep" + episode_id)
            route_dir = episode_dir / "route"
            board_dir = episode_dir / "storyboards"
            route_dir.mkdir(parents=True, exist_ok=True)
            board_dir.mkdir(parents=True, exist_ok=True)

            frame_records = {}
            for prefix in timeline:
                frame_id = "P%04d" % prefix
                path = route_dir / (frame_id + ".jpg")
                image = render(sim, trace[prefix], prefix)
                if not cv2.imwrite(str(path), image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY),
                                    JPEG_QUALITY]):
                    raise RuntimeError("failed to write route frame")
                record = media_record(
                    path, "chronological_route_frame",
                    episode_id=episode_id,
                    frame_id=frame_id,
                    prefix_index=prefix,
                    action=trace[prefix]["action"],
                    position_q=q(trace[prefix]["position"]),
                    heading_rad=round(float(trace[prefix]["heading"]), 6),
                    pixels=[FRAME_SIZE, FRAME_SIZE + HEADER_PX],
                )
                media.append(record)
                frame_records[frame_id] = record

            global_offsets = uniform_indices(len(timeline), GLOBAL_FRAMES)
            global_prefixes = [timeline[value] for value in global_offsets]
            global_ids = ["P%04d" % value for value in global_prefixes]
            global_paths = [ROOT / frame_records[value]["path"]
                            for value in global_ids]
            global_path = board_dir / "GLOBAL.jpg"
            global_image = contact_sheet(global_paths, global_ids)
            if not cv2.imwrite(str(global_path), global_image,
                               [int(cv2.IMWRITE_JPEG_QUALITY),
                                JPEG_QUALITY]):
                raise RuntimeError("failed to write global storyboard")
            global_record = media_record(
                global_path, "global_route_storyboard",
                episode_id=episode_id, frame_ids=global_ids,
                pixels=[global_image.shape[1], global_image.shape[0]],
            )
            media.append(global_record)

            chunks = []
            for chunk_index, (start, end) in enumerate(
                    chunk_ranges(len(timeline))):
                chunk_id = "C%02d" % chunk_index
                prefixes = timeline[start:end]
                frame_ids = ["P%04d" % value for value in prefixes]
                paths = [ROOT / frame_records[value]["path"]
                         for value in frame_ids]
                board_path = board_dir / (chunk_id + ".jpg")
                board = contact_sheet(paths, frame_ids)
                if not cv2.imwrite(str(board_path), board,
                                   [int(cv2.IMWRITE_JPEG_QUALITY),
                                    JPEG_QUALITY]):
                    raise RuntimeError("failed to write chunk storyboard")
                board_record = media_record(
                    board_path, "chronological_chunk_storyboard",
                    episode_id=episode_id, chunk_id=chunk_id,
                    frame_ids=frame_ids,
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
            event_records.append({
                "trajectory_id": str(episode["trajectory_id"]),
                "episode_id": episode_id,
                "scene_id": scene,
                "instruction_id": str(instruction["instruction_id"]),
                "language": instruction["language"],
                "instruction_text": text,
                "instruction_sha256": sha256_text(text),
                "deterministic_segments": instruction_segments(text),
                "trace_length": len(trace),
                "timeline_frame_ids": ["P%04d" % value
                                       for value in timeline],
                "timeline_prefix_indices": timeline,
                "timeline_sampling": {
                    "all_30_degree_turn_prefixes_retained": True,
                    "move_sample_max_m": MOVE_SAMPLE_M,
                    "first_and_last_retained": True,
                },
                "global_storyboard": {
                    "path": global_record["path"],
                    "sha256": global_record["sha256"],
                    "frame_ids": global_ids,
                },
                "chunks": chunks,
                "trace_pose_action_sha256": stable_sha([
                    {
                        "prefix_index": index,
                        "position": q(row["position"]),
                        "heading_rad": round(float(row["heading"]), 6),
                        "action": row["action"],
                    }
                    for index, row in enumerate(trace)
                ]),
                "legacy_target_fields_in_model_input": False,
                "mllm_output": None,
            })
        finally:
            sim.close()

    output = {
        "manifest": "MF2-CR5 blinded full-trajectory hindsight preflight",
        "revision": "cr5-hindsight-preflight-inputs/2-explicit-json-shape",
        "status": "READY_FOR_DRY_RUN",
        "source_scope": "RxR train only",
        "rxr_train": {
            "path": str(RXR_TRAIN.relative_to(ROOT)),
            "bytes": RXR_TRAIN.stat().st_size,
            "sha256": sha256_file(RXR_TRAIN),
        },
        "contract": {
            "prompt_path": str(PROMPT.relative_to(ROOT)),
            "prompt_sha256": sha256_file(PROMPT),
            "schema_path": str(SCHEMA.relative_to(ROOT)),
            "schema_sha256": sha256_file(SCHEMA),
        },
        "rendering": {
            "habitat_sim_source": "project-local pinned Habitat-Sim v0.1.7",
            "gpu_device_id": GPU_DEVICE,
            "rgb_pixels": [FRAME_SIZE, FRAME_SIZE],
            "frame_header_pixels": HEADER_PX,
            "hfov_deg": 63.0,
            "sensor_height_m": 0.88,
            "jpeg_quality": JPEG_QUALITY,
        },
        "chunking": {
            "max_frames": CHUNK_FRAMES,
            "minimum_overlap_frames": CHUNK_OVERLAP,
            "global_storyboard_max_frames": GLOBAL_FRAMES,
        },
        "episode_count": len(event_records),
        "episodes": event_records,
        "media_manifest": sorted(media, key=lambda value: value["path"]),
        "media_file_count": len(media),
        "media_total_bytes": sum(value["bytes"] for value in media),
        "network_calls_made": 0,
        "future_frames_are_offline_annotation_only": True,
        "online_prefix_causality_verified": False,
        "branch_labels_created": 0,
        "training_authorized": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "episodes": output["episode_count"],
        "chunks": sum(len(row["chunks"]) for row in event_records),
        "media_files": output["media_file_count"],
        "media_total_bytes": output["media_total_bytes"],
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
