#!/usr/bin/env python3
"""Render 3 positions x 12 headings for 35 CR5 hindsight candidates."""

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

from phase0c_oracle_lowlevel_probe import (  # noqa: E402
    absolute_heading,
    build_lowlevel_trace,
)


AGGREGATED = ROOT / (
    "artifacts/phase0/phase0c_cr5_preflight/hindsight_locator/"
    "CR5_HINDSIGHT_PREFLIGHT_AGGREGATED.json"
)
LOCATOR_INPUT = AGGREGATED.with_name("CR5_HINDSIGHT_PREFLIGHT_INPUTS_V2.json")
EXPECTED_AGGREGATED_SHA = "c7ced3d55c08fe8380ac9e8fde009bd329f822f9299ad95ec6d8aab848a104c2"
EXPECTED_LOCATOR_INPUT_SHA = "939945e2a21fb571aeec7c7f8914be6873bf73ef08b7e7b12d3e2d94ac9d999d"
RXR_TRAIN = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
CONTRACT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_contract"
PROMPT = CONTRACT_DIR / "CR5_MLLM_BRANCH_PROPOSAL_PROMPT_V2.md"
SCHEMA = CONTRACT_DIR / "CR5_MLLM_BRANCH_PROPOSAL_SCHEMA.json"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_preflight/multiview_branch"
MEDIA_DIR = OUT_DIR / "private_media"
OUT = OUT_DIR / "CR5_MULTIVIEW_PREFLIGHT_INPUTS.json"
FRAME_SIZE = 448
HEADER_PX = 52
JPEG_QUALITY = 92
POSITION_DISTANCE_M = 1.0
PANORAMA_HEADINGS = 12
PANORAMA_STEP_DEG = 30.0
CONTEXT_RADIUS = 4
GPU_DEVICE = int(os.environ.get("CR5_MULTIVIEW_GPU", "1"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


def q(values, places: int = 6):
    return [round(float(value), places) for value in values]


def prefix_id(index: int) -> str:
    return "P%04d" % index


def prefix_number(value: str) -> int:
    if not value.startswith("P") or not value[1:].isdigit():
        raise ValueError("bad prefix ID")
    return int(value[1:])


def relative_yaw(index: int) -> float:
    value = index * PANORAMA_STEP_DEG
    return value if value <= 180 else value - 360


def position_prefix(trace, center: int, direction: int,
                    target_distance: float):
    current = center
    travelled = 0.0
    while 0 <= current + direction < len(trace):
        next_index = current + direction
        travelled += distance(trace[current]["position"],
                              trace[next_index]["position"])
        current = next_index
        if travelled + 1e-9 >= target_distance:
            break
    return current, travelled


def route_forward_heading(trace, index: int) -> float:
    current = trace[index]["position"]
    for candidate in range(index + 1, len(trace)):
        if distance(current, trace[candidate]["position"]) > 1e-4:
            return absolute_heading(current, trace[candidate]["position"])
    for candidate in range(index - 1, -1, -1):
        if distance(current, trace[candidate]["position"]) > 1e-4:
            return absolute_heading(trace[candidate]["position"], current)
    return float(trace[index]["heading"])


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


def render(sim, position, heading: float, label: str):
    import habitat_sim
    from scipy.spatial.transform import Rotation

    state = habitat_sim.AgentState()
    state.position = np.asarray(sim.pathfinder.snap_point(position),
                                dtype="float32")
    state.rotation = Rotation.from_rotvec([0.0, heading, 0.0]).as_quat()
    sim.get_agent(0).set_state(state, True)
    rgb = sim.get_sensor_observations()["rgb"][..., :3].copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    canvas = np.zeros((FRAME_SIZE + HEADER_PX, FRAME_SIZE, 3),
                      dtype=np.uint8)
    canvas[HEADER_PX:] = bgr
    cv2.putText(canvas, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def contact_sheet(paths, labels, columns: int = 4, tile: int = 320):
    rows = int(math.ceil(len(paths) / columns))
    sheet = np.full((rows * tile, columns * tile, 3), 24,
                    dtype=np.uint8)
    for index, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to decode panorama view")
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        y, x = row * tile, column * tile
        sheet[y:y + tile, x:x + tile] = image
        cv2.rectangle(sheet, (x, y), (x + tile - 1, y + tile - 1),
                      (130, 130, 130), 1)
        cv2.putText(sheet, label, (x + 8, y + tile - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
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
    if sha256_file(AGGREGATED) != EXPECTED_AGGREGATED_SHA:
        raise SystemExit("aggregate SHA drift")
    if sha256_file(LOCATOR_INPUT) != EXPECTED_LOCATOR_INPUT_SHA:
        raise SystemExit("locator input SHA drift")
    if not PROMPT.is_file() or not SCHEMA.is_file():
        raise SystemExit("branch proposer contract missing")
    aggregate = json.loads(AGGREGATED.read_text())
    locator = json.loads(LOCATOR_INPUT.read_text())
    locator_episodes = {row["episode_id"]: row
                        for row in locator["episodes"]}
    locator_media = {row["frame_id"] + "@" + row["episode_id"]: row
                     for row in locator["media_manifest"]
                     if row["kind"] == "chronological_route_frame"}
    wanted = set(locator_episodes)
    with gzip.open(RXR_TRAIN, "rt") as handle:
        episodes = {str(row["episode_id"]): row
                    for row in json.load(handle)["episodes"]
                    if str(row["episode_id"]) in wanted}
    if set(episodes) != wanted:
        raise SystemExit("RxR episode closure failure")

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    by_scene = {}
    for candidate in aggregate["candidates"]:
        by_scene.setdefault(candidate["scene_id"], []).append(candidate)
    output_events, media = [], []
    for scene, candidates in sorted(by_scene.items()):
        sim = build_sim(scene)
        try:
            trace_cache = {}
            for candidate in sorted(candidates,
                                    key=lambda value: value[
                                        "hindsight_candidate_id"]):
                episode_id = candidate["episode_id"]
                if episode_id not in trace_cache:
                    trace_cache[episode_id] = build_lowlevel_trace(
                        sim.pathfinder, episodes[episode_id])
                trace = trace_cache[episode_id]
                center = prefix_number(candidate["interval"][
                    "representative_center_frame_id"])
                if not 0 <= center < len(trace):
                    raise RuntimeError("candidate center outside trace")
                a_prefix, a_distance = position_prefix(
                    trace, center, -1, POSITION_DISTANCE_M)
                d_prefix, d_distance = position_prefix(
                    trace, center, 1, POSITION_DISTANCE_M)
                roles = [("A", a_prefix, a_distance),
                         ("Q", center, 0.0),
                         ("D", d_prefix, d_distance)]
                event_dir = MEDIA_DIR / candidate["hindsight_candidate_id"]
                event_dir.mkdir(parents=True, exist_ok=True)
                role_records = {}
                for role, trace_prefix, achieved in roles:
                    base_heading = route_forward_heading(trace, trace_prefix)
                    view_records, view_paths, view_labels = [], [], []
                    for view_index in range(PANORAMA_HEADINGS):
                        view_id = "%s_V%02d" % (role, view_index)
                        offset = relative_yaw(view_index)
                        heading = (base_heading + math.radians(offset)) % (
                            2 * math.pi)
                        direction = "LEFT" if offset > 0 else (
                            "RIGHT" if offset < 0 else "FORWARD")
                        label = "%s | %+d deg %s" % (
                            view_id, int(offset), direction)
                        path = event_dir / (view_id + ".jpg")
                        image = render(sim, trace[trace_prefix]["position"],
                                       heading, label)
                        if not cv2.imwrite(
                                str(path), image,
                                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
                            raise RuntimeError("failed to write panorama view")
                        record = media_record(
                            path, "panorama_view",
                            event_id=candidate["hindsight_candidate_id"],
                            role=role, view_id=view_id,
                            trace_prefix=trace_prefix,
                            relative_yaw_deg=offset,
                            habitat_heading_rad=round(heading, 6),
                            position_q=q(trace[trace_prefix]["position"]),
                            pixels=[FRAME_SIZE, FRAME_SIZE + HEADER_PX],
                        )
                        media.append(record)
                        view_records.append(record)
                        view_paths.append(path)
                        view_labels.append(view_id)
                    board_path = event_dir / (role + "_PANORAMA.jpg")
                    board = contact_sheet(view_paths, view_labels)
                    if not cv2.imwrite(
                            str(board_path), board,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
                        raise RuntimeError("failed to write panorama board")
                    board_record = media_record(
                        board_path, "panorama_contact_sheet",
                        event_id=candidate["hindsight_candidate_id"],
                        role=role,
                        view_ids=[row["view_id"] for row in view_records],
                        pixels=[board.shape[1], board.shape[0]],
                    )
                    media.append(board_record)
                    role_records[role] = {
                        "trace_prefix": trace_prefix,
                        "frame_id": prefix_id(trace_prefix),
                        "requested_route_distance_m": POSITION_DISTANCE_M
                            if role != "Q" else 0.0,
                        "achieved_route_distance_m": round(achieved, 6),
                        "position_q": q(trace[trace_prefix]["position"]),
                        "route_forward_heading_rad": round(base_heading, 6),
                        "views": view_records,
                        "contact_sheet": board_record,
                    }

                timeline = locator_episodes[episode_id][
                    "timeline_prefix_indices"]
                nearest_offset = min(range(len(timeline)),
                                     key=lambda index: abs(timeline[index] -
                                                           center))
                low = max(0, nearest_offset - CONTEXT_RADIUS)
                high = min(len(timeline), nearest_offset + CONTEXT_RADIUS + 1)
                context_prefixes = timeline[low:high]
                support = [prefix_number(value) for value in
                           candidate["interval"]["supporting_frame_ids"]]
                context_prefixes = sorted(set(context_prefixes + support))
                context_records = []
                for value in context_prefixes:
                    key = prefix_id(value) + "@" + episode_id
                    if key not in locator_media:
                        # Supporting frames are always drawn from sampled
                        # locator frames; this also fails closed on drift.
                        raise RuntimeError("missing locator context frame: " + key)
                    context_records.append(locator_media[key])

                source = locator_episodes[episode_id]
                output_events.append({
                    "event_id": candidate["hindsight_candidate_id"],
                    "episode_id": episode_id,
                    "trajectory_id": candidate["trajectory_id"],
                    "scene_id": scene,
                    "instruction_id": candidate["instruction_id"],
                    "language": source["language"],
                    "instruction_text": source["instruction_text"],
                    "instruction_sha256": source["instruction_sha256"],
                    "deterministic_segments": source[
                        "deterministic_segments"],
                    "candidate_interval": candidate["interval"],
                    "candidate_center_source": "full_trajectory_hindsight_"
                                               "locator_merged",
                    "locator_free_text_in_model_input": False,
                    "legacy_bt_in_model_input": False,
                    "positions": role_records,
                    "chronological_context_frames": context_records,
                    "mllm_branch_proposal": None,
                    "geometry_verified": False,
                    "training_label": False,
                })
        finally:
            sim.close()

    output_events.sort(key=lambda value: value["event_id"])
    output = {
        "manifest": "MF2-CR5 event-level multi-view preflight inputs",
        "revision": "cr5-multiview-preflight-inputs/1",
        "status": "READY_FOR_VALIDATION",
        "source_scope": "RxR train only",
        "aggregated_source": {
            "path": str(AGGREGATED.relative_to(ROOT)),
            "sha256": EXPECTED_AGGREGATED_SHA,
        },
        "locator_input_source": {
            "path": str(LOCATOR_INPUT.relative_to(ROOT)),
            "sha256": EXPECTED_LOCATOR_INPUT_SHA,
        },
        "contract": {
            "prompt_path": str(PROMPT.relative_to(ROOT)),
            "prompt_sha256": sha256_file(PROMPT),
            "schema_path": str(SCHEMA.relative_to(ROOT)),
            "schema_sha256": sha256_file(SCHEMA),
        },
        "rendering": {
            "gpu_device_id": GPU_DEVICE,
            "positions": ["A", "Q", "D"],
            "requested_position_spacing_m": POSITION_DISTANCE_M,
            "headings_per_position": PANORAMA_HEADINGS,
            "heading_step_deg": PANORAMA_STEP_DEG,
            "positive_habitat_yaw_is_left": True,
            "v00_is_local_reference_route_forward": True,
            "rgb_pixels": [FRAME_SIZE, FRAME_SIZE],
            "header_pixels": HEADER_PX,
            "hfov_deg": 63.0,
            "sensor_height_m": 0.88,
            "jpeg_quality": JPEG_QUALITY,
        },
        "event_count": len(output_events),
        "events": output_events,
        "media_manifest": sorted(media, key=lambda value: value["path"]),
        "media_file_count": len(media),
        "media_total_bytes": sum(row["bytes"] for row in media),
        "network_calls_made": 0,
        "branch_labels_created": 0,
        "geometry_verified_candidates": 0,
        "future_frames_are_offline_annotation_only": True,
        "training_authorized": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "events": output["event_count"],
        "panorama_views": sum(
            len(role["views"]) for event in output_events
            for role in event["positions"].values()),
        "media_files": output["media_file_count"],
        "media_total_bytes": output["media_total_bytes"],
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
