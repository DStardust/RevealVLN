#!/usr/bin/env python3
"""Build private, ordered-image inputs for train-only clause grounding.

The output contains no MLLM or human judgment.  Each event receives a
chronological sequence made from a fixed global sample plus a dense window
around P/D1/D2/D3.  Instruction segments are exact source substrings split by
punctuation; they are proposals for alignment, not official RxR annotations.
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
HABSIM = ROOT / "third_party/habitat-sim"
for item in (str(SCRIPTS), str(HABSIM)):
    if item not in sys.path:
        sys.path.insert(0, item)

from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402


PACKET = ROOT / ("artifacts/phase0/phase0c_language_review_35_v2_localmap/"
                 "PHASE0C_LANGUAGE_REVIEW_35_V2_LOCALMAP.json")
EXPECTED_PACKET_SHA = \
    "3c3f650fa26ceb1d948614e3c1eb6800dca85504e1cad7690c52ab1294424c7c"
RXR_TRAIN = ROOT / ("third_party/ETP-R1/data/datasets/"
                    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz")
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_clause_grounding_mllm"
FRAME_DIR = OUT_DIR / "private_frames"
OUT = OUT_DIR / "MLLM_CLAUSE_GROUNDING_INPUTS.json"
PROMPT_OUT = OUT_DIR / "MLLM_ALIGNMENT_SYSTEM_PROMPT.md"
FRAME_SIZE = 448
GLOBAL_SAMPLE_COUNT = 20
LOCAL_RADIUS = 6
JPEG_QUALITY = 90
MODEL = "qwen3.8-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALLOWED_LANGUAGES = {"en-IN", "en-US"}

SYSTEM_PROMPT = """# RxR local clause-grounding proposal prompt

You are an offline alignment proposer, not a ground-truth annotator. You will
receive one complete RxR instruction, deterministic exact-substring segments,
and a chronological list of first-person route images. Each image visibly
contains its immutable low-level prefix ID. P, D1, D2 and D3 identify the
local causal observation window whose matching instruction segment must be
proposed.

Select the smallest one to three adjacent segment IDs that describe the
physical route portion shown around P--D3. Use the global frames only to locate
that portion within the complete route. Never rewrite a segment, never claim
official word-to-waypoint alignment, and never decide whether a RevealEvent is
valid. If more than one alignment remains plausible, return
MULTIPLE_PLAUSIBLE. If no segment matches, return NO_MATCH. If the images are
insufficient, return INSUFFICIENT_VISUAL_EVIDENCE. Do not force a match.

Return exactly one JSON object with these keys:
status, selected_segment_ids, alternative_segment_groups,
evidence_frame_ids, confidence, rationale.

status must be one of UNIQUE_MATCH, MULTIPLE_PLAUSIBLE, NO_MATCH,
INSUFFICIENT_VISUAL_EVIDENCE. Segment and frame IDs must come from the input.
confidence must be a number from 0 through 1. rationale must be concise and
must not contain hidden chain-of-thought.
"""


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


def q(values, places=6):
    return [round(float(value), places) for value in values]


def instruction_segments(source: str):
    """Split at source punctuation while retaining exact character spans."""
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
        raise RuntimeError("exact-substring instruction segmentation failed")
    return segments


def uniform_indices(length: int, count: int):
    if length <= 0:
        return []
    count = min(length, count)
    if count == 1:
        return [0]
    return sorted({int(round(index * (length - 1) / (count - 1)))
                   for index in range(count)})


def event_indices(trace_length: int, prefixes):
    global_indices = uniform_indices(trace_length, GLOBAL_SAMPLE_COUNT)
    low = max(0, int(prefixes["pre_reveal"]) - LOCAL_RADIUS)
    high = min(trace_length - 1, int(prefixes["d3"]) + LOCAL_RADIUS)
    dense = list(range(low, high + 1))
    fixed = [int(prefixes[key]) for key in
             ("pre_reveal", "d1", "d2", "d3")]
    selected = sorted(set(global_indices + dense + fixed))
    if any(index < 0 or index >= trace_length for index in selected):
        raise RuntimeError("selected trace index outside trace")
    return selected


def build_sim(scene: str):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(MP3D / scene / (scene + ".glb"))
    sim_cfg.gpu_device_id = int(os.environ.get("PHASE0C_MLLM_GPU", "0"))
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
    agent_state.position = np.asarray(sim.pathfinder.snap_point(
        state["position"]), dtype="float32")
    agent_state.rotation = Rotation.from_rotvec(
        [0.0, float(state["heading"]), 0.0]).as_quat()
    sim.get_agent(0).set_state(agent_state, True)
    rgb = sim.get_sensor_observations()["rgb"][..., :3].copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (FRAME_SIZE - 1, 42), (0, 0, 0), -1)
    label = "P%04d | %s" % (prefix, state["action"])
    cv2.putText(bgr, label, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def frame_id(episode_id: str, prefix: int) -> str:
    return "EP%s_P%04d" % (episode_id, prefix)


def main() -> int:
    if sha256_file(PACKET) != EXPECTED_PACKET_SHA:
        raise SystemExit("fixed local-map packet SHA drift")
    packet = json.loads(PACKET.read_text())
    rows = packet.get("rows", [])
    if (packet.get("status") != "PASS_PENDING_HUMAN_REVIEW"
            or len(rows) != 35
            or {row["language"] for row in rows} - ALLOWED_LANGUAGES):
        raise SystemExit("unexpected source packet state")
    wanted = {str(row["episode_id"]) for row in rows}
    with gzip.open(RXR_TRAIN, "rt") as handle:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(handle)["episodes"]
                    if str(item["episode_id"]) in wanted}
    if set(episodes) != wanted:
        raise SystemExit("RxR train episode closure failed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_OUT.write_text(SYSTEM_PROMPT)
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], []).append(row)

    event_records, media = [], {}
    trace_metadata = {}
    for scene, scene_rows in sorted(by_scene.items()):
        sim = build_sim(scene)
        try:
            by_episode = {}
            for row in scene_rows:
                by_episode.setdefault(str(row["episode_id"]), []).append(row)
            for episode_id, episode_rows in sorted(by_episode.items()):
                trace = build_lowlevel_trace(
                    sim.pathfinder, episodes[episode_id])
                if not trace:
                    raise RuntimeError("empty low-level trace: " + episode_id)
                selected_by_event = {
                    row["event_id"]: event_indices(
                        len(trace), row["causal_prefixes"])
                    for row in episode_rows
                }
                union = sorted(set(index for values in selected_by_event.values()
                                   for index in values))
                episode_dir = FRAME_DIR / ("ep" + episode_id)
                episode_dir.mkdir(parents=True, exist_ok=True)
                for prefix in union:
                    identifier = frame_id(episode_id, prefix)
                    path = episode_dir / ("p%04d.jpg" % prefix)
                    image = render(sim, trace[prefix], prefix)
                    if not cv2.imwrite(str(path), image,
                                       [int(cv2.IMWRITE_JPEG_QUALITY),
                                        JPEG_QUALITY]):
                        raise RuntimeError("failed to write route frame")
                    media[identifier] = {
                        "frame_id": identifier,
                        "episode_id": episode_id,
                        "prefix_index": prefix,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                        "pixels": [FRAME_SIZE, FRAME_SIZE],
                        "action": trace[prefix]["action"],
                        "position_q": q(trace[prefix]["position"]),
                        "heading_rad": round(float(
                            trace[prefix]["heading"]), 6),
                    }
                trace_metadata[episode_id] = {
                    "scene_id": scene,
                    "trace_length": len(trace),
                    "sampled_union_count": len(union),
                    "trace_pose_action_sha256": sha256_text(json.dumps([
                        {"position": q(item["position"]),
                         "heading": round(float(item["heading"]), 6),
                         "action": item["action"]}
                        for item in trace], sort_keys=True,
                        separators=(",", ":"))),
                }
                for row in episode_rows:
                    source = row["instruction_text_for_private_review"]
                    segments = instruction_segments(source)
                    indices = selected_by_event[row["event_id"]]
                    roles = {
                        key: frame_id(episode_id, int(value))
                        for key, value in row["causal_prefixes"].items()
                    }
                    event_records.append({
                        "row_order": int(row["row_order"]),
                        "event_id": row["event_id"],
                        "episode_id": episode_id,
                        "scene_id": scene,
                        "language": row["language"],
                        "reference_turn_index": row["reference_turn_index"],
                        "instruction_text": source,
                        "instruction_sha256": row["instruction_sha256"],
                        "deterministic_segments": segments,
                        "segmentation_contract": {
                            "revision": "exact-punctuation-segments/1",
                            "official_rxr_alignment": False,
                            "model_may_rewrite_segments": False,
                            "maximum_adjacent_selection": 3,
                        },
                        "sequence_frame_ids": [
                            frame_id(episode_id, value) for value in indices],
                        "causal_frame_roles": roles,
                        "sequence_is_chronological": True,
                        "mllm_proposal": None,
                        "human_judgment": None,
                    })
        finally:
            sim.close()

    event_records.sort(key=lambda item: item["row_order"])
    media_records = [media[key] for key in sorted(media)]
    if (len(event_records) != 35
            or any(item["mllm_proposal"] is not None
                   or item["human_judgment"] is not None
                   for item in event_records)):
        raise SystemExit("non-fabrication/cardinality gate failed")
    output = {
        "manifest": "MF2-CR4 private MLLM clause-grounding inputs",
        "revision": "phase0c-mllm-clause-inputs/1-ordered-images",
        "status": "READY_FOR_MLLM_PROPOSAL_UNCALLED",
        "source_scope": "RxR train only",
        "source_packet": {
            "path": str(PACKET.relative_to(ROOT)),
            "sha256": EXPECTED_PACKET_SHA,
        },
        "rxr_train": {
            "path": str(RXR_TRAIN.relative_to(ROOT)),
            "sha256": sha256_file(RXR_TRAIN),
        },
        "model_request_contract": {
            "provider": "DashScope OpenAI-compatible",
            "base_url": BASE_URL,
            "model": MODEL,
            "input": "chronological image_url data-URI sequence plus text",
            "temperature": 0,
            "system_prompt_path": str(PROMPT_OUT.relative_to(ROOT)),
            "system_prompt_sha256": sha256_file(PROMPT_OUT),
            "proposal_is_ground_truth": False,
        },
        "sampling": {
            "global_uniform_frames": GLOBAL_SAMPLE_COUNT,
            "dense_radius_around_causal_window": LOCAL_RADIUS,
            "P_D1_D2_D3_forced": True,
            "frame_pixels": [FRAME_SIZE, FRAME_SIZE],
            "jpeg_quality": JPEG_QUALITY,
        },
        "event_count": len(event_records),
        "episode_count": len(trace_metadata),
        "scene_count": len({item["scene_id"] for item in event_records}),
        "events": event_records,
        "episodes": trace_metadata,
        "media_manifest": media_records,
        "media_file_count": len(media_records),
        "media_total_bytes": sum(item["bytes"] for item in media_records),
        "network_calls_made": 0,
        "private_distribution_authorized": False,
        "training_authorized": False,
    }
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": output["status"],
        "events": output["event_count"],
        "episodes": output["episode_count"],
        "scenes": output["scene_count"],
        "media_files": output["media_file_count"],
        "media_total_bytes": output["media_total_bytes"],
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
