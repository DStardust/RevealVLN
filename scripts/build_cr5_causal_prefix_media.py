#!/usr/bin/env python3
"""Render exact 63-degree front views for CR5 causal-language adjudication."""

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
for value in (SCRIPTS, HABSIM):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402


ANALYSIS = ROOT / (
    "artifacts/phase0/phase0c_cr5_causal_gate/"
    "CR5_CAUSAL_CANDIDATE_ANALYSIS.json"
)
EXPECTED_ANALYSIS_SHA256 = (
    "df4a5cd387b721b4b16a8285e376fa387458f6c1d4028505f47ded9cf9fed5c1"
)
RXR_TRAIN = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_causal_gate"
MEDIA_DIR = OUT_DIR / "private_media"
OUT = OUT_DIR / "CR5_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
FRAME_SIZE = 448
HEADER_PX = 48
JPEG_QUALITY = 92
HFOV_DEG = 63.0
SENSOR_HEIGHT_M = 0.88
GPU_DEVICE = int(os.environ.get("CR5_CAUSAL_MEDIA_GPU", "1"))
HISTORY_PADDING = 2
K = 3
EXPECTED_CAUSAL_COUNT = 7
OUTPUT_REVISION = "cr5-causal-prefix-media/1"
USE_ALL_BRANCHES = False
REUSE_MEDIA_MANIFEST = None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def qpoint(values, places: int = 6):
    return [round(float(value), places) for value in values]


def scene_name(episode) -> str:
    parts = Path(episode["scene_id"]).parts
    if len(parts) != 3 or parts[0] != "mp3d" or parts[2] != parts[1] + ".glb":
        raise RuntimeError("unexpected scene path")
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
    sensor.position = [0.0, SENSOR_HEIGHT_M, 0.0]
    sensor.hfov = HFOV_DEG
    agent = habitat_sim.AgentConfiguration()
    agent.height = SENSOR_HEIGHT_M
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
    canvas = np.zeros((FRAME_SIZE + HEADER_PX, FRAME_SIZE, 3), dtype=np.uint8)
    canvas[HEADER_PX:] = bgr
    cv2.putText(canvas, "P%04d | %s | 63deg FRONT" % (
        prefix, state["action"]), (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
        0.66, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(part, path)


def main() -> int:
    if not ANALYSIS.is_file() or ANALYSIS.is_symlink() \
            or sha256_file(ANALYSIS) != EXPECTED_ANALYSIS_SHA256:
        raise SystemExit("candidate analysis drift")
    analysis = json.loads(ANALYSIS.read_text())
    causal = [row for row in analysis["events"] if row["status"] ==
              "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"]
    if len(causal) != EXPECTED_CAUSAL_COUNT:
        raise SystemExit("unexpected language-gate event count")
    episode_ids = {row["episode_id"] for row in causal}
    with gzip.open(RXR_TRAIN, "rt") as handle:
        episodes = {str(row["episode_id"]): row
                    for row in json.load(handle)["episodes"]
                    if str(row["episode_id"]) in episode_ids}
    if set(episodes) != episode_ids:
        raise SystemExit("episode closure failure")

    event_ranges = {}
    wanted_by_episode = {episode_id: set() for episode_id in episode_ids}
    for event in causal:
        if USE_ALL_BRANCHES:
            competitors = [branch_id for branch_id in
                           event["candidate_branch_ids"]
                           if branch_id != event["target_branch_id"]]
            confirmations = [event[
                "branch_established_at_confirmation_prefix"][branch_id]
                for branch_id in competitors]
            if not competitors or any(value is None for value in confirmations):
                raise RuntimeError(
                    "causal-ready event lacks complete competition history"
                )
            competition_start = min(value - K + 1 for value in confirmations)
        else:
            alternative = event["alternative_branch_id"]
            alt_confirmation = event[
                "branch_established_at_confirmation_prefix"][alternative]
            if alt_confirmation is None:
                raise RuntimeError("causal-ready event has no alternative history")
            competition_start = alt_confirmation - K + 1
        ready_start = min(span[0] for span in event[
            "stable_geometric_ready_runs"])
        history_start = max(
            0, min(competition_start, ready_start) - HISTORY_PADDING
        )
        end = event["D_prefix"]
        event_ranges[event["event_id"]] = {
            "episode_id": event["episode_id"],
            "history_start_prefix": history_start,
            "history_end_prefix": end,
            "geometric_ready_prefixes": [
                prefix for start, stop in event["stable_geometric_ready_runs"]
                for prefix in range(start, stop + 1)
            ],
        }
        wanted_by_episode[event["episode_id"]].update(
            range(history_start, end + 1))

    reusable = {}
    if REUSE_MEDIA_MANIFEST is not None:
        reuse_path = Path(REUSE_MEDIA_MANIFEST)
        if not reuse_path.is_file() or reuse_path.is_symlink():
            raise RuntimeError("reuse media manifest is unsafe")
        reuse_value = json.loads(reuse_path.read_text())
        rendering = reuse_value.get("rendering", {})
        if (rendering.get("hfov_deg") != HFOV_DEG
                or rendering.get("frame_pixels") != [FRAME_SIZE, FRAME_SIZE]
                or rendering.get("header_pixels") != HEADER_PX):
            raise RuntimeError("reuse rendering contract mismatch")
        for record in reuse_value["media_manifest"]:
            path = ROOT / record["path"]
            if (not path.is_file() or path.is_symlink()
                    or ROOT.resolve() not in path.resolve().parents
                    or path.stat().st_size != record["bytes"]
                    or sha256_file(path) != record["sha256"]):
                raise RuntimeError("unsafe or drifted reusable causal media")
            reusable[(record["episode_id"], record["prefix_index"])] = record

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    media = []
    reused_count = 0
    rendered_count = 0
    episodes_by_scene = {}
    for episode_id in episode_ids:
        episodes_by_scene.setdefault(
            scene_name(episodes[episode_id]), []).append(episode_id)
    for scene in sorted(episodes_by_scene):
        sim = build_sim(scene)
        try:
            for episode_id in sorted(episodes_by_scene[scene]):
                episode = episodes[episode_id]
                trace = build_lowlevel_trace(sim.pathfinder, episode)
                for prefix in sorted(wanted_by_episode[episode_id]):
                    if not 0 <= prefix < len(trace):
                        raise RuntimeError("prefix outside trace")
                    prior = reusable.get((episode_id, prefix))
                    if prior is not None:
                        media.append(prior)
                        reused_count += 1
                        continue
                    episode_dir = MEDIA_DIR / ("ep" + episode_id)
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    path = episode_dir / ("P%04d.jpg" % prefix)
                    image = render(sim, trace[prefix], prefix)
                    part = path.with_name(path.stem + ".part.jpg")
                    if not cv2.imwrite(str(part), image,
                                       [int(cv2.IMWRITE_JPEG_QUALITY),
                                        JPEG_QUALITY]):
                        raise RuntimeError("failed to write causal frame")
                    os.replace(part, path)
                    media.append({
                        "episode_id": episode_id,
                        "scene_id": scene,
                        "prefix_index": prefix,
                        "frame_id": "P%04d" % prefix,
                        "action": trace[prefix]["action"],
                        "position_q": qpoint(trace[prefix]["position"]),
                        "heading_rad": round(
                            float(trace[prefix]["heading"]), 8),
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "pixels": [FRAME_SIZE, FRAME_SIZE + HEADER_PX],
                        "hfov_deg": HFOV_DEG,
                        "view": "current ego-forward only",
                    })
                    rendered_count += 1
        finally:
            sim.close()

    output = {
        "revision": OUTPUT_REVISION,
        "status": "READY_FOR_PREFIX_LANGUAGE_GATE",
        "source": {"path": str(ANALYSIS.relative_to(ROOT)),
                   "sha256": sha256_file(ANALYSIS)},
        "source_scope": "RxR-train only",
        "event_count": len(causal),
        "episode_count": len(episode_ids),
        "event_ranges": event_ranges,
        "rendering": {
            "habitat_sim": "project-local pinned 0.1.7",
            "gpu_device_id": GPU_DEVICE,
            "hfov_deg": HFOV_DEG,
            "sensor_height_m": SENSOR_HEIGHT_M,
            "frame_pixels": [FRAME_SIZE, FRAME_SIZE],
            "header_pixels": HEADER_PX,
            "jpeg_quality": JPEG_QUALITY,
            "panorama_or_future_frame_in_online_request": False,
        },
        "media_count": len(media),
        "reused_media_count": reused_count,
        "rendered_media_count": rendered_count,
        "media_manifest": media,
        "network_calls_made": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "event_count": output["event_count"],
        "media_count": output["media_count"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
