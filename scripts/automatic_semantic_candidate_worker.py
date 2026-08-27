#!/usr/bin/env python3
"""One GPU shard of the MF2-CR1 automatic candidate/semantic audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import socket
import sys
from collections import defaultdict


ROOT = "/mnt/daiyang/vla"
ETPR1 = os.path.join(ROOT, "third_party", "ETP-R1")
HABLAB = os.path.join(ROOT, "third_party", "habitat-lab")
HABSIM = os.environ.get(
    "VLA_HABITAT_SIM_ROOT", os.path.join(ROOT, "third_party", "habitat-sim"))
SCRIPTS = os.path.join(ROOT, "scripts")
for path in (ROOT, ETPR1, HABLAB, HABSIM, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)


AUDIT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                     "ORACLE_SEMANTIC_BRANCH_TRACK_AUDIT.json")
PROBE = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                     "PHASE0C_ORACLE_LOWLEVEL_PROBE.json")
RXR_TRAIN = os.path.join(
    ETPR1, "data", "datasets", "RxR_VLNCE_v0_enc_xlmr", "train",
    "train_guide.json.gz")
MP3D = os.path.join(ETPR1, "data", "scene_datasets", "mp3d")
FINAL_CKPT = os.path.join(
    ETPR1, "data", "logs", "checkpoints", "release_rxr_grpo", "store",
    "ckpt.iter1320.pth")
JOINT = os.path.join(
    ETPR1, "pretrained", "r2r_rxr_ce", "mlm.sap_habitat_depth", "store2",
    "model_step_367500.pt")
WAYPOINT = os.path.join(ETPR1, "data", "wp_pred",
                        "check_cwp_bestdist_hfov63")
EXPECTED_AUDIT_SHA = \
    "e4b570dc9cdbe317d28b57507f1f74b9a16f92c8350810beb6b0f4dacd9df6a4"
EXPECTED_PROBE_SHA = \
    "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac"
TARGET_TUBE_M = 1.0
PROGRESS_MIN = 0.05
SEPARATION_MARGIN_M = 0.25
K = 3


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def install_network_guard():
    state = {"attempts": 0}
    original = socket.socket.connect

    def guarded(sock, address):
        if getattr(sock, "family", None) in (socket.AF_INET, socket.AF_INET6):
            state["attempts"] += 1
            raise RuntimeError("automatic semantic worker network denied")
        return original(sock, address)

    def guarded_create(*args, **kwargs):
        state["attempts"] += 1
        raise RuntimeError("automatic semantic worker network denied")

    socket.socket.connect = guarded
    socket.create_connection = guarded_create
    return state


def segment_distance(point, start, end):
    vx, vz = end[0] - start[0], end[2] - start[2]
    wx, wz = point[0] - start[0], point[2] - start[2]
    denominator = vx * vx + vz * vz
    progress = max(0.0, min(1.0, (wx * vx + wz * vz) / denominator)) \
        if denominator > 1e-12 else 0.0
    qx, qz = start[0] + progress * vx, start[2] + progress * vz
    return math.hypot(point[0] - qx, point[2] - qz), progress


def candidate_position(position, heading, angle, distance):
    absolute = (float(heading) + float(angle)) % (2 * math.pi)
    return [float(position[0]) - float(distance) * math.sin(absolute),
            float(position[1]),
            float(position[2]) - float(distance) * math.cos(absolute)]


def q3(point):
    return [round(float(value), 3) for value in point]


def build_sim(scene):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.join(MP3D, scene, scene + ".glb")
    sim_cfg.gpu_device_id = 0
    sim_cfg.allow_sliding = False
    rgb = habitat_sim.SensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb.resolution = [224, 224]
    rgb.position = [0.0, 0.88, 0.0]
    rgb.hfov = 63.0
    depth = habitat_sim.SensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth.resolution = [256, 256]
    depth.position = [0.0, 0.88, 0.0]
    depth.hfov = 63.0
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = [rgb, depth]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg,
                                                           [agent_cfg]))
    navmesh = os.path.join(MP3D, scene, scene + ".navmesh")
    if not sim.pathfinder.load_nav_mesh(navmesh):
        sim.close()
        raise RuntimeError("navmesh load failed: " + navmesh)
    return sim


def set_state(sim, position, heading):
    import habitat_sim
    import numpy as np
    from scipy.spatial.transform import Rotation

    state = habitat_sim.AgentState()
    state.position = np.asarray(sim.pathfinder.snap_point(position),
                                dtype="float32")
    state.rotation = Rotation.from_rotvec(
        [0.0, float(heading), 0.0]).as_quat()
    sim.get_agent(0).set_state(state, True)


def make_observations(sim, device):
    import torch

    raw = sim.get_sensor_observations()
    rgb = torch.from_numpy(raw["rgb"][..., :3].copy()).unsqueeze(0).to(device)
    depth = torch.from_numpy(raw["depth"].copy()).unsqueeze(0).unsqueeze(
        -1).to(device)
    observations = {"rgb": rgb, "depth": depth}
    for angle in range(30, 360, 30):
        observations["rgb_%d" % angle] = torch.zeros_like(rgb)
        observations["depth_%d" % angle] = torch.zeros_like(depth)
    return observations


def build_models():
    import gym
    import numpy as np
    import torch
    from etpr1_compat import (configure_project_cache_env,
                             load_project_checkpoint,
                             require_matching_state_dict)
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.R1Policy import R1Policy
    from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM

    configure_project_cache_env()
    config = get_config("run_rxr/iter_train.yaml", [
        "MODEL.pretrained_path", JOINT, "GPU_NUMBERS", "1",
        "TORCH_GPU_ID", "0"])
    observation_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3),
                              dtype=np.uint8),
        "depth": gym.spaces.Box(low=0.0, high=1.0,
                                shape=(256, 256, 1), dtype=np.float32),
    })
    policy = R1Policy.from_config(config, observation_space,
                                  gym.spaces.Discrete(5))
    checkpoint = load_project_checkpoint(FINAL_CKPT, map_location="cpu")
    state = checkpoint["state_dict"]
    keys = [str(key) for key in state]
    if not keys or not all(key.startswith("net.module.") for key in keys):
        raise RuntimeError("unexpected final checkpoint prefix")
    normalized = {}
    for key, value in state.items():
        new_key = "net." + str(key)[len("net.module."):]
        if new_key in normalized:
            raise RuntimeError("checkpoint prefix collision")
        normalized[new_key] = value
    incompatible = policy.load_state_dict(normalized, strict=False)
    require_matching_state_dict(incompatible,
                                "automatic semantic final policy")
    predictor = BinaryDistPredictor_TRM(device=torch.device("cuda:0"))
    wp_checkpoint = load_project_checkpoint(WAYPOINT, map_location="cpu")
    predictor.load_state_dict(wp_checkpoint["predictor"]["state_dict"],
                              strict=True)
    policy.to(torch.device("cuda:0")).eval()
    predictor.to(torch.device("cuda:0")).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    del checkpoint, state, normalized, wp_checkpoint
    return policy, predictor, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")
    if not os.path.realpath(args.output).startswith(ROOT + os.sep):
        raise SystemExit("output outside workspace")
    if sha256_file(AUDIT) != EXPECTED_AUDIT_SHA or \
            sha256_file(PROBE) != EXPECTED_PROBE_SHA:
        raise SystemExit("semantic input SHA drift")
    network = install_network_guard()
    audit = json.load(open(AUDIT))
    probe = json.load(open(PROBE))
    admitted = [event for event in audit["events"] if event[
        "machine_geometric_semantic_status"] == "ADMITTED"]
    event_ids = {event["provisional_event_id"] for event in admitted}
    probe_events = {event["provisional_event_id"]: event
                    for event in probe["events"]
                    if event["provisional_event_id"] in event_ids}
    scenes = sorted({event["scene_id"] for event in admitted})
    shard_scenes = [scene for index, scene in enumerate(scenes)
                    if index % args.shard_count == args.shard_index]
    shard_events = [event for event in admitted
                    if event["scene_id"] in shard_scenes]
    episode_ids = {str(event["episode_id"]) for event in shard_events}
    with gzip.open(RXR_TRAIN, "rt") as fh:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(fh)["episodes"]
                    if str(item["episode_id"]) in episode_ids}

    import torch
    from phase0c_oracle_lowlevel_probe import build_lowlevel_trace
    from revealnav_cr1.causal_frontend import (apply_raw_view_mask,
                                               filter_waypoint_outputs)

    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    policy, predictor, config = build_models()
    device = torch.device("cuda:0")
    by_scene = defaultdict(list)
    for event in shard_events:
        by_scene[event["scene_id"]].append(event)
    results = []
    inference_cache = {}
    for scene in shard_scenes:
        sim = build_sim(scene)
        try:
            trace_cache = {}
            targets_by_episode = defaultdict(dict)
            for event in by_scene[scene]:
                target = event["target_exit_region"]
                targets_by_episode[str(event["episode_id"])][
                    event["provisional_event_id"]] = {
                        "start": target["directed_start_q"],
                        "end": target["directed_end_q"],
                        "branch_id": event["semantic_branch_id"],
                    }
            for event in by_scene[scene]:
                event_id = event["provisional_event_id"]
                episode_id = str(event["episode_id"])
                if episode_id not in trace_cache:
                    trace_cache[episode_id] = build_lowlevel_trace(
                        sim.pathfinder, episodes[episode_id])
                trace = trace_cache[episode_id]
                prefix_records, reasons = [], []
                for prefix in event["k_prefixes"]:
                    cache_key = (scene, episode_id, int(prefix))
                    if cache_key not in inference_cache:
                        state = trace[int(prefix)]
                        set_state(sim, state["position"], state["heading"])
                        observations = make_observations(sim, device)
                        mask = torch.zeros((1, 12), dtype=torch.bool,
                                           device=device)
                        mask[:, 0] = True
                        observations = apply_raw_view_mask(observations, mask)
                        with torch.no_grad():
                            raw_output = policy.net(
                                mode="waypoint",
                                waypoint_predictor=predictor,
                                observations=observations,
                                in_train=False)
                            output = filter_waypoint_outputs(raw_output, mask)
                        candidates = []
                        for angle, distance in zip(
                                output["cand_angles"][0],
                                output["cand_distances"][0]):
                            candidates.append({
                                "angle": float(angle),
                                "distance": float(distance),
                                "endpoint": candidate_position(
                                    state["position"], state["heading"],
                                    angle, distance),
                            })
                        inference_cache[cache_key] = candidates
                    candidates = inference_cache[cache_key]
                    own = targets_by_episode[episode_id][event_id]
                    own_scores = []
                    for index, candidate in enumerate(candidates):
                        distance, progress = segment_distance(
                            candidate["endpoint"], own["start"], own["end"])
                        own_scores.append((distance, -progress, index,
                                           candidate))
                    own_scores.sort(key=lambda value: (value[0], value[1],
                                                       value[2]))
                    if not own_scores:
                        reasons.append("NO_CAUSAL_AUTOMATIC_CANDIDATE")
                        prefix_records.append({
                            "prefix_index": int(prefix),
                            "candidate_count": 0,
                            "status": "NO_CAUSAL_AUTOMATIC_CANDIDATE"})
                        continue
                    best = own_scores[0]
                    candidate_margin = (own_scores[1][0] - best[0]
                                        if len(own_scores) > 1 else math.inf)
                    target_scores = []
                    for other_id, target in targets_by_episode[
                            episode_id].items():
                        distance, progress = segment_distance(
                            best[3]["endpoint"], target["start"],
                            target["end"])
                        if distance <= TARGET_TUBE_M and \
                                progress > PROGRESS_MIN:
                            target_scores.append((distance, other_id,
                                                  target["branch_id"],
                                                  progress))
                    target_scores.sort(key=lambda value: (value[0], value[1]))
                    own_first = bool(target_scores) and target_scores[0][1] \
                        == event_id
                    target_margin = (target_scores[1][0] -
                                     target_scores[0][0]
                                     if len(target_scores) > 1 else math.inf)
                    accepted = (best[0] <= TARGET_TUBE_M and
                                -best[1] > PROGRESS_MIN and
                                candidate_margin >= SEPARATION_MARGIN_M and
                                own_first and
                                target_margin >= SEPARATION_MARGIN_M)
                    if not accepted:
                        if best[0] > TARGET_TUBE_M:
                            reason = "AUTOMATIC_CANDIDATE_OUTSIDE_TARGET"
                        elif -best[1] <= PROGRESS_MIN:
                            reason = "AUTOMATIC_CANDIDATE_NO_PROGRESS"
                        elif candidate_margin < SEPARATION_MARGIN_M:
                            reason = "AUTOMATIC_CANDIDATE_AMBIGUITY"
                        else:
                            reason = "AUTOMATIC_CROSS_TARGET_AMBIGUITY"
                        reasons.append(reason)
                    else:
                        reason = "TRACKED"
                    prefix_records.append({
                        "prefix_index": int(prefix),
                        "candidate_count": len(candidates),
                        "status": reason,
                        "selected_candidate_endpoint_q": q3(
                            best[3]["endpoint"]),
                        "own_target_distance_m": round(best[0], 6),
                        "own_target_progress": round(-best[1], 6),
                        "candidate_separation_margin_m":
                            round(candidate_margin, 6)
                            if math.isfinite(candidate_margin) else None,
                        "cross_target_margin_m": round(target_margin, 6)
                            if math.isfinite(target_margin) else None,
                    })
                reasons = sorted(set(reasons))
                tracked = len(prefix_records) == K and not reasons
                results.append({
                    "provisional_event_id": event_id,
                    "episode_id": episode_id, "scene_id": scene,
                    "semantic_branch_id": event["semantic_branch_id"],
                    "status": "TRACKED_K3" if tracked else "NOT_TRACKED",
                    "prefix_records": prefix_records,
                    "reasons": reasons,
                })
        finally:
            sim.close()
    results.sort(key=lambda value: value["provisional_event_id"])
    output = {
        "worker": "automatic_semantic_candidate_worker",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "physical_gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu": 0,
        "scenes": shard_scenes,
        "events": results,
        "counts": {"scenes": len(shard_scenes),
                   "events": len(results),
                   "tracked_k3": sum(event["status"] == "TRACKED_K3"
                                     for event in results),
                   "unique_inference_prefixes": len(inference_cache)},
        "model_contract": {
            "policy_class": type(policy).__name__,
            "policy_net_class": type(policy.net).__name__,
            "task_type": config.MODEL.task_type,
            "final_checkpoint_strict": True,
            "waypoint_checkpoint_strict": True,
            "sensor_count": 2,
            "sensor_hfov_deg": 63,
            "causal_acquired_slots": [0],
        },
        "network_attempts": network["attempts"],
    }
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({"shard": args.shard_index,
                      "counts": output["counts"],
                      "network_attempts": output["network_attempts"]}))


if __name__ == "__main__":
    main()
