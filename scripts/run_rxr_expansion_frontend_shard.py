#!/usr/bin/env python3
"""Run one resumable GPU shard of the frozen causal waypoint frontend."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cr5_causal_frontend_worker as worker  # noqa: E402
from automatic_semantic_candidate_worker import (  # noqa: E402
    build_models,
    build_sim,
    candidate_position,
    install_network_guard,
    make_observations,
    set_state,
)
from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402
from revealnav_cr1.causal_frontend import (  # noqa: E402
    apply_raw_view_mask,
    filter_waypoint_outputs,
)


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
GEOMETRY = BASE / "geometry/RXR_EXPANSION_DIRECTED_GEOMETRY.json"
CONTROLLER = BASE / "geometry/RXR_EXPANSION_CONTROLLER_EXECUTION.json"
INPUTS = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
OUT_DIR = BASE / "causal_frontend"
SHARD_DIR = OUT_DIR / "frontend_shards"
RUN_DIR = OUT_DIR / "runs"
RXR_TRAIN = worker.RXR_TRAIN
SEED = 20260825


def selected_events(shard_index: int, shard_count: int):
    geometry_document = json.loads(GEOMETRY.read_text())
    controller_document = json.loads(CONTROLLER.read_text())
    input_document = json.loads(INPUTS.read_text())
    geometry = {row["event_id"]: row for row in geometry_document["events"]}
    controller = {row["event_id"]: row for row in controller_document["events"]}
    inputs = {row["event_id"]: row for row in input_document["events"]}
    survivors = []
    for event_id, event in inputs.items():
        if (event_id in geometry and event_id in controller
                and geometry[event_id]["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
                and controller[event_id]["status"] == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
                and event["expansion_order"] % shard_count == shard_index):
            survivors.append(event)
    survivors.sort(key=lambda row: row["expansion_order"])
    sources = {
        "geometry": worker.sha256_file(GEOMETRY),
        "controller": worker.sha256_file(CONTROLLER),
        "inputs": worker.sha256_file(INPUTS),
    }
    return survivors, sources


def valid_existing(path: Path, episode_id: str, sources: dict) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = json.loads(path.read_text())
        return (value.get("episode_id") == episode_id
                and value.get("selection_sources") == sources
                and value.get("network_attempts") == 0
                and value.get("model_contract", {}).get("sensor_hfov_deg") == 63)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def process_episode(episode, event, sim, policy, predictor, config, device,
                    network, sources):
    import torch

    scene = worker.scene_name(episode)
    trace = build_lowlevel_trace(sim.pathfinder, episode)
    records = []
    for prefix_index, state in enumerate(trace):
        set_state(sim, state["position"], state["heading"])
        observations = make_observations(sim, device)
        acquired = torch.zeros((1, 12), dtype=torch.bool, device=device)
        acquired[:, 0] = True
        observations = apply_raw_view_mask(observations, acquired)
        with torch.no_grad():
            raw = policy.net(
                mode="waypoint", waypoint_predictor=predictor,
                observations=observations, in_train=False)
            filtered = filter_waypoint_outputs(raw, acquired)
        candidates = []
        for local_index, (angle, distance) in enumerate(zip(
                filtered["cand_angles"][0], filtered["cand_distances"][0])):
            angle = float(angle)
            distance = float(distance)
            endpoint = candidate_position(
                state["position"], state["heading"], angle, distance)
            signed = (angle + math.pi) % (2.0 * math.pi) - math.pi
            candidates.append({
                "candidate_local_id": f"C{local_index:02d}",
                "relative_angle_rad": round(angle, 8),
                "relative_angle_signed_deg": round(math.degrees(signed), 5),
                "distance_m": round(distance, 6),
                "endpoint_q": worker.qpoint(endpoint),
            })
        records.append({
            "prefix_index": prefix_index,
            "action": state["action"],
            "position_q": worker.qpoint(state["position"]),
            "heading_rad": round(float(state["heading"]), 8),
            "candidate_count": len(candidates),
            "candidates": candidates,
        })
        del observations, raw, filtered
    trace_contract = [{key: row[key] for key in (
        "prefix_index", "action", "position_q", "heading_rad")}
        for row in records]
    return {
        "revision": "rxr-expansion-causal-frozen-frontend/1",
        "event_id": event["event_id"],
        "expansion_order": event["expansion_order"],
        "episode_id": str(episode["episode_id"]),
        "scene_id": scene,
        "source_scope": "RxR-train only",
        "selection_sources": sources,
        "prefix_count": len(records),
        "trace_pose_action_sha256": worker.stable_sha(trace_contract),
        "prefix_records": records,
        "model_contract": {
            "policy_class": type(policy).__name__,
            "policy_net_class": type(policy.net).__name__,
            "task_type": config.MODEL.task_type,
            "final_checkpoint_strict": True,
            "waypoint_checkpoint_strict": True,
            "sensor_hfov_deg": 63,
            "causal_acquired_slots": [0],
            "hidden_panorama_slots_zeroed_before_encoder": True,
            "candidate_outputs_filtered_to_acquired_hfov": True,
        },
        "physical_gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu": 0,
        "seed": SEED,
        "network_attempts": network["attempts"],
        "observation_tensors_written": 0,
        "checkpoint_tensors_written": 0,
        "training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")
    events, sources = selected_events(args.shard_index, args.shard_count)
    wanted = {row["episode_id"] for row in events}
    with gzip.open(RXR_TRAIN, "rt", encoding="utf-8") as handle:
        episodes = {str(row["episode_id"]): row
                    for row in json.load(handle)["episodes"]
                    if str(row["episode_id"]) in wanted}
    if set(episodes) != wanted:
        raise SystemExit("RxR episode closure failure")

    pending = [event for event in events if not valid_existing(
        SHARD_DIR / ("ep" + event["episode_id"] + ".json"),
        event["episode_id"], sources)]
    network = install_network_guard()
    failures = []
    completed = 0
    if pending:
        import torch
        os.chdir(worker.ETPR1)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        policy, predictor, config = build_models()
        device = torch.device("cuda:0")
        by_scene = defaultdict(list)
        for event in pending:
            by_scene[worker.scene_name(episodes[event["episode_id"]])].append(event)
        for scene in sorted(by_scene):
            sim = build_sim(scene)
            try:
                for event in by_scene[scene]:
                    output = SHARD_DIR / ("ep" + event["episode_id"] + ".json")
                    try:
                        value = process_episode(
                            episodes[event["episode_id"]], event, sim,
                            policy, predictor, config, device, network, sources)
                        if network["attempts"] != 0:
                            raise RuntimeError("network attempt observed")
                        worker.atomic_json(output, value)
                        completed += 1
                    except Exception as error:
                        failures.append({
                            "event_id": event["event_id"],
                            "episode_id": event["episode_id"],
                            "error_type": type(error).__name__,
                            "error": str(error)[:2000],
                        })
                    print(f"[{completed + len(failures)}/{len(pending)}] {event['event_id']}", flush=True)
            finally:
                sim.close()
    result = {
        "status": "PASS" if not failures else "PASS_WITH_FAIL_CLOSED_FAILURES",
        "revision": "rxr-expansion-frontend-shard/1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_count": len(events),
        "reused_count": len(events) - len(pending),
        "completed_count": completed,
        "failure_count": len(failures),
        "failures": failures,
        "selection_sources": sources,
        "network_attempts": network["attempts"],
        "training_authorized": False,
    }
    output = RUN_DIR / f"shard_{args.shard_index:02d}.json"
    worker.atomic_json(output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
