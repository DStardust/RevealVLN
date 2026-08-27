#!/usr/bin/env python3
"""Execute target and counterfactual CR5 branches with Habitat-Sim controls."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/mnt/daiyang/vla")
HABSIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / "third_party/habitat-sim"))
if str(HABSIM) not in sys.path:
    sys.path.insert(0, str(HABSIM))

import habitat_sim  # noqa: E402
from habitat_sim.agent import ActionSpec, ActuationSpec  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_preflight/multiview_branch"
GEOMETRY = BASE / "CR5_DIRECTED_GEOMETRY_PREFLIGHT.json"
OUT = BASE / "CR5_CONTROLLER_EXECUTION_PREFLIGHT.json"
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
EXPECTED_GEOMETRY_SHA256 = (
    "92a461a5cebfe84c53bce211bd3c78bec59f8aaa0d2be73b46e814b5bcb374f0"
)
EXPECTED_CANDIDATE_COUNT = 10
OUTPUT_MANIFEST = "MF2-CR5 discrete controller execution preflight"
OUTPUT_REVISION = "cr5-controller-execution-preflight/1"
OUTPUT_SCOPE = "10 geometry-pass RxR-train preflight candidates"
USE_ALL_BRANCHES = False

GPU_DEVICE = int(os.environ.get("CR5_CONTROLLER_GPU", "1"))
MOVE_M = 0.25
TURN_DEG = 30.0
GOAL_RADIUS_M = 0.20
FINAL_DISTANCE_MAX_M = 0.25
MAX_ACTIONS = 100
REPLAYS = 2


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def stable_sha(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def qpoint(value, places: int = 6):
    return [round(float(item), places) for item in value]


def qfloat(value, places: int = 6):
    return round(float(value), places)


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)
                                - np.asarray(b, dtype=float)))


def make_sim(scene: str):
    scene_glb = MP3D / scene / (scene + ".glb")
    navmesh = MP3D / scene / (scene + ".navmesh")
    for path in (scene_glb, navmesh):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("unsafe scene asset: " + str(path))
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_glb)
    sim_cfg.gpu_device_id = GPU_DEVICE
    sim_cfg.allow_sliding = False
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.action_space = {
        "move_forward": ActionSpec(
            "move_forward", ActuationSpec(amount=MOVE_M)),
        "turn_left": ActionSpec(
            "turn_left", ActuationSpec(amount=TURN_DEG)),
        "turn_right": ActionSpec(
            "turn_right", ActuationSpec(amount=TURN_DEG)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(
        sim_cfg, [agent_cfg]))
    # Simulator v0.1.7 may recompute a navmesh from the configured agent body
    # during construction.  Reload the independently acquired official MP3D
    # navmesh after construction so planning and step filtering use the exact
    # geometry gate asset.
    if not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError("official navmesh reload failed: " + scene)
    return sim


def agent_state(position, heading: float):
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype="float32")
    state.rotation = Rotation.from_rotvec(
        [0.0, float(heading), 0.0]).as_quat()
    return state


def rollout(sim, start, heading: float, goal):
    start = np.asarray(start, dtype="float32")
    goal = np.asarray(goal, dtype="float32")
    initial = agent_state(start, heading)
    sim.get_agent(0).set_state(initial, True)
    follower = sim.make_greedy_follower(
        agent_id=0, goal_radius=GOAL_RADIUS_M,
        fix_thrashing=True, thrashing_threshold=16)
    try:
        actions = follower.find_path(goal)
    except Exception as error:  # fail closed; exact class retained
        return {
            "status": "PLANNER_ERROR",
            "error_type": type(error).__name__,
            "actions": [],
            "positions": [qpoint(start)],
            "collision_count": None,
            "final_distance_m": None,
        }
    if not actions or actions[-1] is not None:
        return {
            "status": "PLANNER_NO_STOP",
            "actions": [value for value in actions if value is not None],
            "positions": [qpoint(start)],
            "collision_count": None,
            "final_distance_m": None,
        }
    executable_actions = actions[:-1]
    if len(executable_actions) > MAX_ACTIONS:
        return {
            "status": "ACTION_LIMIT_EXCEEDED",
            "actions": executable_actions,
            "positions": [qpoint(start)],
            "collision_count": None,
            "final_distance_m": None,
        }

    # find_path is a planning call; restore the exact initial state before the
    # actual discrete action execution.
    sim.get_agent(0).set_state(initial, True)
    positions = [qpoint(start)]
    collision_count = 0
    for action in executable_actions:
        observation = sim.step(action)
        collision_count += int(bool(observation.get("collided", False)))
        state = sim.get_agent(0).get_state()
        if not np.isfinite(state.position).all():
            return {
                "status": "NONFINITE_STATE",
                "actions": executable_actions,
                "positions": positions,
                "collision_count": collision_count,
                "final_distance_m": None,
            }
        positions.append(qpoint(state.position))
    final = np.asarray(sim.get_agent(0).get_state().position, dtype=float)
    final_distance = distance(final, goal)
    status = "ROLLOUT_PASS" if (
        final_distance <= FINAL_DISTANCE_MAX_M
        and collision_count == 0
    ) else "ROLLOUT_FAIL"
    return {
        "status": status,
        "actions": executable_actions,
        "action_count": len(executable_actions),
        "action_sequence_sha256": stable_sha(executable_actions),
        "positions": positions,
        "position_trace_sha256": stable_sha(positions),
        "collision_count": collision_count,
        "final_position_q": qpoint(final),
        "goal_q": qpoint(goal),
        "final_distance_m": qfloat(final_distance),
    }


def main() -> int:
    if sha256_file(GEOMETRY) != EXPECTED_GEOMETRY_SHA256:
        raise SystemExit("geometry preflight SHA drift")
    geometry = json.loads(GEOMETRY.read_text())
    candidates = [row for row in geometry["events"]
                  if row["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"]
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise SystemExit("expected exactly %d geometry-pass candidates" %
                         EXPECTED_CANDIDATE_COUNT)

    by_scene = {}
    for row in candidates:
        by_scene.setdefault(row["scene_id"], []).append(row)
    results = []
    for scene in sorted(by_scene):
        sim = make_sim(scene)
        try:
            for event in sorted(by_scene[scene], key=lambda value:
                                value["event_id"]):
                branch_results = {}
                start = event["trace"]["Q"]
                heading = event["trace"]["agent_heading_rad"]
                if USE_ALL_BRANCHES:
                    branch_rows = [event["target"]] + event["alternatives"]
                    goals = [(
                        row["branch_id"],
                        row["T_star_at_1_75m"] if index == 0 else
                        row["T_i_at_1_75m"],
                    ) for index, row in enumerate(branch_rows)]
                else:
                    goals = [
                        ("target", event["target"]["T_star_at_1_75m"]),
                        ("alternative", event["alternative"][
                            "T_i_at_1_75m"]),
                    ]
                for role, goal in goals:
                    replays = [rollout(sim, start, heading, goal)
                               for _ in range(REPLAYS)]
                    deterministic = all(
                        replay.get("actions") == replays[0].get("actions")
                        and replay.get("positions") ==
                        replays[0].get("positions")
                        and replay.get("status") == replays[0].get("status")
                        for replay in replays[1:]
                    )
                    branch_results[role] = {
                        "branch_id": role if USE_ALL_BRANCHES else (
                            event["target"]["branch_id"]
                            if role == "target" else
                            event["alternative"]["branch_id"]),
                        "replays": replays,
                        "deterministic_exact": deterministic,
                        "pass": deterministic and all(
                            replay["status"] == "ROLLOUT_PASS"
                            for replay in replays),
                    }
                passed = all(value["pass"]
                             for value in branch_results.values())
                result = {
                    "event_id": event["event_id"],
                    "episode_id": event["episode_id"],
                    "scene_id": scene,
                    "status": "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
                        if passed else "CONTROLLER_REJECT",
                    "geometry_verified": True,
                    "controller_verified": passed,
                    "causal_prefix_verified": False,
                    "human_label": None,
                    "training_label": False,
                }
                if USE_ALL_BRANCHES:
                    result.update({
                        "target_branch_id": event["target"]["branch_id"],
                        "candidate_branch_ids": event[
                            "candidate_branch_ids"],
                        "branches": [branch_results[branch_id]
                                     for branch_id in event[
                                         "candidate_branch_ids"]],
                        "all_candidate_branches_executed": True,
                    })
                else:
                    result.update({
                        "target": branch_results["target"],
                        "alternative": branch_results["alternative"],
                    })
                results.append(result)
        finally:
            sim.close()

    counts = Counter(row["status"] for row in results)
    output = {
        "manifest": OUTPUT_MANIFEST,
        "revision": OUTPUT_REVISION,
        "status": "COMPLETE_CAUSAL_AND_HUMAN_GATES_REQUIRED",
        "scope": OUTPUT_SCOPE,
        "source": {
            "path": str(GEOMETRY.relative_to(ROOT)),
            "sha256": EXPECTED_GEOMETRY_SHA256,
        },
        "controller": {
            "habitat_sim_version": "0.1.7",
            "gpu_device_id": GPU_DEVICE,
            "move_forward_m": MOVE_M,
            "turn_deg": TURN_DEG,
            "goal_radius_m": GOAL_RADIUS_M,
            "final_distance_max_m": FINAL_DISTANCE_MAX_M,
            "allow_sliding": False,
            "replays_per_branch": REPLAYS,
            "official_navmesh_reloaded_after_sim_construction": True,
        },
        "candidate_count": len(results),
        "branch_rollout_count": (
            sum(len(row["branches"]) for row in results) * REPLAYS
            if USE_ALL_BRANCHES else len(results) * 2 * REPLAYS
        ),
        "status_counts": dict(sorted(counts.items())),
        "events": sorted(results, key=lambda value: value["event_id"]),
        "network_calls_made": 0,
        "images_or_observation_tensors_written": 0,
        "causal_prefix_verified_count": 0,
        "human_verified_count": 0,
        "training_authorized": False,
    }
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "candidate_count": output["candidate_count"],
        "branch_rollout_count": output["branch_rollout_count"],
        "status_counts": output["status_counts"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
