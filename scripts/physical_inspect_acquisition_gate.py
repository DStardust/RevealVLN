#!/usr/bin/env python3
"""Real Habitat-Sim TURN/observe/MOVE gate for the MF2-CR1 view buffer."""

import hashlib
import json
import math
import os
import sys

import numpy as np


ROOT = "/mnt/daiyang/vla"
HABSIM = os.path.join(ROOT, "third_party", "habitat-sim")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HABSIM not in sys.path:
    sys.path.insert(0, HABSIM)

from revealnav_cr1.causal_frontend import CausalPoseViewBuffer  # noqa: E402


SCENE = os.path.join(ROOT, "third_party", "ETP-R1", "data",
                     "scene_datasets", "mp3d", "17DRP5sb8fy",
                     "17DRP5sb8fy.glb")
NAVMESH = SCENE[:-4] + ".navmesh"
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHYSICAL_INSPECT_ACQUISITION_GATE.json")
SEED = 20260824
TURN_DEG = 30.0
MOVE_M = 0.25


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def heading(rotation):
    from habitat_sim.utils.common import quat_rotate_vector

    forward = quat_rotate_vector(rotation, np.asarray([0.0, 0.0, -1.0]))
    return math.atan2(-float(forward[0]), -float(forward[2])) % (
        2 * math.pi)


def delta_deg(a, b):
    return math.degrees(abs((b - a + math.pi) % (2 * math.pi) - math.pi))


def pose_token(state):
    return tuple(round(float(x), 6) for x in state.position)


def main():
    import habitat_sim
    from habitat_sim.agent import ActionSpec, ActuationSpec

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE
    sim_cfg.gpu_device_id = 0
    sim_cfg.random_seed = SEED
    sim_cfg.allow_sliding = False

    rgb = habitat_sim.SensorSpec()
    rgb.uuid = "front_rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb.resolution = [224, 224]
    rgb.position = [0.0, 0.88, 0.0]
    rgb.hfov = 63.0
    depth = habitat_sim.SensorSpec()
    depth.uuid = "front_depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth.resolution = [256, 256]
    depth.position = [0.0, 0.88, 0.0]
    depth.hfov = 63.0
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = [rgb, depth]
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward",
                                   ActuationSpec(amount=MOVE_M)),
        "turn_left": ActionSpec("turn_left",
                                ActuationSpec(amount=TURN_DEG)),
        "turn_right": ActionSpec("turn_right",
                                 ActuationSpec(amount=TURN_DEG)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg,
                                                           [agent_cfg]))
    try:
        if not sim.pathfinder.load_nav_mesh(NAVMESH):
            raise RuntimeError("navmesh load failed")
        sim.pathfinder.seed(SEED)
        # Fixed selection rule independent of observations: first seeded
        # navigable point with >=0.75m clearance in world -Z.
        start = None
        for index in range(500):
            candidate = sim.pathfinder.get_random_navigable_point()
            desired = candidate + np.asarray([0.0, 0.0, -1.0])
            endpoint = sim.pathfinder.try_step_no_sliding(candidate, desired)
            if float(np.linalg.norm(endpoint - candidate)) >= 0.75:
                start = candidate
                sample_index = index
                break
        if start is None:
            raise RuntimeError("no deterministic clearance point")
        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        agent = sim.initialize_agent(0, state)

        buffer = CausalPoseViewBuffer()
        buffer.reset_pose(pose_token(agent.get_state()), heading_slot=0)
        obs0 = sim.get_sensor_observations()
        pose0 = agent.get_state()
        h0 = heading(pose0.rotation)
        frame0 = {"rgb_sha256": array_sha(obs0["front_rgb"]),
                  "depth_sha256": array_sha(obs0["front_depth"])}

        obs1 = sim.step("turn_left")
        buffer.turn(1)
        pose1 = agent.get_state()
        h1 = heading(pose1.rotation)
        frame1 = {"rgb_sha256": array_sha(obs1["front_rgb"]),
                  "depth_sha256": array_sha(obs1["front_depth"])}
        turn_position_delta = float(np.linalg.norm(
            np.asarray(pose1.position) - np.asarray(pose0.position)))
        turn_heading_delta = delta_deg(h0, h1)
        acquired_after_turn = buffer.relative_mask().nonzero().flatten()
        acquired_after_turn = [int(x) for x in acquired_after_turn.tolist()]

        obs2 = sim.step("move_forward")
        pose2 = agent.get_state()
        move_delta = float(np.linalg.norm(
            np.asarray(pose2.position) - np.asarray(pose1.position)))
        buffer.move(pose_token(pose2))
        frame2 = {"rgb_sha256": array_sha(obs2["front_rgb"]),
                  "depth_sha256": array_sha(obs2["front_depth"])}
        acquired_after_move = buffer.relative_mask().nonzero().flatten()
        acquired_after_move = [int(x) for x in
                               acquired_after_move.tolist()]

        checks = {
            "only_front_rgb_depth_sensors_exist":
                sorted(obs0) == ["front_depth", "front_rgb"],
            "front_frames_finite_and_typed":
                obs0["front_rgb"].dtype == np.uint8 and
                bool(np.isfinite(obs0["front_depth"]).all()),
            "physical_turn_keeps_position": turn_position_delta <= 1e-6,
            "physical_turn_is_30_degrees":
                abs(turn_heading_delta - TURN_DEG) <= 1e-4,
            "new_front_observation_acquired_after_turn": frame0 != frame1,
            "two_world_headings_cached_after_counted_turn":
                acquired_after_turn == [0, 11] and
                buffer.lowlevel_turn_count == 1,
            "physical_move_executed": move_delta >= 0.20,
            "move_resets_pose_local_cache": acquired_after_move == [0],
            "shared_lowlevel_budget_is_two": buffer.counted_actions == 2 and
                                             buffer.lowlevel_move_count == 1,
        }
        passed = all(checks.values())
        output = {
            "gate": "mf2_cr1_physical_inspect_acquisition",
            "revision": "physical-inspect-acquisition/1",
            "status": "PASS" if passed else "FAIL",
            "decision": "PHYSICAL_ACQUISITION_PASS" if passed else
                        "PHYSICAL_ACQUISITION_NO_GO",
            "scene_id": "17DRP5sb8fy",
            "split_scope": "MP3D geometry only; no episode payload parsed",
            "seed": SEED,
            "deterministic_start_sample_index": sample_index,
            "sensor_contract": {"count": 2, "hfov_deg": 63.0,
                                "headings_rendered_per_step": 1},
            "action_contract": {"turn_deg": TURN_DEG,
                                "move_m": MOVE_M},
            "checks": checks,
            "measurements": {
                "turn_position_delta_m": turn_position_delta,
                "turn_heading_delta_deg": turn_heading_delta,
                "move_position_delta_m": move_delta,
                "acquired_relative_slots_after_turn": acquired_after_turn,
                "acquired_relative_slots_after_move": acquired_after_move,
                "frames": [frame0, frame1, frame2],
            },
            "boundaries": {
                "image_or_depth_values_written": False,
                "checkpoint_loaded": False,
                "training_performed": False,
                "val_unseen_or_test_used": False,
                "network_used": False,
            },
        }
        with open(OUT, "w") as fh:
            json.dump(output, fh, indent=2)
            fh.write("\n")
        print(json.dumps({"status": output["status"],
                          "decision": output["decision"],
                          "checks": checks,
                          "measurements": output["measurements"],
                          "output": os.path.relpath(OUT, ROOT)}, indent=2))
        return 0 if passed else 1
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())

