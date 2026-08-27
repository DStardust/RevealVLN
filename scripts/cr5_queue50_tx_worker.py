#!/usr/bin/env python3
"""Build one CR5 resource-conditioned expiry witness in Habitat-Sim.

The worker evaluates the two MF2-CR1 declared options at every observed
prefix from the CR5 decision checkpoint through the end of the trajectory:
go directly to the target after strict reveal, or return through the saved
checkpoint and then enter the target branch.  It writes costs and hashes, not
images or observation tensors.  A separate orchestrator runs this worker in
two independent processes and compares the evidence digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
CAUSAL = BASE / "causal_gate"
GEOMETRY_PATH = (
    BASE / "multiview_primary/CR5_QUEUE50_DIRECTED_GEOMETRY.json"
)
CONTROLLER_PATH = (
    BASE / "multiview_primary/CR5_QUEUE50_CONTROLLER_EXECUTION.json"
)
ACCEPTANCE_PATH = CAUSAL / "CR5_QUEUE50_HUMAN50_ACCEPTANCE.json"
ANALYSIS_PATH = CAUSAL / "CR5_QUEUE50_CAUSAL_CANDIDATE_ANALYSIS.json"
LANGUAGE_PATH = CAUSAL / "CR5_QUEUE50_CAUSAL_PREFIX_LANGUAGE_GATE.json"
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
ETPR1 = ROOT / "third_party/ETP-R1"
HABLAB = ROOT / "third_party/habitat-lab"
HABSIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", str(ROOT / "third_party/habitat-sim")))
for dependency in (ETPR1, HABLAB, HABSIM):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import habitat_sim  # noqa: E402
from habitat_sim.agent import ActionSpec, ActuationSpec  # noqa: E402
from habitat_extensions.shortest_path_follower import (  # noqa: E402
    ShortestPathFollowerCompat,
)


EXPECTED_SHA256 = {
    ACCEPTANCE_PATH:
        "fa0e126be303d53767b367ab90673ec4914282c589583cfa6178ccf4f7e3e681",
    GEOMETRY_PATH:
        "46609126537ddb9c4936bc93d683dd06243b91203d6bf2f7c03f30cca7deb850",
    CONTROLLER_PATH:
        "567039afac8f53141b9f1d2114ee79a47611ca7e68b1eefc9d2ea40d72eff574",
    ANALYSIS_PATH:
        "6e85507666bd6a94746b9b9ecb4a8229fe3fb71184f8e43d49a0938044fdedae",
    LANGUAGE_PATH:
        "eaf494b5348b0c6cdbb01a0199fe794d00cf99fb7d3b8c2059248a5a64081e23",
}
FOLLOWER_PATH = ETPR1 / "habitat_extensions/shortest_path_follower.py"
EXPECTED_FOLLOWER_SHA256 = (
    "d5e5890ad35c1bc73525505da875df8fe314f8d727f48681345bdde16702b7fb"
)

MOVE_M = 0.25
TURN_DEG = 30.0
GOAL_RADIUS_M = 0.20
FINAL_DISTANCE_MAX_M = 0.25
MAX_ROUTE_ACTIONS = 1000
DENOMINATOR_FLOOR_ACTIONS = 5
NORMALIZED_BUDGETS = (1.5, 2.0, 3.0, 4.0)
CONTROLLERS = ("oracle_greedy", "frozen_shortest_path_compat")
EVIDENCE_REVISION = "cr5-queue50-resource-conditioned-tx-event/1"
RUN_REVISION = "cr5-queue50-tx-worker-run/1"
ACTION_NAMES = {
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def stable_sha(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def qpoint(value):
    return [round(float(component), 6) for component in value]


def qfloat(value):
    return round(float(value), 6)


def project_file(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise RuntimeError("path resolves outside project: " + str(path))
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("input is not a regular project file: " + str(path))
    return path


def atomic_json(path: Path, value) -> None:
    resolved_parent = path.parent.resolve()
    if ROOT.resolve() != resolved_parent and ROOT.resolve() not in resolved_parent.parents:
        raise RuntimeError("output resolves outside project")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(part, path)


def install_network_guard():
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append("blocked")
        raise RuntimeError("network disabled in T_X worker")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    return attempts


def geodesic(pathfinder, start, end) -> float:
    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(start, dtype="float32")
    query.requested_end = np.asarray(end, dtype="float32")
    if not pathfinder.find_path(query):
        return math.inf
    distance = float(query.geodesic_distance)
    return distance if math.isfinite(distance) else math.inf


class FrozenFollowerAdapter:
    class Config:
        FORWARD_STEP_SIZE = MOVE_M
        TURN_ANGLE = TURN_DEG

    def __init__(self, sim):
        self.sim = sim
        self.habitat_config = self.Config()

    def geodesic_distance(self, start, end):
        return geodesic(self.sim.pathfinder, start, end)

    def get_straight_shortest_path_points(self, start, end):
        query = habitat_sim.ShortestPath()
        query.requested_start = np.asarray(start, dtype="float32")
        query.requested_end = np.asarray(end, dtype="float32")
        if not self.sim.pathfinder.find_path(query):
            return []
        return query.points

    @property
    def up_vector(self):
        return np.asarray([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        return -np.asarray([0.0, 0.0, 1.0])

    def get_agent_state(self, agent_id=0):
        return self.sim.get_agent(agent_id).get_state()

    def set_agent_state(
            self, position, rotation, agent_id=0, reset_sensors=True):
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype="float32")
        state.rotation = rotation
        self.sim.get_agent(agent_id).set_state(state, reset_sensors)

    def step(self, action):
        return self.sim.get_agent(0).act(int(action))


def make_sim(scene: str, gpu: int):
    scene_glb = project_file(MP3D / scene / (scene + ".glb"))
    navmesh = project_file(MP3D / scene / (scene + ".navmesh"))
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = str(scene_glb)
    sim_config.gpu_device_id = int(gpu)
    sim_config.allow_sliding = False
    sim_config.enable_physics = False
    sim_config.create_renderer = True

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.height = 0.88
    agent_config.radius = 0.18
    sensor = habitat_sim.SensorSpec()
    sensor.uuid = "tx_renderer_only"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = [16, 16]
    agent_config.sensor_specifications = [sensor]
    agent_config.action_space = {
        1: ActionSpec("move_forward", ActuationSpec(amount=MOVE_M)),
        2: ActionSpec("turn_left", ActuationSpec(amount=TURN_DEG)),
        3: ActionSpec("turn_right", ActuationSpec(amount=TURN_DEG)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(
        sim_config, [agent_config]))
    if not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError("official navmesh reload failed: " + scene)
    return sim


def set_state(sim, position, heading: float) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(
        sim.pathfinder.snap_point(position), dtype="float32")
    state.rotation = Rotation.from_rotvec(
        [0.0, float(heading), 0.0]).as_quat()
    sim.get_agent(0).set_state(state, True)


def execute_action(sim, action: int, positions: list[list[float]]):
    collided = bool(sim.get_agent(0).act(int(action)))
    position = qpoint(sim.get_agent(0).get_state().position)
    positions.append(position)
    return collided


def route(sim, controller: str, start, heading: float, goals):
    """Execute all declared legs from one exact prefix state."""

    set_state(sim, start, heading)
    actions: list[int] = []
    positions = [qpoint(sim.get_agent(0).get_state().position)]
    leg_action_counts = []
    collision_count = 0
    controller_calls = 0
    status = "SUCCESS"
    error_type = None
    adapter = FrozenFollowerAdapter(sim)

    for raw_goal in goals:
        goal = np.asarray(sim.pathfinder.snap_point(raw_goal), dtype="float32")
        before = len(actions)
        if controller == "oracle_greedy":
            initial = sim.get_agent(0).get_state()
            follower = habitat_sim.nav.GreedyGeodesicFollower(
                sim.pathfinder, sim.get_agent(0), GOAL_RADIUS_M,
                stop_key=None, forward_key=1, left_key=2, right_key=3,
                fix_thrashing=True,
            )
            try:
                controller_calls += 1
                planned = follower.find_path(goal)
            except Exception as error:  # exact type retained; fail closed
                status = "CONTROLLER_ERROR"
                error_type = type(error).__name__
                break
            sim.get_agent(0).set_state(initial, True)
            planned = [int(action) for action in planned if action is not None]
            if len(actions) + len(planned) > MAX_ROUTE_ACTIONS:
                status = "ACTION_LIMIT_EXCEEDED"
                break
            for action in planned:
                actions.append(action)
                if execute_action(sim, action, positions):
                    collision_count += 1
                    status = "COLLISION"
                    break
            if status != "SUCCESS":
                break
        elif controller == "frozen_shortest_path_compat":
            follower = ShortestPathFollowerCompat(
                adapter, GOAL_RADIUS_M, return_one_hot=False)
            while len(actions) < MAX_ROUTE_ACTIONS:
                controller_calls += 1
                try:
                    action = follower.get_next_action(goal)
                except Exception as error:  # exact type retained; fail closed
                    status = "CONTROLLER_ERROR"
                    error_type = type(error).__name__
                    break
                if action is None:
                    break
                action = int(action)
                actions.append(action)
                if execute_action(sim, action, positions):
                    collision_count += 1
                    status = "COLLISION"
                    break
            else:
                status = "ACTION_LIMIT_EXCEEDED"
            if status != "SUCCESS":
                break
        else:
            raise ValueError("unknown controller: " + controller)

        remaining = geodesic(
            sim.pathfinder, sim.get_agent(0).get_state().position, goal)
        if not math.isfinite(remaining) or remaining > GOAL_RADIUS_M + 1e-3:
            status = "GOAL_NOT_REACHED"
            break
        leg_action_counts.append(len(actions) - before)

    final = qpoint(sim.get_agent(0).get_state().position)
    goal = qpoint(sim.pathfinder.snap_point(goals[-1]))
    final_distance = float(np.linalg.norm(
        np.asarray(final, dtype=float) - np.asarray(goal, dtype=float)))
    path_length = sum(float(np.linalg.norm(
        np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)))
        for previous, current in zip(positions, positions[1:]))
    success = (
        status == "SUCCESS" and collision_count == 0
        and final_distance <= FINAL_DISTANCE_MAX_M
    )
    if status == "SUCCESS" and not success:
        status = "FINAL_DISTANCE_FAIL"
    result = {
        "status": status,
        "success": success,
        "error_type": error_type,
        "action_count": len(actions),
        "actions": actions,
        "action_sequence_sha256": stable_sha(actions),
        "leg_action_counts": leg_action_counts,
        "controller_call_count": controller_calls,
        "collision_count": collision_count,
        "path_length_m": qfloat(path_length),
        "position_trace_sha256": stable_sha(positions),
        "start_position_q": positions[0],
        "final_position_q": final,
        "goal_q": goal,
        "final_distance_m": qfloat(final_distance),
    }
    result["replay_sha256"] = stable_sha(result)
    return result


def classify_frontier(feasible: list[bool], prefix_indices: list[int]):
    if not any(feasible):
        return "NEVER_FEASIBLE", None, 0
    last_offset = len(feasible) - 1 - list(reversed(feasible)).index(True)
    if last_offset == len(feasible) - 1:
        return "RIGHT_CENSORED", None, 0
    seen_safe = False
    seen_unsafe_after_safe = False
    reentry = False
    for value in feasible[:last_offset + 1]:
        if value:
            if seen_unsafe_after_safe:
                reentry = True
            seen_safe = True
        elif seen_safe:
            seen_unsafe_after_safe = True
    transitions = sum(
        current != previous
        for previous, current in zip(feasible, feasible[1:]))
    status = (
        "UNIQUE_LAST_SAFE_WITH_REENTRY" if reentry
        else "UNIQUE_LAST_SAFE_MONOTONE"
    )
    return status, prefix_indices[last_offset], transitions


def select_unique(mapping, key, value):
    rows = [row for row in mapping if str(row[key]) == str(value)]
    if len(rows) != 1:
        raise RuntimeError("expected one %s=%s row" % (key, value))
    return rows[0]


def make_event_evidence(event_id: str, gpu: int, attempts):
    sources = []
    for path, expected in EXPECTED_SHA256.items():
        project_file(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError("source SHA drift: " + str(path))
        sources.append({
            "path": str(path.relative_to(ROOT)), "sha256": observed,
        })
    if sha256_file(project_file(FOLLOWER_PATH)) != EXPECTED_FOLLOWER_SHA256:
        raise RuntimeError("frozen follower SHA drift")

    acceptance = load_json(ACCEPTANCE_PATH)
    if event_id not in acceptance["eligible_event_ids"]:
        raise RuntimeError("event not in the sealed strict T_R set")
    geometry = select_unique(
        load_json(GEOMETRY_PATH)["events"], "event_id", event_id)
    controller = select_unique(
        load_json(CONTROLLER_PATH)["events"], "event_id", event_id)
    analysis_payload = load_json(ANALYSIS_PATH)
    analysis = select_unique(
        analysis_payload["events"], "event_id", event_id)
    language = select_unique(
        load_json(LANGUAGE_PATH)["events"], "event_id", event_id)
    if (
        geometry["status"] != "GEOMETRY_PASS_CONTROLLER_REQUIRED"
        or controller["status"] != "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
        or analysis["status"] != "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
        or language["status"] != "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED"
    ):
        raise RuntimeError("upstream gate status mismatch")

    episode_id = str(geometry["episode_id"])
    shard_path = CAUSAL / "frontend_shards" / ("ep" + episode_id + ".json")
    shard_sources = {
        row["path"]: row["sha256"]
        for row in analysis_payload["sources"]["frontend_shards"]
    }
    relative_shard = str(shard_path.relative_to(ROOT))
    expected_shard = shard_sources.get(relative_shard)
    if expected_shard is None or sha256_file(project_file(shard_path)) != expected_shard:
        raise RuntimeError("frontend shard provenance mismatch")
    sources.append({"path": relative_shard, "sha256": expected_shard})
    sources.append({
        "path": str(FOLLOWER_PATH.relative_to(ROOT)),
        "sha256": EXPECTED_FOLLOWER_SHA256,
    })
    shard = load_json(shard_path)
    if (
        str(shard["episode_id"]) != episode_id
        or shard["scene_id"] != geometry["scene_id"]
        or shard["network_attempts"] != 0
        or shard["model_contract"]["sensor_hfov_deg"] != 63
        or shard["model_contract"]["causal_acquired_slots"] != [0]
    ):
        raise RuntimeError("frontend shard contract mismatch")

    checkpoint_prefix = int(geometry["trace"]["Q_prefix"])
    records = shard["prefix_records"]
    if checkpoint_prefix >= len(records):
        raise RuntimeError("checkpoint outside observed prefix horizon")
    checkpoint = geometry["trace"]["Q"]
    checkpoint_state = records[checkpoint_prefix]
    checkpoint_error = float(np.linalg.norm(
        np.asarray(checkpoint_state["position_q"], dtype=float)
        - np.asarray(checkpoint, dtype=float)))
    if checkpoint_error > 1e-4:
        raise RuntimeError("Q does not match causal trace prefix")
    reveal_start, reveal_end = map(int, language["reveal_interval"])
    if (
        language["confirmation_prefix"] != reveal_end
        or reveal_end - reveal_start + 1 != 3
        or not (0 <= reveal_start <= reveal_end < len(records))
    ):
        raise RuntimeError("strict K3 reveal interval mismatch")

    target = geometry["target"]["T_star_at_1_75m"]
    target_branch_id = geometry["target"]["branch_id"]
    scene = geometry["scene_id"]
    sim = make_sim(scene, gpu)
    try:
        prefix_indices = list(range(checkpoint_prefix, len(records)))
        controller_evidence = {}
        for controller_name in CONTROLLERS:
            normalization = route(
                sim, controller_name, checkpoint,
                float(checkpoint_state["heading_rad"]), [target])
            denominator = max(
                normalization["action_count"], DENOMINATOR_FLOOR_ACTIONS)
            rows = []
            parent_hash = None
            for prefix_index in prefix_indices:
                state = records[prefix_index]
                direct = None
                if prefix_index >= reveal_start:
                    direct = route(
                        sim, controller_name, state["position_q"],
                        float(state["heading_rad"]), [target])
                saved = route(
                    sim, controller_name, state["position_q"],
                    float(state["heading_rad"]), [checkpoint, target])
                options = []
                if direct is not None and direct["success"]:
                    options.append((direct["action_count"], 0, "direct", direct))
                if saved["success"]:
                    options.append((saved["action_count"], 1, "saved", saved))
                best = min(options) if options else None
                row = {
                    "prefix_index": prefix_index,
                    "source_prefix_sha256": stable_sha(state),
                    "parent_cost_prefix_sha256": parent_hash,
                    "strict_target_revealed": prefix_index >= reveal_start,
                    "direct": direct if direct is not None else {
                        "status": "NOT_STRICTLY_REVEALED",
                        "success": False,
                    },
                    "saved_via_checkpoint": saved,
                    "cstar_action_count": best[0] if best else None,
                    "cstar_source": best[2] if best else None,
                    "cstar_replay_sha256": best[3]["replay_sha256"]
                        if best else None,
                }
                row["cost_prefix_sha256"] = stable_sha(row)
                parent_hash = row["cost_prefix_sha256"]
                rows.append(row)

            frontiers = {}
            if normalization["success"]:
                for normalized_budget in NORMALIZED_BUDGETS:
                    absolute_budget = normalized_budget * denominator
                    feasible = [
                        row["cstar_action_count"] is not None
                        and row["cstar_action_count"] <= absolute_budget
                        for row in rows
                    ]
                    status, last_safe, transitions = classify_frontier(
                        feasible, prefix_indices)
                    frontier = {
                        "normalized_budget": normalized_budget,
                        "absolute_action_budget": absolute_budget,
                        "status": status,
                        "last_safe_prefix": last_safe,
                        "feasibility_transition_count": transitions,
                        "feasibility_sha256": stable_sha(feasible),
                        "safe_witness": None,
                        "post_expiry_no_safe_certificate": None,
                    }
                    if last_safe is not None:
                        safe_row = rows[last_safe - checkpoint_prefix]
                        chosen = (
                            safe_row["direct"]
                            if safe_row["cstar_source"] == "direct"
                            else safe_row["saved_via_checkpoint"]
                        )
                        witness = {
                            "witness_id": "%s:%s:b%s:t%d" % (
                                event_id, controller_name,
                                normalized_budget, last_safe),
                            "destination_kind": "target_branch"
                                if safe_row["cstar_source"] == "direct"
                                else "checkpoint",
                            "destination_id": target_branch_id
                                if safe_row["cstar_source"] == "direct"
                                else event_id + ":Q",
                            "terminal_target_branch_id": target_branch_id,
                            "prefix_index": last_safe,
                            "action_ids": [
                                ACTION_NAMES[action]
                                for action in chosen["actions"]
                            ],
                            "path_cost_actions": chosen["action_count"],
                            "path_length_m": chosen["path_length_m"],
                            "replay_sha256": chosen["replay_sha256"],
                        }
                        witness["witness_sha256"] = stable_sha(witness)
                        post_row = rows[last_safe + 1 - checkpoint_prefix]
                        certificate_payload = {
                            "certificate_id": "%s:%s:b%s:t%d" % (
                                event_id, controller_name,
                                normalized_budget, last_safe + 1),
                            "declared_option_set": [
                                "direct_after_strict_reveal",
                                "saved_checkpoint_then_target",
                            ],
                            "prefix_index": last_safe + 1,
                            "absolute_action_budget": absolute_budget,
                            "direct_status": post_row["direct"]["status"],
                            "direct_action_count": post_row["direct"].get(
                                "action_count"),
                            "saved_status": post_row[
                                "saved_via_checkpoint"]["status"],
                            "saved_action_count": post_row[
                                "saved_via_checkpoint"].get("action_count"),
                            "cstar_action_count": post_row[
                                "cstar_action_count"],
                            "feasible": False,
                        }
                        certificate_payload["search_sha256"] = stable_sha(
                            certificate_payload)
                        frontier["safe_witness"] = witness
                        frontier["post_expiry_no_safe_certificate"] = (
                            certificate_payload)
                    frontiers[str(normalized_budget)] = frontier
            else:
                for normalized_budget in NORMALIZED_BUDGETS:
                    frontiers[str(normalized_budget)] = {
                        "normalized_budget": normalized_budget,
                        "absolute_action_budget": None,
                        "status": "CONTROLLER_NORMALIZATION_FAIL",
                        "last_safe_prefix": None,
                        "feasibility_transition_count": None,
                        "feasibility_sha256": None,
                        "safe_witness": None,
                        "post_expiry_no_safe_certificate": None,
                    }

            unique_count = sum(
                frontier["status"].startswith("UNIQUE_LAST_SAFE")
                for frontier in frontiers.values())
            controller_evidence[controller_name] = {
                "checkpoint_to_target_normalization": normalization,
                "normalization_denominator_actions": denominator,
                "prefix_costs": rows,
                "frontiers": frontiers,
                "unique_last_safe_budget_count": unique_count,
                "complete_prefix_evidence": len(rows) == len(prefix_indices),
            }
    finally:
        sim.close()

    if attempts:
        raise RuntimeError("network attempt observed")
    frozen = controller_evidence["frozen_shortest_path_compat"]
    checkpoint_changes_feasible_set = False
    for budget in NORMALIZED_BUDGETS:
        absolute = budget * frozen["normalization_denominator_actions"]
        checkpoint_changes_feasible_set |= any(
            row["saved_via_checkpoint"]["success"]
            and row["saved_via_checkpoint"]["action_count"] <= absolute
            and (
                not row["direct"]["success"]
                or row["direct"].get("action_count", math.inf) > absolute
            )
            for row in frozen["prefix_costs"]
        )

    evidence = {
        "revision": EVIDENCE_REVISION,
        "definition": "T_X(B)=max{t: C*_t <= B and witnessed sequence is safe}",
        "event_id": event_id,
        "episode_id": episode_id,
        "scene_id": scene,
        "target_branch_id": target_branch_id,
        "checkpoint": {
            "prefix_index": checkpoint_prefix,
            "position_q": qpoint(checkpoint),
            "heading_rad": float(checkpoint_state["heading_rad"]),
            "causal_trace_position_error_m": qfloat(checkpoint_error),
        },
        "strict_reveal_interval": [reveal_start, reveal_end],
        "observed_prefix_horizon": [checkpoint_prefix, len(records) - 1],
        "source_manifest": sorted(sources, key=lambda row: row["path"]),
        "controller_contract": {
            "habitat_sim_version": "0.1.7",
            "move_forward_m": MOVE_M,
            "turn_deg": TURN_DEG,
            "goal_radius_m": GOAL_RADIUS_M,
            "final_distance_max_m": FINAL_DISTANCE_MAX_M,
            "allow_sliding": False,
            "normalized_budgets": list(NORMALIZED_BUDGETS),
            "denominator_floor_actions": DENOMINATOR_FLOOR_ACTIONS,
            "option_set": [
                "direct_to_target_after_strict_reveal",
                "return_to_checkpoint_then_target",
            ],
        },
        "controllers": controller_evidence,
        "nontrivial": {
            "strict_reveal_after_episode_start": reveal_start > 0,
            "tx_before_reveal_any_frozen_budget": any(
                frontier["last_safe_prefix"] is not None
                and frontier["last_safe_prefix"] < reveal_start
                for frontier in frozen["frontiers"].values()
            ),
            "checkpoint_changes_frozen_feasible_set":
                checkpoint_changes_feasible_set,
        },
        "network_attempts": 0,
        "images_or_observation_tensors_written": 0,
        "future_information_used_for_online_input": 0,
        "offline_future_used_only_for_last_passage_label": True,
    }
    return evidence


def self_test() -> int:
    cases = [
        ([True, True, False], [4, 5, 6],
         ("UNIQUE_LAST_SAFE_MONOTONE", 5, 1)),
        ([True, False, True, False], [1, 2, 3, 4],
         ("UNIQUE_LAST_SAFE_WITH_REENTRY", 3, 3)),
        ([False, False], [8, 9], ("NEVER_FEASIBLE", None, 0)),
        ([False, True], [8, 9], ("RIGHT_CENSORED", None, 0)),
    ]
    for feasible, indices, expected in cases:
        observed = classify_frontier(feasible, indices)
        if observed != expected:
            raise AssertionError((feasible, observed, expected))
    print("TX_WORKER_SELF_TEST_PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.event_id is None or args.gpu is None or args.output is None:
        raise SystemExit("--event-id, --gpu and --output are required")
    output = args.output.resolve()
    if ROOT.resolve() not in output.parents:
        raise SystemExit("output resolves outside project")
    attempts = install_network_guard()
    started = time.monotonic()
    evidence = make_event_evidence(args.event_id, args.gpu, attempts)
    result = {
        "revision": RUN_REVISION,
        "event_id": args.event_id,
        "evidence": evidence,
        "event_evidence_sha256": stable_sha(evidence),
        "runtime": {
            "physical_gpu": args.gpu,
            "wall_clock_s": qfloat(time.monotonic() - started),
            "pid": os.getpid(),
        },
    }
    atomic_json(output, result)
    print(json.dumps({
        "event_id": args.event_id,
        "evidence_sha256": result["event_evidence_sha256"],
        "frozen_unique_budget_count": evidence["controllers"]
            ["frozen_shortest_path_compat"]
            ["unique_last_safe_budget_count"],
        "output": str(output.relative_to(ROOT.resolve())),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
