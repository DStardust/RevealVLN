#!/usr/bin/env python3
"""MF2-CR1 cost-frontier witness for all Oracle low-level probe events.

For each provisional fixed route-turn event and each counted low-level prefix,
this worker runs two public deterministic controllers in Habitat-Sim:

  oracle: Habitat-Sim GreedyGeodesicFollower;
  frozen: ETP-R1's accepted ShortestPathFollowerCompat.

It records direct cost only when the fixed branch is causally exposed and
saved-option cost only after the trace has reached the fixed checkpoint.  All
costs are counted MOVE/TURN actions.  The derived C* frontier is evaluated at
the four MF2-CR1 normalized budgets without selecting a favorable budget.

This is geometric RxR-train engineering evidence.  It does not load ETP-R1
weights, render/write observations, validate semantics/language, or authorize
training.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import sys
from collections import Counter, defaultdict


ROOT = "/mnt/daiyang/vla"
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHASE0C_COST_FRONTIER_WITNESS.json")
PROBE = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                     "PHASE0C_ORACLE_LOWLEVEL_PROBE.json")
MAPPING = os.path.join(ROOT, "artifacts", "phase0",
                       "REVEAL_QUEUE_50_MAPPING.json")
RXR_TRAIN = os.path.join(
    ROOT, "third_party", "ETP-R1", "data", "datasets",
    "RxR_VLNCE_v0_enc_xlmr", "train", "train_guide.json.gz")
ETPR1 = os.path.join(ROOT, "third_party", "ETP-R1")
HABLAB = os.path.join(ROOT, "third_party", "habitat-lab")
HABSIM = os.path.join(ROOT, "third_party", "habitat-sim")
MP3D = os.path.join(ETPR1, "data", "scene_datasets", "mp3d")

EXPECTED_PROBE_SHA = \
    "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac"
FORWARD_M = 0.25
TURN_DEG = 30
AGENT_HEIGHT = 0.88
AGENT_RADIUS = 0.18
GOAL_RADIUS_M = 0.3
MAX_LEG_ACTIONS = 1000
CHECKPOINT_REACHED_M = 0.3
NORMALIZED_BUDGETS = (1.5, 2.0, 3.0, 4.0)
DENOMINATOR_FLOOR_ACTIONS = 5
GATE4_MIN_FRACTION = 0.60
GATE4_MIN_UNIQUE_BUDGETS = 2
GATE5_MIN_FRACTION = 0.25

for _path in (HABSIM, HABLAB, ETPR1,
              os.path.join(ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def euclid(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


def quat_from_heading(heading):
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec([0.0, float(heading), 0.0]).as_quat()


def geodesic(pathfinder, a, b):
    import habitat_sim
    import numpy as np

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype="float32")
    path.requested_end = np.asarray(b, dtype="float32")
    return float(path.geodesic_distance) if pathfinder.find_path(path) \
        else math.inf


class Adapter:
    """Minimal interface required by frozen ShortestPathFollowerCompat."""

    class Config:
        FORWARD_STEP_SIZE = FORWARD_M
        TURN_ANGLE = TURN_DEG

    def __init__(self, sim):
        self.sim = sim
        self.habitat_config = self.Config()

    def geodesic_distance(self, a, b):
        return geodesic(self.sim.pathfinder, a, b)

    def get_straight_shortest_path_points(self, a, b):
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = a
        path.requested_end = b
        self.sim.pathfinder.find_path(path)
        return path.points

    @property
    def up_vector(self):
        import numpy as np
        return np.asarray([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        import numpy as np
        return -np.asarray([0.0, 0.0, 1.0])

    def get_agent_state(self, agent_id=0):
        return self.sim.get_agent(agent_id).get_state()

    def set_agent_state(self, position, rotation, agent_id=0,
                        reset_sensors=True):
        import habitat_sim
        import numpy as np

        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype="float32")
        state.rotation = rotation
        self.sim.get_agent(agent_id).set_state(state, reset_sensors)

    def step(self, action):
        return self.sim.get_agent(0).act(action)


def build_sim(scene, gpu_index):
    import habitat_sim
    from habitat_sim.agent import ActionSpec, ActuationSpec

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.join(MP3D, scene, scene + ".glb")
    sim_cfg.allow_sliding = False
    # Renderer initialization is required by Habitat-Sim 0.1.7 when it
    # recomputes the navmesh for the accepted 0.88m/0.18m agent dimensions.
    # No sensor observations are requested or materialized.
    sim_cfg.create_renderer = True
    sim_cfg.gpu_device_id = int(gpu_index)
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = AGENT_HEIGHT
    agent_cfg.radius = AGENT_RADIUS
    renderer_spec = habitat_sim.SensorSpec()
    renderer_spec.uuid = "cost_witness_renderer_only"
    renderer_spec.sensor_type = habitat_sim.SensorType.COLOR
    renderer_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    renderer_spec.resolution = [16, 16]
    agent_cfg.sensor_specifications = [renderer_spec]
    agent_cfg.action_space = {
        1: ActionSpec("move_forward", ActuationSpec(amount=FORWARD_M)),
        2: ActionSpec("turn_left", ActuationSpec(amount=TURN_DEG)),
        3: ActionSpec("turn_right", ActuationSpec(amount=TURN_DEG)),
    }
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

    state = habitat_sim.AgentState()
    state.position = np.asarray(sim.pathfinder.snap_point(position),
                                dtype="float32")
    state.rotation = quat_from_heading(heading)
    sim.get_agent(0).set_state(state, True)


def execute_actions(sim, actions):
    collisions = 0
    executed = []
    for action in actions:
        if action is None:
            continue
        collided = sim.get_agent(0).act(int(action))
        collisions += int(bool(collided))
        executed.append(int(action))
    return executed, collisions


def controller_route(sim, controller, start_pos, start_heading, goals):
    """Execute one or two legs without resetting between legs."""
    import habitat_sim
    import numpy as np
    from habitat_extensions.shortest_path_follower import (
        ShortestPathFollowerCompat)

    set_state(sim, start_pos, start_heading)
    adapter = Adapter(sim)
    actions, collisions = [], 0
    status = "SUCCESS"
    leg_counts = []
    for goal in goals:
        goal = np.asarray(sim.pathfinder.snap_point(goal), dtype="float32")
        before = len(actions)
        if controller == "oracle_greedy":
            follower = habitat_sim.nav.GreedyGeodesicFollower(
                sim.pathfinder, sim.get_agent(0), GOAL_RADIUS_M,
                stop_key=None, forward_key=1, left_key=2, right_key=3,
                fix_thrashing=True)
            try:
                planned = follower.find_path(goal)
            except habitat_sim.errors.GreedyFollowerError:
                status = "CONTROLLER_ERROR"
                break
            planned = [x for x in planned if x is not None]
            if len(planned) > MAX_LEG_ACTIONS:
                status = "BUDGET_EXCEEDED"
                break
            done, coll = execute_actions(sim, planned)
            actions.extend(done)
            collisions += coll
        elif controller == "frozen_shortest_path_compat":
            follower = ShortestPathFollowerCompat(
                adapter, GOAL_RADIUS_M, return_one_hot=False)
            for _ in range(MAX_LEG_ACTIONS):
                action = follower.get_next_action(goal)
                if action is None:
                    break
                done, coll = execute_actions(sim, [int(action)])
                actions.extend(done)
                collisions += coll
            else:
                status = "BUDGET_EXCEEDED"
                break
        else:
            raise ValueError(controller)
        remaining = geodesic(sim.pathfinder,
                             sim.get_agent(0).get_state().position, goal)
        if not math.isfinite(remaining) or remaining > GOAL_RADIUS_M + 1e-3:
            status = "GOAL_NOT_REACHED"
            break
        leg_counts.append(len(actions) - before)
    final = sim.get_agent(0).get_state().position
    return {
        "status": status,
        "success": status == "SUCCESS",
        "action_count": len(actions),
        "leg_action_counts": leg_counts,
        "collision_count": collisions,
        "action_sha256": hashlib.sha256(
            canonical(actions).encode()).hexdigest(),
        "final_position_q": [round(float(x), 6) for x in final],
    }


def classify_frontier(feasible):
    if not any(feasible):
        return "NEVER_FEASIBLE", None
    first = feasible.index(True)
    last = len(feasible) - 1 - list(reversed(feasible)).index(True)
    if last == len(feasible) - 1:
        return "RIGHT_CENSORED", None
    if any(not x for x in feasible[first:last + 1]) or any(
            feasible[last + 1:]):
        return "NON_MONOTONE", None
    return "UNIQUE_OBSERVED", last


def scene_worker(payload):
    """One process owns one Simulator at a time; returns all scene events."""
    from phase0c_oracle_lowlevel_probe import build_lowlevel_trace

    scene = payload["scene"]
    sim = build_sim(scene, payload["gpu_index"])
    results = []
    try:
        trace_cache = {}
        for event in payload["events"]:
            eid = event["episode_id"]
            episode = payload["episodes"][eid]
            if eid not in trace_cache:
                trace_cache[eid] = build_lowlevel_trace(sim.pathfinder,
                                                        episode)
            trace = trace_cache[eid]
            j = int(event["reference_turn_index"])
            ref = episode["reference_path"]
            cp = [float(x) for x in sim.pathfinder.snap_point(ref[j])]
            target = [float(x) for x in sim.pathfinder.snap_point(ref[j + 1])]
            cp_dists = [geodesic(sim.pathfinder, x["position"], cp)
                        for x in trace]
            checkpoint_prefix = min(
                range(len(cp_dists)), key=lambda k: (cp_dists[k], k))
            checkpoint_reached = cp_dists[checkpoint_prefix] <= \
                CHECKPOINT_REACHED_M + 1e-6
            exposed = [x["status"] == "EXPOSED"
                       for x in event["prefix_evidence"]]
            if len(exposed) != len(trace):
                results.append({"provisional_event_id":
                                event["provisional_event_id"],
                                "status": "ALIGNMENT_FAIL"})
                continue

            controllers = {}
            for controller in ("oracle_greedy",
                               "frozen_shortest_path_compat"):
                # Stable normalization leg: checkpoint oriented into target.
                target_heading = math.atan2(-(target[0] - cp[0]),
                                            -(target[2] - cp[2])) % \
                    (2 * math.pi)
                leg = controller_route(sim, controller, cp, target_heading,
                                       [target])
                prefix_costs = []
                for k, state in enumerate(trace):
                    direct = None
                    if exposed[k]:
                        direct = controller_route(
                            sim, controller, state["position"],
                            state["heading"], [target])
                    saved = None
                    if checkpoint_reached and k >= checkpoint_prefix:
                        saved = controller_route(
                            sim, controller, state["position"],
                            state["heading"], [cp, target])
                    candidates = []
                    if direct is not None and direct["success"]:
                        candidates.append((direct["action_count"], "direct"))
                    if saved is not None and saved["success"]:
                        candidates.append((saved["action_count"], "saved"))
                    if candidates:
                        cstar, source = min(candidates)
                    else:
                        cstar, source = None, None
                    prefix_costs.append({
                        "prefix_index": k, "action_count_elapsed":
                            state["action_count"],
                        "branch_exposed": exposed[k],
                        "checkpoint_available": checkpoint_reached and
                            k >= checkpoint_prefix,
                        "direct": direct, "saved": saved,
                        "cstar_action_count": cstar,
                        "cstar_source": source,
                    })
                denom = max(
                    leg["action_count"] if leg["success"] else 0,
                    DENOMINATOR_FLOOR_ACTIONS)
                frontiers = {}
                unique_count = 0
                for budget in NORMALIZED_BUDGETS:
                    absolute = budget * denom
                    feasibility = [
                        x["cstar_action_count"] is not None and
                        x["cstar_action_count"] <= absolute
                        for x in prefix_costs[checkpoint_prefix:]
                    ] if checkpoint_reached else []
                    status, last_local = classify_frontier(feasibility) \
                        if feasibility else ("CHECKPOINT_NOT_REACHED", None)
                    last = checkpoint_prefix + last_local \
                        if last_local is not None else None
                    unique_count += int(status == "UNIQUE_OBSERVED")
                    frontiers[str(budget)] = {
                        "normalized_budget": budget,
                        "absolute_action_budget": absolute,
                        "status": status,
                        "last_safe_prefix": last,
                    }
                controllers[controller] = {
                    "checkpoint_to_target_leg": leg,
                    "normalization_denominator_actions": denom,
                    "prefix_costs": prefix_costs,
                    "frontiers": frontiers,
                    "unique_observed_budget_count": unique_count,
                    "complete_evidence": leg is not None and
                        len(prefix_costs) == len(trace),
                }

            reveal = int(event["candidate_reveal_prefix"])
            frozen = controllers["frozen_shortest_path_compat"]
            checkpoint_changes_feasibility = any(
                row["checkpoint_available"]
                and row["saved"] is not None and row["saved"]["success"]
                and (row["direct"] is None or not row["direct"]["success"])
                for row in frozen["prefix_costs"])
            tx_before_reveal = any(
                x["last_safe_prefix"] is not None and
                x["last_safe_prefix"] < reveal
                for x in frozen["frontiers"].values())
            results.append({
                "provisional_event_id": event["provisional_event_id"],
                "episode_id": eid, "scene_id": scene,
                "reference_turn_index": j,
                "status": "COMPLETE",
                "prefix_count": len(trace),
                "candidate_reveal_prefix": reveal,
                "checkpoint_prefix": checkpoint_prefix,
                "checkpoint_min_geodesic_m": round(
                    cp_dists[checkpoint_prefix], 6),
                "checkpoint_reached": checkpoint_reached,
                "checkpoint_position": cp,
                "target_position": target,
                "controllers": controllers,
                "nontrivial": {
                    "reveal_after_start": reveal > 0,
                    "tx_before_reveal_any_budget": tx_before_reveal,
                    "checkpoint_changes_feasible_set":
                        checkpoint_changes_feasibility,
                },
            })
    finally:
        sim.close()
    return {"scene": scene, "gpu_index": payload["gpu_index"],
            "events": results}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if sha256_file(PROBE) != EXPECTED_PROBE_SHA:
        raise SystemExit("low-level probe SHA drift")
    follower_path = os.path.join(
        ETPR1, "habitat_extensions", "shortest_path_follower.py")
    follower_sha = sha256_file(follower_path)
    # The expected constant is advisory until first execution because this
    # accepted upstream file was previously evidenced by path, not pinned in
    # MF2-CR1. Fail only if the source is absent/symlinked; record exact SHA.
    if not os.path.isfile(follower_path) or os.path.islink(follower_path):
        raise SystemExit("frozen follower path invalid")
    probe = load_json(PROBE)
    if probe.get("decision") != "CONTINUE_TO_COST_WITNESS" or \
            len(probe.get("events", [])) != 104:
        raise SystemExit("unexpected low-level probe input")
    mapping = load_json(MAPPING)
    wanted = {str(x["episode_id"]) for x in mapping["items"]}
    with gzip.open(RXR_TRAIN, "rt") as fh:
        episodes = {str(x["episode_id"]): x
                    for x in json.load(fh)["episodes"]
                    if str(x["episode_id"]) in wanted}
    by_scene = defaultdict(list)
    for event in probe["events"]:
        by_scene[event["scene_id"]].append(event)
    gpu_ids = (1, 2, 3, 4, 5, 6, 7)
    payloads = [{"scene": scene, "events": events,
                 "gpu_index": gpu_ids[index % len(gpu_ids)],
                 "episodes": {e["episode_id"]: episodes[e["episode_id"]]
                              for e in events}}
                for index, (scene, events) in enumerate(
                    sorted(by_scene.items()))]

    scene_results = []
    os.environ.setdefault("GLOG_minloglevel", "2")
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(gpu_ids), mp_context=ctx) as pool:
        futures = {pool.submit(scene_worker, payload): payload["scene"]
                   for payload in payloads}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            scene_results.append(result)
            print("scene %s gpu=%s events=%d" %
                  (result["scene"], result["gpu_index"],
                   len(result["events"])), flush=True)
    events = [event for scene in scene_results for event in scene["events"]]
    events.sort(key=lambda x: x.get("provisional_event_id", ""))

    complete = [x for x in events if x.get("status") == "COMPLETE"]
    evidence_complete = len(complete) == 104 and all(
        all(c["complete_evidence"] for c in x["controllers"].values())
        for x in complete)
    gate3 = evidence_complete
    gate4_events = sum(
        x["controllers"]["frozen_shortest_path_compat"]
        ["unique_observed_budget_count"] >= GATE4_MIN_UNIQUE_BUDGETS
        for x in complete)
    gate4_fraction = gate4_events / len(complete) if complete else 0.0
    gate4 = gate4_fraction >= GATE4_MIN_FRACTION
    nontrivial_events = sum(
        any(x["nontrivial"].values()) for x in complete)
    nontrivial_fraction = nontrivial_events / len(complete) \
        if complete else 0.0
    gate5 = nontrivial_fraction >= GATE5_MIN_FRACTION

    controller_budget_counts = {}
    for controller in ("oracle_greedy", "frozen_shortest_path_compat"):
        controller_budget_counts[controller] = {}
        for budget in NORMALIZED_BUDGETS:
            controller_budget_counts[controller][str(budget)] = dict(Counter(
                x["controllers"][controller]["frontiers"][str(budget)]
                ["status"] for x in complete))

    overall = gate3 and gate4 and gate5
    output = {
        "gate": "mf2_cr1_phase0c_cost_frontier_witness",
        "revision": "cost-frontier-witness/1",
        "status": "PASS" if overall else "FAIL",
        "decision": "CONTINUE_PHASE0C" if overall else
                    "PHASE0C_COST_FRONTIER_NO_GO",
        "input": {"path": os.path.relpath(PROBE, ROOT),
                  "sha256": sha256_file(PROBE),
                  "provisional_events": 104},
        "controller_contract": {
            "oracle": "Habitat-Sim GreedyGeodesicFollower",
            "frozen": "ETP-R1 ShortestPathFollowerCompat",
            "frozen_source": os.path.relpath(follower_path, ROOT),
            "frozen_source_sha256": follower_sha,
            "forward_m": FORWARD_M, "turn_deg": TURN_DEG,
            "goal_radius_m": GOAL_RADIUS_M,
            "agent_height_m": AGENT_HEIGHT,
            "agent_radius_m": AGENT_RADIUS,
            "allow_sliding": False,
            "max_leg_actions": MAX_LEG_ACTIONS,
            "normalization_denominator_floor_actions":
                DENOMINATOR_FLOOR_ACTIONS,
            "normalized_budgets": list(NORMALIZED_BUDGETS),
        },
        "gates": {
            "gate3_complete_cost_evidence": gate3,
            "gate4_unique_tx_for_two_budgets_fraction": gate4_fraction,
            "gate4_required_fraction": GATE4_MIN_FRACTION,
            "gate4_pass": gate4,
            "gate5_nontrivial_fraction": nontrivial_fraction,
            "gate5_required_fraction": GATE5_MIN_FRACTION,
            "gate5_pass": gate5,
        },
        "counts": {
            "scene_groups": len(scene_results),
            "events": len(events),
            "complete_events": len(complete),
            "gate4_events": gate4_events,
            "nontrivial_events": nontrivial_events,
            "frontier_status_by_controller_budget":
                controller_budget_counts,
        },
        "events": events,
        "non_conclusions": {
            "semantic_branch_validity": False,
            "language_evidence_closure": False,
            "validated_reveal_event_count": 0,
            "automatic_frontend_gate_pass": False,
            "training_authorized": False,
            "human_review_authorized": False,
            "frozen_spec_modified": False,
            "val_unseen_or_test_used": False,
            "checkpoint_loaded": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "gates": output["gates"], "counts": output["counts"],
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
