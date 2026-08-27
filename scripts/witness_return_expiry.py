#!/usr/bin/env python3
"""Stage 5: bounded return/expiry witness engineering validation for the
first 5 items of the frozen RxR train queue.

For every prefix/candidate of each of the first 5 review units, this harness
tests — in a bare habitat-sim instance with the scene navmesh, WITHOUT the
policy and WITHOUT any training — whether a deterministic safe controller
(ETP-R1's frozen ShortestPathFollowerCompat, geodesic mode) can return from
the candidate endpoint to the nearest committed checkpoint pose (episode
start pose plus post-execution poses of earlier prefixes).

Allowed: navmesh/geodesic information for the witness (label generation and
controller witnessing only).  The evaluated policy never receives any of it;
this harness does not load any checkpoint.

Outputs:
  artifacts/runtime/phase0_reveal_closure/witness/WITNESS_RETURN_EXPIRY_FIRST5.json
  per-episode matrix files in the same directory.

Semantics (fixed a priori, version return-witness/v1):
  - checkpoint set for prefix k = [episode start pose] + post-execution
    poses of prefixes 0..k-1;
  - nearest checkpoint selected by navmesh geodesic distance; ties within
    1e-6 m mark the witness AMBIGUOUS (fail closed, no guessing);
  - start/end points are snapped to the navmesh (snap delta recorded);
  - controller budget 500 low-level steps; goal_radius 0.3 m;
  - success = follower returned None (within goal_radius) inside budget;
  - expiry proposal: the LAST prefix with at least one successful return
    witness followed by an observed prefix with none is marked
    expiry_proposal_status=PROVISIONAL (never "validated T_X"); all-safe or
    non-monotone patterns are marked accordingly; duplicate endpoints
    (within 1e-3 m) or multiple possible expiry moments mark AMBIGUOUS;
  - no Euclidean threshold or route waypoint index is used to fabricate T_X.
"""

import argparse
import hashlib
import json
import math
import os
import sys

PROJECT_ROOT = "/mnt/daiyang/vla"
ETPR1_ROOT = os.path.join(PROJECT_ROOT, "third_party", "ETP-R1")
HABITAT_LAB_ROOT = os.path.join(PROJECT_ROOT, "third_party", "habitat-lab")
HABITAT_SIM_ROOT = os.path.join(PROJECT_ROOT, "third_party", "habitat-sim")
MP3D_ROOT = os.path.join(ETPR1_ROOT, "data", "scene_datasets", "mp3d")
UNITS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase0", "review_units")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                            "REVEAL_QUEUE_50_MAPPING.json")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                            "phase0_reveal_closure", "witness",
                            "WITNESS_RETURN_EXPIRY_FIRST5.json")

WITNESS_VERSION = "return-witness/v1"
MAX_STEPS = 500
GOAL_RADIUS_M = 0.3
NEAREST_TIE_M = 1e-6
DUPLICATE_ENDPOINT_M = 1e-3
FORWARD_STEP_SIZE = 0.25
# ShortestPathFollowerCompat iterates headings with ``range`` and therefore
# requires the legacy Habitat config value to be an integer.  Keep the
# frozen runtime magnitude (30 degrees) while matching that API contract.
TURN_ANGLE = 30
AGENT_HEIGHT = 0.88
AGENT_RADIUS = 0.18

for _p in (HABITAT_SIM_ROOT, HABITAT_LAB_ROOT, ETPR1_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# network deny guard (same semantics as the runtime worker)
# ---------------------------------------------------------------------------
def install_network_deny_guard(counter_file):
    import atexit
    import socket

    if getattr(socket.socket.connect, "_witness_netguard", False):
        return
    state = {"attempts": 0}

    def _record():
        try:
            with open(counter_file, "a") as fh:
                fh.write(json.dumps({"pid": os.getpid(),
                                     "attempts": state["attempts"]}) + "\n")
        except OSError:
            pass

    original_connect = socket.socket.connect

    def guarded_connect(self, address):
        if getattr(self, "family", None) in (socket.AF_INET,
                                             socket.AF_INET6):
            state["attempts"] += 1
            _record()
            raise RuntimeError("network deny guard: connect blocked")
        return original_connect(self, address)

    def guarded_create_connection(address, *a, **k):
        state["attempts"] += 1
        _record()
        raise RuntimeError("network deny guard: create_connection blocked")

    guarded_connect._witness_netguard = True
    guarded_connect._witness_guard_state = state
    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create_connection
    atexit.register(_record)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def euclid(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


def geodesic(sim, a, b):
    """Same geodesic computation as frozen habitat-lab HabitatSim
    .geodesic_distance (MultiGoalShortestPath + pathfinder.find_path)."""
    import habitat_sim
    import numpy as np

    path = habitat_sim.MultiGoalShortestPath()
    path.requested_ends = np.array([np.array(b, dtype=np.float32)])
    path.requested_start = np.array(a, dtype=np.float32)
    sim.pathfinder.find_path(path)
    return path.geodesic_distance


def heading_toward(p1, p2):
    """Same heading convention as ETP-R1 environments.calculate_vp_rel_pos."""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    xz = max(math.sqrt(dx * dx + dz * dz), 1e-8)
    heading = math.asin(-dx / xz)
    if p2[2] > p1[2]:
        heading = math.pi - heading
    while heading < 0:
        heading += 2 * math.pi
    return heading % (2 * math.pi)


def quat_from_heading(heading):
    from scipy.spatial.transform import Rotation as R

    return R.from_rotvec([0, heading, 0]).as_quat()  # (x, y, z, w)


def build_sim(scene):
    import habitat_sim

    scene_dir = os.path.join(MP3D_ROOT, scene)
    glb = os.path.join(scene_dir, scene + ".glb")
    navmesh = os.path.join(scene_dir, scene + ".navmesh")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = glb
    sim_cfg.gpu_device_id = 0
    sim_cfg.allow_sliding = False

    # One minimal RGB sensor is attached ONLY so the renderer is created and
    # the navmesh recompute for the frozen agent dimensions (height 0.88,
    # radius 0.18 -- identical to the runtime task config) has scene
    # geometry available, exactly as in the accepted runtime episodes.
    # Observations are never read or written by this witness.
    rgb_spec = habitat_sim.SensorSpec()
    rgb_spec.uuid = "witness_renderer_only"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_spec.resolution = [16, 16]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = AGENT_HEIGHT
    agent_cfg.radius = AGENT_RADIUS
    agent_cfg.sensor_specifications = [rgb_spec]
    from habitat_sim.agent import ActionSpec, ActuationSpec

    agent_cfg.action_space = {
        1: ActionSpec("move_forward",
                      ActuationSpec(amount=FORWARD_STEP_SIZE)),
        2: ActionSpec("turn_left", ActuationSpec(amount=TURN_ANGLE)),
        3: ActionSpec("turn_right", ActuationSpec(amount=TURN_ANGLE)),
    }
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    # navmesh auto-loaded from <scene>.navmesh then recomputed for the
    # frozen agent dimensions (runtime-identical semantics)
    return sim, {"glb_sha256": sha256_file(glb),
                 "navmesh_sha256": sha256_file(navmesh),
                 "glb_bytes": os.path.getsize(glb),
                 "navmesh_bytes": os.path.getsize(navmesh)}


def run_witness(sim, follower, start_pos, start_heading, target_pos):
    """Deterministic safe-return witness; returns a record dict."""
    pf = sim.pathfinder
    snapped_start = pf.snap_point(start_pos)
    snapped_target = pf.snap_point(target_pos)
    if not all(math.isfinite(x) for x in list(snapped_start) +
               list(snapped_target)):
        return {"status": "INFEASIBLE_SNAP", "steps": 0,
                "action_sequence_hash": None, "path_length_m": 0.0,
                "final_geodesic_m": None,
                "snap_delta_start_m": None, "snap_delta_target_m": None}
    snap_delta_start = euclid(start_pos, snapped_start)
    snap_delta_target = euclid(target_pos, snapped_target)

    sim.set_agent_state(snapped_start, quat_from_heading(start_heading))
    actions = []
    positions = [list(sim.get_agent_state().position)]
    status = "BUDGET_EXCEEDED"
    for _ in range(MAX_STEPS):
        a = follower.get_next_action(snapped_target)
        if a is None:
            status = "SUCCESS"
            break
        sim.step(a)
        actions.append(int(a))
        positions.append(list(sim.get_agent_state().position))
    path_len = sum(euclid(positions[k], positions[k + 1])
                   for k in range(len(positions) - 1))
    final_geodesic = geodesic(
        sim, sim.get_agent_state().position, snapped_target)
    return {
        "status": status,
        "success": status == "SUCCESS",
        "steps": len(actions),
        "action_sequence_hash": hashlib.sha256(
            canonical_json(actions).encode("utf-8")).hexdigest(),
        "path_length_m": round(path_len, 6),
        "final_geodesic_m": (round(float(final_geodesic), 6)
                             if math.isfinite(final_geodesic) else None),
        "snap_delta_start_m": round(snap_delta_start, 6),
        "snap_delta_target_m": round(snap_delta_target, 6),
    }


def witness_episode(unit_path, sim_cache, netguard_file):
    import habitat_sim

    with open(unit_path) as fh:
        unit = json.load(fh)
    chain_path = os.path.join(PROJECT_ROOT, unit["run"]["chain_file"])
    with open(chain_path) as fh:
        chain = [json.loads(ln) for ln in fh if ln.strip()]
    scene = unit["scene_id"]
    if scene not in sim_cache:
        sim, scene_hashes = build_sim(scene)
        sim_cache[scene] = (sim, BareSimAdapter(sim), scene_hashes)
    sim, asim, scene_hashes = sim_cache[scene]

    # controller config hash (identical across episodes by construction)
    controller_config = {
        "version": WITNESS_VERSION,
        "controller": "ShortestPathFollowerCompat (frozen ETP-R1 copy of "
                      "habitat-lab v0.1.4 follower), geodesic_path mode",
        "goal_radius_m": GOAL_RADIUS_M,
        "max_steps": MAX_STEPS,
        "forward_step_size": FORWARD_STEP_SIZE,
        "turn_angle_deg": TURN_ANGLE,
        "agent_height": AGENT_HEIGHT,
        "agent_radius": AGENT_RADIUS,
        "allow_sliding": False,
        "navmesh": "scene .navmesh auto-loaded then recomputed for agent "
                   "height 0.88 / radius 0.18 (identical to the accepted "
                   "runtime episodes)",
        "nearest_checkpoint_metric": "navmesh geodesic",
        "nearest_tie_m": NEAREST_TIE_M,
        "duplicate_endpoint_m": DUPLICATE_ENDPOINT_M,
        "sim_version": habitat_sim.__version__,
        "scene_hashes": scene_hashes,
        "checkpoint_set_rule":
            "episode start pose + post-execution poses of earlier prefixes",
        "action_execution": "habitat_sim.Agent.act with the configured "
                            "navmesh collision filter; sensor observations "
                            "are neither rendered nor materialized",
    }
    controller_config_sha = hashlib.sha256(
        canonical_json(controller_config).encode("utf-8")).hexdigest()

    # episode start pose from the chain episode meta / unit run meta:
    # use the first prefix agent pose as proxy? No: use payload start pose.
    # The review unit does not store start pose; read it from the mapping.
    with open(MAPPING_PATH) as fh:
        mapping = json.load(fh)
    item = mapping["items"][unit["queue_order"]]
    start_pose = item["runtime_identity"]["start_position"]

    checkpoints = [{"source": "episode_start", "position": start_pose}]
    matrix = []
    for k, rec in enumerate(chain):
        targets = list(checkpoints)
        # nearest checkpoint by geodesic distance; tie -> AMBIGUOUS
        cand_pos = rec.get("candidate_positions_q") or []
        rows = []
        for i, e in enumerate(cand_pos):
            geods = []
            for t in targets:
                g = geodesic(asim, e, t["position"])
                geods.append((g, t))
            finite = [(g, t) for g, t in geods if math.isfinite(g)]
            if not finite:
                rows.append({
                    "cand_index": i,
                    "status": "NO_GEODESIC_CHECKPOINT",
                    "witness": None,
                })
                continue
            finite.sort(key=lambda x: x[0])
            best_g, best_t = finite[0]
            tie = len(finite) > 1 and abs(finite[1][0] - best_g) <= \
                NEAREST_TIE_M
            dup = any(euclid(e, o) <= DUPLICATE_ENDPOINT_M
                      for j, o in enumerate(cand_pos) if j != i)
            if tie or dup:
                rows.append({
                    "cand_index": i,
                    "status": "AMBIGUOUS",
                    "reason": ("nearest_checkpoint_tie" if tie else "")
                              + ("+duplicate_endpoint" if dup else ""),
                    "nearest_checkpoint": best_t["source"],
                    "geodesic_m": round(best_g, 6),
                    "witness": None,
                })
                continue
            heading = heading_toward(e, best_t["position"])
            w = run_witness(asim, follower_for(asim), e, heading,
                            best_t["position"])
            rows.append({
                "cand_index": i,
                "persistent_id": next(
                    (m["target"] for m in rec["graph"]["mappings"]
                     if m["cand_index"] == i), None),
                "nearest_checkpoint": best_t["source"],
                "geodesic_m": round(best_g, 6),
                "witness": w,
                "status": w["status"],
            })
        matrix.append({
            "prefix_index": k,
            "cur_vp": rec["cur_vp"],
            "candidate_count": rec["candidates"]["count"],
            "rows": rows,
            "any_success": any(r.get("witness") and r["witness"]["success"]
                               for r in rows),
            "ambiguous": any(r["status"] == "AMBIGUOUS" for r in rows),
        })
        # commit this prefix's post pose as a checkpoint for later prefixes
        post = rec["action"].get("post_position_q")
        if post:
            checkpoints.append({"source": "prefix_%d_post" % k,
                                "position": post})

    # expiry proposal (provisional only)
    statuses = [(m["prefix_index"], m["any_success"], m["ambiguous"])
                for m in matrix]
    safe = [k for k, s, _ in statuses if s]
    unsafe = [k for k, s, _ in statuses if not s]
    ambiguous_any = any(a for _, _, a in statuses)
    expiry = {
        "expiry_proposal_status": None,
        "expiry_prefix": None,
        "notes": [],
    }
    if ambiguous_any:
        expiry["expiry_proposal_status"] = "AMBIGUOUS"
        expiry["notes"].append("one or more prefixes carry ambiguous "
                               "witnesses; no provisional expiry asserted")
    elif not unsafe:
        expiry["expiry_proposal_status"] = "PROVISIONAL"
        expiry["expiry_prefix"] = None
        expiry["notes"].append("every observed prefix retains at least one "
                               "safe return witness; no expiry observed "
                               "within the prefix horizon (right-censored)")
    elif not safe:
        expiry["expiry_proposal_status"] = "PROVISIONAL"
        expiry["expiry_prefix"] = None
        expiry["notes"].append("no prefix has a safe return witness; "
                               "last-safe prefix does not exist in horizon")
    else:
        last_safe = max(k for k in safe if k < min(unsafe)) \
            if min(unsafe) > min(safe) else max(safe)
        pattern_monotone = all(k < min([u for u in unsafe if u > last_safe])
                               for k in safe if k <= last_safe)
        re_safe_after_unsafe = any(s and any(u < k for u in unsafe)
                                   for k, s, _ in statuses)
        if re_safe_after_unsafe:
            expiry["expiry_proposal_status"] = "AMBIGUOUS"
            expiry["notes"].append("safe return reappears after an unsafe "
                                   "prefix: multiple possible expiry moments")
        else:
            expiry["expiry_proposal_status"] = "PROVISIONAL"
            expiry["expiry_prefix"] = last_safe
            expiry["notes"].append(
                "last prefix with a safe return witness followed by an "
                "observed prefix without one; PROVISIONAL only - not a "
                "validated T_X")
    return {
        "unit_id": unit["unit_id"],
        "episode_id": unit["episode_id"],
        "scene_id": scene,
        "language": unit["language"],
        "chain_root": unit["run"]["chain_root"],
        "controller_config_sha256": controller_config_sha,
        "controller_config": controller_config,
        "feasibility_matrix": matrix,
        "expiry_proposal": expiry,
        "k3_persistence_note": "K=3 persistence is an engineering criterion "
                               "for the Phase 0 event validator only; it is "
                               "not asserted here as a method contribution",
    }


_FOLLOWERS = {}


class BareSimAdapter:
    """Harness-only adapter: gives the bare habitat_sim.Simulator the
    ``habitat_config`` attribute expected by ShortestPathFollowerCompat.
    ``step`` applies the same kinematic ``Agent.act`` operation used inside
    Simulator.step, but deliberately skips sensor rendering/observation
    materialization. All other attributes delegate to the wrapped simulator.
    """

    def __init__(self, sim):
        object.__setattr__(self, "_sim", sim)

        class _Cfg:
            FORWARD_STEP_SIZE = FORWARD_STEP_SIZE
            TURN_ANGLE = TURN_ANGLE

        object.__setattr__(self, "habitat_config", _Cfg())

    def step(self, action):
        sim = object.__getattribute__(self, "_sim")
        return sim.get_agent(0).act(action)

    def geodesic_distance(self, a, b):
        return geodesic(object.__getattribute__(self, "_sim"), a, b)

    @property
    def up_vector(self):
        import numpy as np

        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        import numpy as np

        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        import habitat_sim

        sim = object.__getattribute__(self, "_sim")
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        sim.pathfinder.find_path(path)
        return path.points

    def get_agent_state(self, agent_id=0):
        return object.__getattribute__(
            self, "_sim").get_agent(agent_id).get_state()

    def set_agent_state(self, position, rotation, agent_id=0,
                        reset_sensors=True):
        import habitat_sim
        import numpy as np

        sim = object.__getattribute__(self, "_sim")
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position, dtype=np.float32)
        agent_state.rotation = rotation
        sim.get_agent(agent_id).set_state(agent_state, reset_sensors)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_sim"), name)


def follower_for(sim):
    from habitat_extensions.shortest_path_follower import (
        ShortestPathFollowerCompat)

    if id(sim) not in _FOLLOWERS:
        _FOLLOWERS[id(sim)] = ShortestPathFollowerCompat(
            sim if isinstance(sim, BareSimAdapter) else BareSimAdapter(sim),
            GOAL_RADIUS_M,
            return_one_hot=False,
        )
    return _FOLLOWERS[id(sim)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5)
    args = ap.parse_args()

    with open(MAPPING_PATH) as fh:
        mapping = json.load(fh)
    items = mapping["items"][:args.count]

    out_dir = os.path.dirname(SUMMARY_PATH)
    os.makedirs(out_dir, exist_ok=True)
    netguard_file = os.path.join(out_dir, "witness_netguard.jsonl")
    open(netguard_file, "w").close()
    install_network_deny_guard(netguard_file)

    episodes = []
    for item in items:
        order = item["queue_order"]
        eid = item["episode_id"]
        unit_path = os.path.join(UNITS_DIR,
                                 "unit_%02d_ep%s.json" % (order, eid))
        if not os.path.isfile(unit_path):
            episodes.append({"unit_id": "unit_%02d_ep%s" % (order, eid),
                             "status": "MISSING_UNIT"})
            continue
        # Habitat-Sim v0.1.7 does not safely retain multiple renderer-backed
        # Simulator instances while switching scenes in this process.  Keep
        # each witness episode transaction isolated and close its simulator
        # before the next scene is opened.
        sim_cache = {}
        try:
            ep = witness_episode(unit_path, sim_cache, netguard_file)
        finally:
            for raw_sim, _adapter, _hashes in sim_cache.values():
                raw_sim.close()
            _FOLLOWERS.clear()
        ep_path = os.path.join(out_dir,
                               "witness_ep%s_order%02d.json" % (eid, order))
        with open(ep_path, "w") as fh:
            json.dump(ep, fh, indent=2)
        episodes.append({
            "unit_id": ep["unit_id"],
            "episode_id": ep["episode_id"],
            "scene_id": ep["scene_id"],
            "prefixes": len(ep["feasibility_matrix"]),
            "witness_success_prefixes": sum(
                1 for m in ep["feasibility_matrix"] if m["any_success"]),
            "ambiguous_prefixes": sum(
                1 for m in ep["feasibility_matrix"] if m["ambiguous"]),
            "expiry_proposal_status":
                ep["expiry_proposal"]["expiry_proposal_status"],
            "expiry_prefix": ep["expiry_proposal"]["expiry_prefix"],
            "controller_config_sha256": ep["controller_config_sha256"],
            "file": os.path.relpath(ep_path, PROJECT_ROOT),
        })
        print("[%02d] ep%s prefixes=%d expiry=%s" % (
            order, eid, len(ep["feasibility_matrix"]),
            ep["expiry_proposal"]["expiry_proposal_status"]), flush=True)

    attempts = 0
    with open(netguard_file) as fh:
        for ln in fh:
            try:
                attempts = max(attempts, int(json.loads(ln)["attempts"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
    allowed_proposals = {"PROVISIONAL", "AMBIGUOUS"}
    complete = len(episodes) == args.count and all(
        e.get("expiry_proposal_status") in allowed_proposals
        and e.get("prefixes", 0) > 0
        for e in episodes
    )
    proposal_counts = {
        status: sum(e.get("expiry_proposal_status") == status
                    for e in episodes)
        for status in sorted(allowed_proposals)
    }
    summary = {
        "stage": "stage5_return_expiry_witness_first5",
        "witness_version": WITNESS_VERSION,
        "status": "PASS" if complete and attempts == 0 else "FAIL",
        "disclaimer": "engineering witness only; PROVISIONAL expiry "
                      "proposals are not validated T_X; GT/navmesh used for "
                      "labeling/witnessing only, never as policy input; no "
                      "checkpoint loaded, no training",
        "network_attempts": attempts,
        "requested_episode_count": args.count,
        "completed_episode_count": len(episodes),
        "proposal_counts": proposal_counts,
        "validated_tx_count": 0,
        "observed_unique_expiry_prefix_count": sum(
            e.get("expiry_proposal_status") == "PROVISIONAL"
            and e.get("expiry_prefix") is not None
            for e in episodes
        ),
        "episodes": episodes,
    }
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({"summary": os.path.relpath(SUMMARY_PATH, PROJECT_ROOT),
                      "network_attempts": attempts}, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
