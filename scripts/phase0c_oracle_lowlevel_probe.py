#!/usr/bin/env python3
"""MF2-CR1 Oracle Ego-FOV probe on a counted low-level prefix clock.

The prior high-level-trace probe is retained as a failed gate.  This follow-up
changes only the prefix clock: an offline oracle trace follows each public RxR
reference route using 0.25m MOVE prefixes and 30-degree TURN prefixes.  The
same branch/event thresholds and the same >=15 events / >=10 scenes gate are
used unchanged.

This script is geometric train-only feasibility evidence.  It does not render
observations, run/load ETP-R1, create semantic labels, or authorize training.
"""

import gzip
import hashlib
import json
import math
import os
import sys


ROOT = "/mnt/daiyang/vla"
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHASE0C_ORACLE_LOWLEVEL_PROBE.json")
HIGHLEVEL = os.path.join(ROOT, "artifacts", "runtime",
                         "phase0_correctness",
                         "PHASE0C_ORACLE_EGOFOV_PROBE.json")
MAPPING = os.path.join(ROOT, "artifacts", "phase0",
                       "REVEAL_QUEUE_50_MAPPING.json")
RXR_TRAIN = os.path.join(
    ROOT, "third_party", "ETP-R1", "data", "datasets",
    "RxR_VLNCE_v0_enc_xlmr", "train", "train_guide.json.gz")
MP3D = os.path.join(ROOT, "third_party", "ETP-R1", "data",
                    "scene_datasets", "mp3d")

EXPECTED_HIGHLEVEL_SHA = \
    "97f0de47610bf4f388cdf2527d702b3c248e3fdb345a05fb6ed1b81d6e566f99"
EXPECTED_REVISION_SHA = None  # recorded, not pinned across documentation only

MOVE_M = 0.25
TURN_DEG = 30.0
RAY_OFFSETS_DEG = (-30.0, 0.0, 30.0)
RAY_LENGTH_M = 3.0
MIN_CANDIDATE_MOVE_M = 0.5
TURN_THRESHOLD_DEG = 45.0
MIN_SEGMENT_M = 0.5
TARGET_TUBE_M = 1.0
SEPARATION_MARGIN_M = 0.25
K = 3
ENCOUNTER_RADIUS_M = 3.0
GATE_MIN_EVENTS = 15
GATE_MIN_SCENES = 10


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def xz(a, b):
    return math.hypot(float(a[0]) - float(b[0]),
                      float(a[2]) - float(b[2]))


def euclid(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


def absolute_heading(a, b):
    dx, dz = b[0] - a[0], b[2] - a[2]
    r = max(math.hypot(dx, dz), 1e-9)
    h = math.asin(-dx / r)
    if b[2] > a[2]:
        h = math.pi - h
    return h % (2 * math.pi)


def signed_delta(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def turn_angle(a, b, c):
    u = (b[0] - a[0], b[2] - a[2])
    v = (c[0] - b[0], c[2] - b[2])
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    value = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) /
                               (nu * nv)))
    return math.degrees(math.acos(value))


def segment_distance(point, a, b):
    vx, vz = b[0] - a[0], b[2] - a[2]
    wx, wz = point[0] - a[0], point[2] - a[2]
    denom = vx * vx + vz * vz
    t = max(0.0, min(1.0, (wx * vx + wz * vz) / denom)) \
        if denom > 1e-12 else 0.0
    qx, qz = a[0] + t * vx, a[2] + t * vz
    return math.hypot(point[0] - qx, point[2] - qz), t


def resample_segment(a, b, step=MOVE_M):
    distance = euclid(a, b)
    if distance < 1e-9:
        return []
    n = int(math.floor(distance / step))
    points = []
    for i in range(1, n + 1):
        t = min(1.0, i * step / distance)
        points.append([a[j] + t * (b[j] - a[j]) for j in range(3)])
    if not points or euclid(points[-1], b) > 1e-4:
        points.append([float(x) for x in b])
    return points


def build_lowlevel_trace(pathfinder, episode):
    import habitat_sim

    ref = episode.get("reference_path") or []
    if not ref:
        return []
    # The route polyline comes from navmesh shortest paths between consecutive
    # public reference anchors.  Each recorded prefix is exactly one counted
    # TURN or <=0.25m MOVE in this offline oracle clock.
    current = [float(x) for x in pathfinder.snap_point(ref[0])]
    heading = 0.0
    trace = [{"position": current, "heading": heading,
              "action": "START", "action_count": 0}]
    count = 0
    for goal in ref[1:]:
        path = habitat_sim.ShortestPath()
        path.requested_start = current
        path.requested_end = pathfinder.snap_point(goal)
        if not pathfinder.find_path(path) or len(path.points) < 2:
            return []
        for waypoint in path.points[1:]:
            waypoint = [float(x) for x in waypoint]
            desired = absolute_heading(current, waypoint)
            delta = signed_delta(desired, heading)
            # Match a 30-degree discrete action clock.  Residual angle remains
            # as it would for a controller that moves once within one turn.
            turns = int(math.floor((abs(delta) + 1e-9) /
                                   math.radians(TURN_DEG)))
            direction = 1.0 if delta >= 0 else -1.0
            for _ in range(turns):
                heading = (heading + direction * math.radians(TURN_DEG)) % \
                    (2 * math.pi)
                count += 1
                trace.append({"position": list(current), "heading": heading,
                              "action": "TURN", "action_count": count})
            for point in resample_segment(current, waypoint):
                current = point
                count += 1
                trace.append({"position": list(current), "heading": heading,
                              "action": "MOVE", "action_count": count})
    return trace


def oracle_candidates(pathfinder, position, heading):
    import numpy as np

    start = pathfinder.snap_point(position)
    result = []
    for offset in RAY_OFFSETS_DEG:
        angle = heading + math.radians(offset)
        desired = np.asarray([
            start[0] - RAY_LENGTH_M * math.sin(angle), start[1],
            start[2] - RAY_LENGTH_M * math.cos(angle)], dtype="float32")
        endpoint = pathfinder.try_step_no_sliding(start, desired)
        moved = xz(start, endpoint)
        if moved >= MIN_CANDIDATE_MOVE_M:
            result.append({"offset_deg": offset,
                           "endpoint": [float(x) for x in endpoint],
                           "moved_m": moved})
    return result


def runs(values):
    result, start = [], None
    for i, value in enumerate(values + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            result.append((start, i - 1))
            start = None
    return result


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if sha256_file(HIGHLEVEL) != EXPECTED_HIGHLEVEL_SHA:
        raise SystemExit("high-level probe SHA drift")
    high = load_json(HIGHLEVEL)
    if high.get("decision") != "PHASE0C_GATE2_NO_GO":
        raise SystemExit("high-level gate was not retained as failed")
    mapping = load_json(MAPPING)
    wanted = {str(x["episode_id"]) for x in mapping["items"]}
    with gzip.open(RXR_TRAIN, "rt") as fh:
        episodes = {str(x["episode_id"]): x
                    for x in json.load(fh)["episodes"]
                    if str(x["episode_id"]) in wanted}

    sys.path.insert(0, os.path.join(ROOT, "third_party", "habitat-sim"))
    import habitat_sim

    pfs = {}
    events = []
    traces = []
    for item in mapping["items"]:
        eid, scene = str(item["episode_id"]), item["scene_id"]
        if scene not in pfs:
            pf = habitat_sim.PathFinder()
            navmesh = os.path.join(MP3D, scene, scene + ".navmesh")
            if not pf.load_nav_mesh(navmesh):
                raise RuntimeError("navmesh load failed: " + navmesh)
            pfs[scene] = pf
        pf = pfs[scene]
        ep = episodes[eid]
        ref = ep.get("reference_path") or []
        trace = build_lowlevel_trace(pf, ep)
        if not trace:
            traces.append({"episode_id": eid, "status": "TRACE_FAIL"})
            continue
        candidates = [oracle_candidates(pf, x["position"], x["heading"])
                      for x in trace]
        trace_digest = hashlib.sha256(json.dumps(
            trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        traces.append({"episode_id": eid, "scene_id": scene,
                       "status": "OK", "prefix_count": len(trace),
                       "turn_prefixes": sum(x["action"] == "TURN"
                                            for x in trace),
                       "move_prefixes": sum(x["action"] == "MOVE"
                                            for x in trace),
                       "trace_sha256": trace_digest})
        for j in range(1, len(ref) - 1):
            angle = turn_angle(ref[j - 1], ref[j], ref[j + 1])
            if (angle is None or angle < TURN_THRESHOLD_DEG
                    or xz(ref[j - 1], ref[j]) < MIN_SEGMENT_M
                    or xz(ref[j], ref[j + 1]) < MIN_SEGMENT_M):
                continue
            cp = pf.snap_point(ref[j])
            if min(xz(x["position"], cp) for x in trace) > \
                    ENCOUNTER_RADIUS_M:
                continue
            exposed = []
            evidence = []
            for k, current in enumerate(candidates):
                scored = []
                for candidate in current:
                    distance, progress = segment_distance(
                        candidate["endpoint"], ref[j], ref[j + 1])
                    scored.append((distance, -progress, candidate))
                scored.sort(key=lambda x: (x[0], x[1],
                                            x[2]["offset_deg"]))
                if not scored:
                    ok, status, best, margin = False, "NO_CANDIDATE", None, None
                else:
                    best = scored[0]
                    margin = scored[1][0] - best[0] \
                        if len(scored) > 1 else math.inf
                    ok = (best[0] <= TARGET_TUBE_M and best[1] < -0.05
                          and margin >= SEPARATION_MARGIN_M)
                    status = ("OUTSIDE_TARGET_TUBE"
                              if best[0] > TARGET_TUBE_M else
                              "NO_POST_TURN_PROGRESS" if best[1] >= -0.05 else
                              "COMPETING_ORACLE_CANDIDATE"
                              if margin < SEPARATION_MARGIN_M else "EXPOSED")
                exposed.append(ok)
                evidence.append({
                    "prefix_index": k, "action": trace[k]["action"],
                    "status": status,
                    "best_target_tube_distance_m":
                        round(best[0], 6) if best else None,
                    "separation_margin_m": round(margin, 6)
                    if margin is not None and math.isfinite(margin) else None,
                    "best_offset_deg": best[2]["offset_deg"] if best else None,
                })
            k_runs = [r for r in runs(exposed) if r[1] - r[0] + 1 >= K]
            if k_runs:
                events.append({
                    "provisional_event_id": "ep%s_turn%02d" % (eid, j),
                    "episode_id": eid, "scene_id": scene,
                    "reference_turn_index": j,
                    "turn_angle_deg": round(angle, 6),
                    "k": K, "stable_exposure_runs": k_runs,
                    "candidate_reveal_prefix": k_runs[0][0],
                    "prefix_evidence": evidence,
                    "label_status": "PROVISIONAL_GEOMETRIC_ONLY",
                })

    scene_count = len({x["scene_id"] for x in events})
    episode_count = len({x["episode_id"] for x in events})
    gate = len(events) >= GATE_MIN_EVENTS and scene_count >= GATE_MIN_SCENES
    output = {
        "gate": "mf2_cr1_phase0c_oracle_egofov_lowlevel_clock",
        "revision": "oracle-egofov-lowlevel-probe/1",
        "status": "PASS" if gate else "FAIL",
        "gate2_oracle_event_pass": gate,
        "retained_highlevel_result": {
            "path": os.path.relpath(HIGHLEVEL, ROOT),
            "sha256": sha256_file(HIGHLEVEL),
            "status": high["status"], "decision": high["decision"],
            "provisional_k3_events": high["counts"][
                "provisional_k3_events"],
        },
        "only_changed_variable": "prefix clock: one <=0.25m MOVE or one "
                                 "30-degree TURN per causal prefix",
        "constructor": {
            "move_m": MOVE_M, "turn_deg": TURN_DEG,
            "ray_offsets_deg": list(RAY_OFFSETS_DEG),
            "ray_length_m": RAY_LENGTH_M,
            "min_candidate_move_m": MIN_CANDIDATE_MOVE_M,
            "turn_threshold_deg": TURN_THRESHOLD_DEG,
            "min_segment_m": MIN_SEGMENT_M,
            "target_tube_m": TARGET_TUBE_M,
            "separation_margin_m": SEPARATION_MARGIN_M,
            "k": K, "encounter_radius_m": ENCOUNTER_RADIUS_M,
            "thresholds_unchanged_from_highlevel_probe": True,
        },
        "counts": {
            "queue_episodes": 50,
            "successful_oracle_traces": sum(x["status"] == "OK"
                                            for x in traces),
            "total_lowlevel_prefixes": sum(x.get("prefix_count", 0)
                                            for x in traces),
            "provisional_k3_events": len(events),
            "episodes_with_event": episode_count,
            "scenes_with_event": scene_count,
            "gate_min_events": GATE_MIN_EVENTS,
            "gate_min_scenes": GATE_MIN_SCENES,
        },
        "traces": traces,
        "events": events,
        "decision": "CONTINUE_TO_COST_WITNESS" if gate else
                    "PHASE0C_LOWLEVEL_GATE2_NO_GO",
        "non_conclusions": {
            "semantic_branch_validity": False,
            "language_evidence_closure": False,
            "validated_reveal_event_count": 0,
            "validated_tx_count": 0,
            "frozen_controller_cost_witnessed": False,
            "automatic_frontend_gate_pass": False,
            "training_authorized": False,
            "human_review_authorized": False,
            "frozen_spec_modified": False,
            "val_unseen_or_test_used": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "counts": output["counts"], "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
