#!/usr/bin/env python3
"""Conservative Oracle Ego-FOV feasibility probe for MF2-CR1 Gate 2.

Candidate construction is deterministic and uses only the current recorded
pose/heading plus the public scene navmesh:
  * 63-degree ego-FOV represented by rays at -30, 0, +30 degrees;
  * each ray requests a 3m no-sliding navmesh step;
  * endpoints moving <0.5m are discarded.

A provisional fixed branch is a >=45-degree RxR reference-route turn.  Its
target region is a 1m tube around the post-turn segment.  At a prefix the
branch is exposed only when the unique best current oracle endpoint is inside
that tube with >=0.25m margin.  K=3 consecutive exposed prefixes are required.

This is RxR-train engineering evidence only.  Reference paths/navmesh are used
offline and are never model inputs.  No checkpoint, policy, image, hidden view,
or val/test split is accessed.
"""

import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter


ROOT = "/mnt/daiyang/vla"
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHASE0C_ORACLE_EGOFOV_PROBE.json")
STAGE4 = os.path.join(ROOT, "artifacts", "runtime",
                      "phase0_reveal_closure",
                      "STAGE4_TRACE_GENERATION_SUMMARY.json")
MAPPING = os.path.join(ROOT, "artifacts", "phase0",
                       "REVEAL_QUEUE_50_MAPPING.json")
TX_AUDIT = os.path.join(ROOT, "artifacts", "runtime",
                        "phase0_correctness", "TX_FEASIBILITY_AUDIT.json")
REVISION = os.path.join(ROOT,
                        "METHOD_FREEZE_2_CORRECTNESS_REVISION_1.md")
RXR_TRAIN = os.path.join(
    ROOT, "third_party", "ETP-R1", "data", "datasets",
    "RxR_VLNCE_v0_enc_xlmr", "train", "train_guide.json.gz")
MP3D = os.path.join(ROOT, "third_party", "ETP-R1", "data",
                    "scene_datasets", "mp3d")

EXPECTED = {
    "artifacts/runtime/phase0_correctness/TX_FEASIBILITY_AUDIT.json":
        "b24926f11f78e8ec6ecf78f18b4a48f99cffd1e041c8c06ac295f86d393e7472",
    "artifacts/runtime/phase0_correctness/IDENTITY_V3_RERUN_SUMMARY.json":
        "cf4e5d51b1052bf789ae9747bfaf8136a9438e526b2c0206fadb0ec0afe59109",
    "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
}

FOV_DEG = 63.0
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


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def xz(a, b):
    return math.hypot(float(a[0]) - float(b[0]),
                      float(a[2]) - float(b[2]))


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
    q = (a[0] + t * vx, a[2] + t * vz)
    return math.hypot(point[0] - q[0], point[2] - q[1]), t


def chain_path(item):
    order, eid = int(item["queue_order"]), str(item["episode_id"])
    if item["status"] == "OK":
        return os.path.join(
            ROOT, "artifacts", "runtime", "phase0_reveal_closure", "collect",
            "rpc50_%02d_ep%s" % (order, eid), "reveal_prefix_chain.jsonl")
    return os.path.join(
        ROOT, "artifacts", "runtime", "phase0_correctness", "collect_v3",
        "rpcv3_%02d_ep%s" % (order, eid), "reveal_prefix_chain.jsonl")


def oracle_candidates(pathfinder, position, heading):
    import numpy as np

    start = pathfinder.snap_point(np.asarray(position, dtype="float32"))
    result = []
    for offset in RAY_OFFSETS_DEG:
        angle = float(heading) + math.radians(offset)
        desired = np.asarray([
            start[0] - RAY_LENGTH_M * math.sin(angle),
            start[1],
            start[2] - RAY_LENGTH_M * math.cos(angle),
        ], dtype="float32")
        endpoint = pathfinder.try_step_no_sliding(start, desired)
        moved = xz(start, endpoint)
        if moved >= MIN_CANDIDATE_MOVE_M:
            result.append({"offset_deg": offset,
                           "endpoint": [float(x) for x in endpoint],
                           "moved_m": moved})
    return result


def longest_runs(values):
    runs = []
    start = None
    for i, value in enumerate(values + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    preflight = []
    for rel, expected in EXPECTED.items():
        actual = sha256_file(os.path.join(ROOT, rel))
        preflight.append({"path": rel, "expected_sha256": expected,
                          "actual_sha256": actual,
                          "pass": actual == expected})
    if not all(x["pass"] for x in preflight):
        raise SystemExit("preflight SHA mismatch")
    if not os.path.isfile(REVISION):
        raise SystemExit("MF2-CR1 missing")
    tx = load_json(TX_AUDIT)
    if tx.get("decision") != \
            "METHOD_FREEZE_2_NO_GO_WITHOUT_CORRECTNESS_REVISION":
        raise SystemExit("unexpected T_X audit decision")

    stage = load_json(STAGE4)
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
    source_turns = 0
    candidate_prefixes = 0
    for item in stage["items"]:
        eid, scene = str(item["episode_id"]), item["scene_id"]
        if scene not in pfs:
            pf = habitat_sim.PathFinder()
            navmesh = os.path.join(MP3D, scene, scene + ".navmesh")
            if not pf.load_nav_mesh(navmesh):
                raise RuntimeError("navmesh load failed: " + navmesh)
            pfs[scene] = pf
        pf = pfs[scene]
        chain = load_jsonl(chain_path(item))
        ref = episodes[eid].get("reference_path") or []
        # Cache candidates: fixed current pose/heading only; no future/hidden
        # observation can enter this deterministic constructor.
        per_prefix_candidates = [
            oracle_candidates(pf, rec["agent_pose"]["position_q"],
                              rec["agent_pose"]["heading_q"])
            for rec in chain]
        candidate_prefixes += len(per_prefix_candidates)
        for j in range(1, len(ref) - 1):
            angle = turn_angle(ref[j - 1], ref[j], ref[j + 1])
            if (angle is None or angle < TURN_THRESHOLD_DEG
                    or xz(ref[j - 1], ref[j]) < MIN_SEGMENT_M
                    or xz(ref[j], ref[j + 1]) < MIN_SEGMENT_M):
                continue
            source_turns += 1
            cp = pf.snap_point(ref[j])
            prefix_to_cp = [xz(rec["agent_pose"]["position_q"], cp)
                            for rec in chain]
            if min(prefix_to_cp, default=math.inf) > ENCOUNTER_RADIUS_M:
                continue
            exposed = []
            prefix_evidence = []
            for k, candidates in enumerate(per_prefix_candidates):
                scores = []
                for cand in candidates:
                    distance, progress = segment_distance(
                        cand["endpoint"], ref[j], ref[j + 1])
                    scores.append((distance, -progress, cand))
                scores.sort(key=lambda x: (x[0], x[1], x[2]["offset_deg"]))
                if not scores:
                    ok, reason, best, margin = False, "NO_CANDIDATE", None, None
                else:
                    best = scores[0]
                    margin = (scores[1][0] - best[0]
                              if len(scores) > 1 else math.inf)
                    ok = (best[0] <= TARGET_TUBE_M
                          and best[1] < -0.05
                          and margin >= SEPARATION_MARGIN_M)
                    if best[0] > TARGET_TUBE_M:
                        reason = "OUTSIDE_TARGET_TUBE"
                    elif best[1] >= -0.05:
                        reason = "NO_POST_TURN_PROGRESS"
                    elif margin < SEPARATION_MARGIN_M:
                        reason = "COMPETING_ORACLE_CANDIDATE"
                    else:
                        reason = "EXPOSED"
                exposed.append(ok)
                prefix_evidence.append({
                    "prefix_index": k, "status": reason,
                    "candidate_count": len(candidates),
                    "best_target_tube_distance_m":
                        round(best[0], 6) if best else None,
                    "best_progress": round(-best[1], 6) if best else None,
                    "separation_margin_m":
                        round(margin, 6) if margin is not None and
                        math.isfinite(margin) else None,
                    "best_offset_deg": best[2]["offset_deg"] if best else None,
                })
            all_runs = longest_runs(exposed)
            k_runs = [x for x in all_runs if x[1] - x[0] + 1 >= K]
            if k_runs:
                events.append({
                    "provisional_event_id": "ep%s_turn%02d" % (eid, j),
                    "episode_id": eid, "scene_id": scene,
                    "reference_turn_index": j,
                    "turn_angle_deg": round(angle, 6),
                    "k": K, "stable_exposure_runs": k_runs,
                    "candidate_reveal_prefix": k_runs[0][0],
                    "prefix_evidence": prefix_evidence,
                    "label_status": "PROVISIONAL_GEOMETRIC_ONLY",
                })

    scenes = sorted({x["scene_id"] for x in events})
    episode_ids = sorted({x["episode_id"] for x in events})
    gate2 = len(events) >= GATE_MIN_EVENTS and len(scenes) >= GATE_MIN_SCENES
    status_counts = Counter(
        p["status"] for event in events for p in event["prefix_evidence"])
    output = {
        "gate": "mf2_cr1_phase0c_oracle_egofov_feasibility",
        "revision": "oracle-egofov-probe/1",
        "status": "PASS" if gate2 else "FAIL",
        "gate2_oracle_event_pass": gate2,
        "scope": "RxR train; geometric oracle screening only; no semantic "
                 "or language-evidence validation",
        "preflight": {"pass": True, "files": preflight,
                      "mf2_cr1_sha256": sha256_file(REVISION)},
        "frozen_constructor": {
            "hfov_deg": FOV_DEG,
            "ray_offsets_deg": list(RAY_OFFSETS_DEG),
            "ray_length_m": RAY_LENGTH_M,
            "min_candidate_move_m": MIN_CANDIDATE_MOVE_M,
            "turn_threshold_deg": TURN_THRESHOLD_DEG,
            "min_adjacent_segment_m": MIN_SEGMENT_M,
            "target_tube_m": TARGET_TUBE_M,
            "separation_margin_m": SEPARATION_MARGIN_M,
            "k": K,
            "encounter_radius_m": ENCOUNTER_RADIUS_M,
            "candidate_method": "PathFinder.try_step_no_sliding from current "
                                "pose at -30/0/+30 degrees; navmesh oracle",
            "causal_boundary": "constructor receives only current recorded "
                               "pose/heading; no RGB/depth/future/panorama",
        },
        "counts": {
            "queue_episodes": 50,
            "candidate_prefixes": candidate_prefixes,
            "eligible_reference_turns_before_encounter_filter": source_turns,
            "provisional_k3_events": len(events),
            "episodes_with_event": len(episode_ids),
            "scenes_with_event": len(scenes),
            "gate_min_events": GATE_MIN_EVENTS,
            "gate_min_scenes": GATE_MIN_SCENES,
            "admitted_prefix_status_counts": dict(status_counts),
        },
        "events": events,
        "decision": "CONTINUE_TO_COST_WITNESS" if gate2 else
                    "PHASE0C_GATE2_NO_GO",
        "non_conclusions": {
            "semantic_branch_validity": False,
            "language_evidence_closure": False,
            "validated_reveal_event_count": 0,
            "validated_tx_count": 0,
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
        "counts": output["counts"],
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if gate2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
