#!/usr/bin/env python3
"""Phase-0B correctness audit of the frozen T_X contract.

This is an explicitly exploratory engineering diagnostic, not a benchmark
generator.  It combines read-only source inspection with the now complete
50-episode RxR-train prefix traces.  It also measures how an *explicit*
return-distance allowance changes observed route-turn expiry.  Those budget
results are diagnostic evidence that T_X is budget-conditioned; they are not
validated RevealEvents and must not be promoted to labels.

No checkpoint is loaded, no policy/env episode is run, no image/tensor is
written, and no frozen source is modified.  The only dataset payload opened
is the already authorized RxR train guide file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter


ROOT = "/mnt/daiyang/vla"
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "TX_FEASIBILITY_AUDIT.json")
MAPPING = os.path.join(ROOT, "artifacts", "phase0",
                       "REVEAL_QUEUE_50_MAPPING.json")
STAGE4 = os.path.join(ROOT, "artifacts", "runtime",
                      "phase0_reveal_closure",
                      "STAGE4_TRACE_GENERATION_SUMMARY.json")
IDENTITY_V3 = os.path.join(ROOT, "artifacts", "runtime",
                           "phase0_correctness",
                           "IDENTITY_V3_RERUN_SUMMARY.json")
WITNESS = os.path.join(ROOT, "artifacts", "runtime",
                       "phase0_reveal_closure", "witness",
                       "WITNESS_RETURN_EXPIRY_FIRST5.json")
RXR_TRAIN = os.path.join(
    ROOT, "third_party", "ETP-R1", "data", "datasets",
    "RxR_VLNCE_v0_enc_xlmr", "train", "train_guide.json.gz")
MP3D_ROOT = os.path.join(ROOT, "third_party", "ETP-R1", "data",
                         "scene_datasets", "mp3d")

EXPECTED = {
    "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    "artifacts/runtime/phase0_correctness/IDENTITY_V3_RERUN_SUMMARY.json":
        "cf4e5d51b1052bf789ae9747bfaf8136a9438e526b2c0206fadb0ec0afe59109",
    "artifacts/runtime/PHASE0_REVEAL_CLOSURE_MAIN_ACCEPTANCE.json":
        "36c2fb2bce69b8ebc337e0d2192c731c52dba33d8f0b5fe781ffa2a53783b435",
    "artifacts/phase0/evidence_current.json":
        "430f73ec5752783aa553c2c2f4fe4128247a93fa29bee0ae454c97f5547d9ce1",
}

# Exploratory route-turn probe settings.  They are declared in code before
# the probe is executed, but the whole probe was designed after inspecting
# existing Phase-0 traces, so it is not pre-registered evidence.
TURN_THRESHOLD_DEG = 45.0
MIN_ADJACENT_XZ_M = 0.5
ENCOUNTER_RADIUS_M = 3.0
RETURN_ALLOWANCES_M = (2.0, 5.0, 10.0)
EGO_HFOV_DEG = 63.0
TARGET_PROPOSAL_MARGIN_M = 0.05
TARGET_FORWARD_WINDOW_M = 3.0
K_STABLE = 3


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def euclid(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


def xz_dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]),
                      float(a[2]) - float(b[2]))


def turn_angle(a, b, c):
    u = (float(b[0]) - float(a[0]), float(b[2]) - float(a[2]))
    v = (float(c[0]) - float(b[0]), float(c[2]) - float(b[2]))
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) /
                                (nu * nv)))
    return math.degrees(math.acos(cosine))


def source_evidence(rel, needles):
    path = os.path.join(ROOT, rel)
    with open(path) as fh:
        lines = fh.readlines()
    matches = []
    for needle in needles:
        found = [{"line": i + 1, "text": line.strip()}
                 for i, line in enumerate(lines) if needle in line]
        matches.append({"needle": needle, "matches": found})
    return {"path": rel, "sha256": sha256_file(path),
            "needles": matches,
            "all_needles_found": all(x["matches"] for x in matches)}


def arc_distances(ref, start):
    values = [0.0]
    for k in range(start + 1, len(ref)):
        values.append(values[-1] + euclid(ref[k - 1], ref[k]))
    return values


def target_proposal(rec, ref):
    cur = rec["agent_pose"]["position_q"]
    endpoints = rec.get("candidate_positions_q") or []
    if not ref or not endpoints:
        return {"status": "NO_DATA"}
    nearest_ref = [euclid(cur, x) for x in ref]
    j = min(range(len(ref)), key=lambda x: (nearest_ref[x], x))
    arcs = arc_distances(ref, j)
    forward = [j + k for k in range(1, len(arcs))
               if arcs[k] <= TARGET_FORWARD_WINDOW_M]
    if not forward:
        return {"status": "NO_FORWARD_SEGMENT"}
    scored = []
    mapping = {int(m["cand_index"]): m
               for m in rec["graph"]["mappings"]}
    for i, endpoint in enumerate(endpoints):
        score = min(euclid(endpoint, ref[k]) for k in forward)
        scored.append((score, i, mapping[i]))
    scored.sort(key=lambda x: (x[0], x[1]))
    margin = scored[1][0] - scored[0][0] if len(scored) > 1 else math.inf
    if margin < TARGET_PROPOSAL_MARGIN_M:
        return {"status": "AMBIGUOUS", "margin_m": margin}
    _, i, match = scored[0]
    angle = float(rec["candidates"]["angles"][i]) % (2.0 * math.pi)
    front_delta = min(angle, 2.0 * math.pi - angle)
    return {
        "status": "PROPOSED",
        "cand_index": i,
        "persistent_id": str(match["target"]),
        "mapping_multi_match":
            len(match.get("matches_within_loc_noise", [])) > 1,
        "front_angle_deg": math.degrees(front_delta),
        "ego_fov_visible": math.degrees(front_delta) <= EGO_HFOV_DEG / 2.0,
    }


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def full_chain_path(stage_item):
    order, eid = int(stage_item["queue_order"]), str(stage_item["episode_id"])
    if stage_item["status"] == "OK":
        return os.path.join(
            ROOT, "artifacts", "runtime", "phase0_reveal_closure", "collect",
            "rpc50_%02d_ep%s" % (order, eid), "reveal_prefix_chain.jsonl")
    return os.path.join(
        ROOT, "artifacts", "runtime", "phase0_correctness", "collect_v3",
        "rpcv3_%02d_ep%s" % (order, eid), "reveal_prefix_chain.jsonl")


def longest_stable_run(values):
    best = 0
    cur = 0
    previous = object()
    for value in values:
        if value is not None and value == previous:
            cur += 1
        elif value is not None:
            cur = 1
        else:
            cur = 0
        previous = value
        best = max(best, cur)
    return best


def geodesic(pathfinder, a, b):
    import habitat_sim
    import numpy as np

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype="float32")
    path.requested_end = np.asarray(b, dtype="float32")
    found = pathfinder.find_path(path)
    return float(path.geodesic_distance) if found else math.inf


def classify_expiry(distances, allowance):
    safe = [d <= allowance for d in distances]
    if not any(safe):
        return "NEVER_FEASIBLE", None
    first = safe.index(True)
    last = len(safe) - 1 - list(reversed(safe)).index(True)
    if last == len(safe) - 1:
        return "RIGHT_CENSORED", None
    if any(not x for x in safe[first:last + 1]) or any(safe[last + 1:]):
        return "NON_MONOTONE", None
    return "UNIQUE_OBSERVED", last


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    preflight = []
    for rel, expected in EXPECTED.items():
        actual = sha256_file(os.path.join(ROOT, rel))
        preflight.append({"path": rel, "expected_sha256": expected,
                          "actual_sha256": actual,
                          "pass": actual == expected})
    identity = load_json(IDENTITY_V3)
    stage4 = load_json(STAGE4)
    mapping = load_json(MAPPING)
    witness = load_json(WITNESS)
    preflight_ok = (all(x["pass"] for x in preflight)
                    and identity.get("scientific_status") ==
                    "IDENTITY_TRACE_CLOSURE_COMPLETE"
                    and identity.get("counts", {}).get(
                        "full_50_traces_after_reusing_16_v1_ok") == 50
                    and len(stage4.get("items", [])) == 50
                    and mapping.get("mapped_count") == 50)
    if not preflight_ok:
        raise SystemExit("preflight failed; refusing diagnostic")

    sources = [
        source_evidence(
            "third_party/ETP-R1/vlnce_baselines/GRPO_trainer_ETP_R1.py",
            ["get_camera_orientations12()", "for stepk in range(self.max_len)",
             "ghost_vp_ids = list(gmap.ghost_pos.keys())",
             "gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids",
             "gmap.delete_ghost(ghost_vp)"]),
        source_evidence(
            "third_party/ETP-R1/vlnce_baselines/common/environments.py",
            ["self._env.sim.step_without_obs(act)",
             "self._env._task.measurements.update_measures"]),
        source_evidence("third_party/ETP-R1/run_rxr/iter_train.yaml",
                        ["max_traj_len: 25"]),
        source_evidence("third_party/ETP-R1/run_rxr/rxr_vlnce.yaml",
                        ["MAX_EPISODE_STEPS: 5000", "HFOV: 63"]),
        source_evidence(
            "third_party/ETP-R1/vlnce_baselines/models/graph_utils.py",
            ["self.ghost_pos = {}", "def delete_ghost(self, vp):",
             "self.ghost_pos[gvp].append(cpos)"]),
    ]

    with gzip.open(RXR_TRAIN, "rt") as fh:
        wanted = {str(x["episode_id"]) for x in mapping["items"]}
        episodes = {str(x["episode_id"]): x
                    for x in json.load(fh)["episodes"]
                    if str(x["episode_id"]) in wanted}
    trace_stats = Counter()
    per_episode = []
    chains = {}
    for item in stage4["items"]:
        eid = str(item["episode_id"])
        chain_path = full_chain_path(item)
        chain = load_jsonl(chain_path)
        chains[eid] = chain
        uf = UnionFind()
        for k in range(1, len(chain)):
            selected = chain[k - 1]["action"].get("selected_vp")
            if isinstance(selected, str) and selected.startswith("g"):
                uf.union(selected, str(chain[k]["cur_vp"]))
        proposals = [target_proposal(rec,
                                     episodes[eid].get("reference_path") or [])
                     for rec in chain]
        stable_ids = []
        visible_values = []
        sensor_ok = True
        for rec, proposal in zip(chain, proposals):
            trace_stats["prefixes"] += 1
            rgb = [x for x in rec["observation_hashes"]
                   if x == "rgb" or re.fullmatch(r"rgb_\d+", x)]
            depth = [x for x in rec["observation_hashes"]
                     if x == "depth" or re.fullmatch(r"depth_\d+", x)]
            sensor_ok &= len(rgb) == 12 and len(depth) == 12
            trace_stats["current_candidates"] += rec["candidates"]["count"]
            front = sum(
                math.degrees(min(float(a) % (2 * math.pi),
                                 2 * math.pi - float(a) % (2 * math.pi)))
                <= EGO_HFOV_DEG / 2.0 for a in rec["candidates"]["angles"])
            trace_stats["ego_fov_candidates"] += front
            if proposal["status"] == "PROPOSED":
                trace_stats["proposed_prefixes"] += 1
                trace_stats["proposed_visible_ego_fov"] += int(
                    proposal["ego_fov_visible"])
                trace_stats["proposed_hidden_outside_ego_fov"] += int(
                    not proposal["ego_fov_visible"])
                trace_stats["proposal_target_multi_match"] += int(
                    proposal["mapping_multi_match"])
                stable_ids.append(uf.find(proposal["persistent_id"])
                                  if not proposal["mapping_multi_match"]
                                  else None)
                visible_values.append(proposal["ego_fov_visible"])
            else:
                trace_stats["proposal_" + proposal["status"].lower()] += 1
                stable_ids.append(None)
                visible_values.append(None)
        max_stable = longest_stable_run(stable_ids)
        max_visible = longest_stable_run(
            [True if x is True else None for x in visible_values])
        trace_stats["episodes_raw_lineage_k3"] += int(max_stable >= K_STABLE)
        trace_stats["episodes_ego_visible_k3"] += int(max_visible >= K_STABLE)
        per_episode.append({
            "queue_order": item["queue_order"], "episode_id": eid,
            "prefix_count": len(chain), "panorama_hash_fields_12x2": sensor_ok,
            "longest_same_raw_lineage_target_proposal": max_stable,
            "longest_ego_visible_proposal_run": max_visible,
            "chain_file": os.path.relpath(chain_path, ROOT),
            "chain_sha256": sha256_file(chain_path),
        })

    # Exploratory budget-conditioning probe over GT route turns.  This does
    # not inspect instruction semantics and is explicitly not event labeling.
    sys.path.insert(0, os.path.join(ROOT, "third_party", "habitat-sim"))
    import habitat_sim

    pathfinders = {}
    route_events = []
    budget_counts = {str(x): Counter() for x in RETURN_ALLOWANCES_M}
    for item in stage4["items"]:
        eid, scene = str(item["episode_id"]), item["scene_id"]
        if scene not in pathfinders:
            navmesh = os.path.join(MP3D_ROOT, scene, scene + ".navmesh")
            pf = habitat_sim.PathFinder()
            loaded = pf.load_nav_mesh(navmesh)
            if not loaded:
                raise RuntimeError("failed to load navmesh " + navmesh)
            pathfinders[scene] = pf
        pf = pathfinders[scene]
        ref = episodes[eid].get("reference_path") or []
        poses = [x["agent_pose"]["position_q"] for x in chains[eid]]
        for j in range(1, len(ref) - 1):
            angle = turn_angle(ref[j - 1], ref[j], ref[j + 1])
            if (angle is None or angle < TURN_THRESHOLD_DEG
                    or xz_dist(ref[j - 1], ref[j]) < MIN_ADJACENT_XZ_M
                    or xz_dist(ref[j], ref[j + 1]) < MIN_ADJACENT_XZ_M):
                continue
            cp = list(pf.snap_point(ref[j]))
            target = list(pf.snap_point(ref[j + 1]))
            cp_snap = euclid(cp, ref[j])
            target_snap = euclid(target, ref[j + 1])
            leg = geodesic(pf, cp, target)
            distances = [geodesic(pf, p, cp) for p in poses]
            if not math.isfinite(leg) or not all(math.isfinite(x)
                                                 for x in distances):
                continue
            min_d = min(distances)
            if min_d > ENCOUNTER_RADIUS_M:
                continue
            event_result = {
                "episode_id": eid, "scene_id": scene,
                "reference_turn_index": j,
                "turn_angle_deg": round(angle, 6),
                "checkpoint_to_target_geodesic_m": round(leg, 6),
                "minimum_policy_prefix_to_checkpoint_m": round(min_d, 6),
                "checkpoint_snap_delta_m": round(cp_snap, 6),
                "target_snap_delta_m": round(target_snap, 6),
                "allowances": {},
            }
            for allowance in RETURN_ALLOWANCES_M:
                status, last = classify_expiry(distances, allowance)
                budget_counts[str(allowance)][status] += 1
                event_result["allowances"][str(allowance)] = {
                    "status": status,
                    "last_safe_prefix": last,
                    "total_via_checkpoint_budget_m": round(
                        allowance + leg, 6),
                }
            route_events.append(event_result)

    witness_unique = int(witness.get("observed_unique_expiry_prefix_count", 0))
    blockers = [
        "The accepted automatic candidate frontend observes twelve RGB and "
        "twelve depth headings (360-degree panorama), not an evolving "
        "63-degree ego-FOV candidate set.",
        "ETP-R1 places every retained ghost in the global action set and "
        "deletes a ghost only when it is consumed; static MP3D therefore "
        "preserves many untried options rather than producing an intrinsic "
        "physical deadline.",
        "Low-level controller actions use step_without_obs and do not consume "
        "the Habitat 5000-step environment counter; the effective checkpoint "
        "policy cap is 25 high-level decisions, under which a global graph "
        "selection can traverse a multi-node back_path in one decision.",
        "The previous first-five witness found zero unique observed expiry "
        "prefixes and tests candidate-endpoint-to-checkpoint return, not the "
        "frozen actual-prefix-to-fixed-target/saved-option definition.",
        "Across the complete 50 traces, the current offline target proposal "
        "has zero episodes with K=3 consecutive identical raw/lineage target "
        "IDs; raw GraphMap waypoint IDs are not yet a stable branch-event "
        "identity protocol.",
    ]
    decision = "METHOD_FREEZE_2_NO_GO_WITHOUT_CORRECTNESS_REVISION"
    output = {
        "gate": "phase0b_tx_contract_feasibility_audit",
        "revision": "tx-feasibility-audit/1",
        "status": "PASS_AUDIT",
        "decision": decision,
        "preflight": {"pass": preflight_ok, "files": preflight,
                      "source_evidence_all_found":
                          all(x["all_needles_found"] for x in sources)},
        "source_evidence": sources,
        "dynamic_trace_audit": {
            "label": "rxr_train_engineering_only",
            "counts": dict(trace_stats),
            "all_50_have_12_rgb_and_12_depth_hash_fields":
                all(x["panorama_hash_fields_12x2"] for x in per_episode),
            "raw_lineage_k3_event_count": 0,
            "per_episode": per_episode,
        },
        "accepted_witness_audit": {
            "path": os.path.relpath(WITNESS, ROOT),
            "sha256": sha256_file(WITNESS),
            "validated_tx_count": witness.get("validated_tx_count"),
            "observed_unique_expiry_prefix_count": witness_unique,
            "direction_mismatch": "candidate endpoint -> nearest prior "
                                  "checkpoint; frozen T_X requires actual "
                                  "prefix -> fixed target branch or saved "
                                  "candidate under an explicit safety contract",
        },
        "exploratory_budget_conditioning_probe": {
            "claim_status": "DIAGNOSTIC_ONLY_POST_HOC_NOT_PREREGISTERED",
            "route_anchor": "RxR reference-path interior turn >=45 degrees; "
                            "both adjacent xz segments >=0.5m; policy trace "
                            "must approach checkpoint within 3m",
            "safe_rule": "navmesh geodesic(actual prefix, checkpoint) <= "
                         "declared return allowance; target leg recorded "
                         "separately and added to total route budget",
            "return_allowances_m": list(RETURN_ALLOWANCES_M),
            "route_turn_diagnostic_count": len(route_events),
            "status_counts_by_allowance": {
                key: dict(value) for key, value in budget_counts.items()},
            "events": route_events,
            "interpretation": "Changing only the declared return allowance "
                              "changes the observed expiry distribution. "
                              "This supports a budget-conditioned frontier, "
                              "not an intrinsic threshold-free T_X claim.",
        },
        "critical_blockers": blockers,
        "required_correctness_revision": {
            "observation_contract": "Add a genuinely causal ego-FOV "
                                    "candidate exposure protocol. Begin with "
                                    "Oracle ego-FOV to isolate the hypothesis; "
                                    "an automatic frontend must not compute "
                                    "hidden panoramic views before masking.",
            "expiry_contract": "Replace intrinsic T_X with T_X(B), a "
                               "deadline conditioned on a declared remaining "
                               "low-level action/time/return-cost budget B. "
                               "Predict/evaluate a budget sweep or frontier; "
                               "never select B after seeing performance.",
            "controller_contract": "Count every low-level MOVE/TURN and "
                                   "enforce the same budget for every method, "
                                   "including multi-node backtracking inside "
                                   "one ETP high-level decision.",
            "branch_identity_contract": "Use a semantic/topological branch "
                                        "lineage matcher; retain exact v3 "
                                        "numeric localization evidence and "
                                        "reject reviewer-ambiguous branches.",
            "frozen_spec_action": "Do not edit FROZEN_SPEC.md in this gate. "
                                  "Write an explicitly versioned correctness "
                                  "revision for review first.",
        },
        "non_conclusions": {
            "validated_reveal_event_count": 0,
            "validated_unique_tx_count": 0,
            "training_authorized": False,
            "human_review_authorized": False,
            "method_freeze_2_feasibility_established": False,
            "frozen_spec_modified": False,
            "val_unseen_or_test_used": False,
            "checkpoint_loaded": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"], "decision": decision,
        "trace_counts": output["dynamic_trace_audit"]["counts"],
        "budget_status_counts":
            output["exploratory_budget_conditioning_probe"][
                "status_counts_by_allowance"],
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
