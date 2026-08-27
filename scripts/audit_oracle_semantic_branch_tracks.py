#!/usr/bin/env python3
"""Fail-closed semantic-region audit for MF2-CR1 oracle events.

This turns each provisional reference-route turn into a fixed directed
navmesh exit region and checks the first K exposed prefixes against every
other provisional target in the same episode.  A track is admitted only when
its oracle candidate uniquely belongs to its own target region with the
pre-registered 0.25m separation margin at all K prefixes.

This is geometric machine evidence only. It never labels the instruction as
branch-dependent and never fills a human-review field.
"""

import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict


ROOT = "/mnt/daiyang/vla"
SCRIPTS = os.path.join(ROOT, "scripts")
HABSIM = os.path.join(ROOT, "third_party", "habitat-sim")
for _path in (SCRIPTS, HABSIM):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from phase0c_oracle_lowlevel_probe import (  # noqa: E402
    K, MP3D, RAY_OFFSETS_DEG, SEPARATION_MARGIN_M, TARGET_TUBE_M,
    build_lowlevel_trace, oracle_candidates, segment_distance,
)


PROBE = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                     "PHASE0C_ORACLE_LOWLEVEL_PROBE.json")
COST = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                    "PHASE0C_COST_FRONTIER_WITNESS.json")
IDENTITY = os.path.join(ROOT, "artifacts", "runtime",
                        "phase0_correctness",
                        "IDENTITY_V3_RERUN_SUMMARY.json")
MAPPING = os.path.join(ROOT, "artifacts", "phase0",
                       "REVEAL_QUEUE_50_MAPPING.json")
RXR_TRAIN = os.path.join(
    ROOT, "third_party", "ETP-R1", "data", "datasets",
    "RxR_VLNCE_v0_enc_xlmr", "train", "train_guide.json.gz")
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "ORACLE_SEMANTIC_BRANCH_TRACK_AUDIT.json")
EXPECTED = {
    PROBE: "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac",
    COST: "9b59ea9b7b9995aeb604b00587dd79a3af863ea51b16ddfe805b3f719f1a16d1",
    IDENTITY: "cf4e5d51b1052bf789ae9747bfaf8136a9438e526b2c0206fadb0ec0afe59109",
}
GATE_MIN_EVENTS = 15
GATE_MIN_SCENES = 10
PROGRESS_MIN = 0.05


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def q3(point):
    return [round(float(value), 3) for value in point]


def branch_id(scene, start, end):
    payload = {"scene_id": scene, "directed_start_q": q3(start),
               "directed_end_q": q3(end)}
    return "exit_" + hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def main():
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise SystemExit("input SHA drift: " + path)
    probe = json.load(open(PROBE))
    cost = json.load(open(COST))
    identity = json.load(open(IDENTITY))
    mapping = json.load(open(MAPPING))
    if probe.get("status") != "PASS" or len(probe["events"]) != 104:
        raise SystemExit("unexpected oracle probe")
    if not cost.get("gates", {}).get("gate3_complete_cost_evidence"):
        raise SystemExit("cost witness incomplete")
    if identity.get("status") != "ENGINEERING_PASS" or identity[
            "counts"]["full_50_traces_after_reusing_16_v1_ok"] != 50:
        raise SystemExit("numeric identity closure missing")

    wanted = {str(item["episode_id"]) for item in mapping["items"]}
    instruction_sha = {str(item["episode_id"]):
                       item["instruction_sha256_queue"]
                       for item in mapping["items"]}
    with gzip.open(RXR_TRAIN, "rt") as fh:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(fh)["episodes"]
                    if str(item["episode_id"]) in wanted}
    by_episode = defaultdict(list)
    for event in probe["events"]:
        by_episode[str(event["episode_id"])].append(event)
    cost_ids = {event["provisional_event_id"] for event in cost["events"]
                if event.get("status") == "COMPLETE"}

    import habitat_sim

    pathfinders = {}
    trace_cache = {}
    candidate_cache = {}
    audited = []
    for episode_id, episode_events in sorted(by_episode.items()):
        episode = episodes[episode_id]
        scene = episode_events[0]["scene_id"]
        if scene not in pathfinders:
            pathfinder = habitat_sim.PathFinder()
            navmesh = os.path.join(MP3D, scene, scene + ".navmesh")
            if not pathfinder.load_nav_mesh(navmesh):
                raise RuntimeError("navmesh load failed: " + navmesh)
            pathfinders[scene] = pathfinder
        pathfinder = pathfinders[scene]
        trace = build_lowlevel_trace(pathfinder, episode)
        candidates = [oracle_candidates(pathfinder, row["position"],
                                        row["heading"]) for row in trace]
        trace_cache[episode_id] = trace
        candidate_cache[episode_id] = candidates
        targets = {}
        reference = episode["reference_path"]
        for event in episode_events:
            index = int(event["reference_turn_index"])
            start = [float(x) for x in pathfinder.snap_point(reference[index])]
            end = [float(x) for x in pathfinder.snap_point(
                reference[index + 1])]
            targets[event["provisional_event_id"]] = {
                "start": start, "end": end,
                "branch_id": branch_id(scene, start, end),
            }

        for event in episode_events:
            event_id = event["provisional_event_id"]
            first_run = event["stable_exposure_runs"][0]
            prefixes = list(range(int(first_run[0]), int(first_run[0]) + K))
            prefix_records, reasons = [], []
            target = targets[event_id]
            for prefix in prefixes:
                evidence = event["prefix_evidence"][prefix]
                offset = float(evidence["best_offset_deg"])
                matches = [candidate for candidate in candidates[prefix]
                           if float(candidate["offset_deg"]) == offset]
                if len(matches) != 1:
                    reasons.append("best_candidate_reconstruction_failed")
                    continue
                candidate = matches[0]
                region_scores = []
                for other_id, other in targets.items():
                    distance, progress = segment_distance(
                        candidate["endpoint"], other["start"], other["end"])
                    if distance <= TARGET_TUBE_M and progress > PROGRESS_MIN:
                        region_scores.append({
                            "event_id": other_id,
                            "branch_id": other["branch_id"],
                            "distance_m": float(distance),
                            "progress": float(progress),
                        })
                region_scores.sort(key=lambda value: (
                    value["distance_m"], value["event_id"]))
                own = [value for value in region_scores
                       if value["event_id"] == event_id]
                own_first = bool(own) and region_scores[0][
                    "event_id"] == event_id
                margin = (region_scores[1]["distance_m"] -
                          region_scores[0]["distance_m"]
                          if len(region_scores) > 1 else math.inf)
                unique = own_first and margin >= SEPARATION_MARGIN_M
                if not unique:
                    reasons.append("cross_target_semantic_ambiguity")
                prefix_records.append({
                    "prefix_index": prefix,
                    "best_offset_deg": offset,
                    "candidate_endpoint_q": q3(candidate["endpoint"]),
                    "qualifying_target_count": len(region_scores),
                    "own_target_is_unique_best": unique,
                    "nearest_second_target_margin_m":
                        round(margin, 6) if math.isfinite(margin) else None,
                    "qualifying_targets": [{
                        "event_id": value["event_id"],
                        "branch_id": value["branch_id"],
                        "distance_m": round(value["distance_m"], 6),
                        "progress": round(value["progress"], 6),
                    } for value in region_scores],
                })
            reasons = sorted(set(reasons))
            admitted = (len(prefix_records) == K and not reasons and
                        event_id in cost_ids)
            audited.append({
                "provisional_event_id": event_id,
                "episode_id": episode_id,
                "scene_id": scene,
                "instruction_sha256": instruction_sha[episode_id],
                "semantic_branch_id": target["branch_id"],
                "target_exit_region": {
                    "directed_start_q": q3(target["start"]),
                    "directed_end_q": q3(target["end"]),
                    "tube_radius_m": TARGET_TUBE_M,
                    "minimum_progress": PROGRESS_MIN,
                },
                "k_prefixes": prefixes,
                "prefix_records": prefix_records,
                "numeric_identity_trace_protocol":
                    "persistent-branch-identity/v3-engineering",
                "numeric_identity_is_same_candidate_system": False,
                "cost_witness_complete": event_id in cost_ids,
                "machine_geometric_semantic_status":
                    "ADMITTED" if admitted else "EXCLUDED_AMBIGUOUS",
                "exclusion_reasons": reasons,
                "human_language_review_status": "PENDING",
            })

    admitted = [event for event in audited if event[
        "machine_geometric_semantic_status"] == "ADMITTED"]
    admitted_scenes = {event["scene_id"] for event in admitted}
    ambiguity_zero = all(all(prefix["own_target_is_unique_best"]
                             for prefix in event["prefix_records"])
                         for event in admitted)
    floor_pass = (len(admitted) >= GATE_MIN_EVENTS and
                  len(admitted_scenes) >= GATE_MIN_SCENES)
    geometric_pass = ambiguity_zero and floor_pass
    exclusions = Counter(reason for event in audited
                         for reason in event["exclusion_reasons"])
    output = {
        "gate": "mf2_cr1_oracle_semantic_branch_track",
        "revision": "oracle-semantic-branch-track/1",
        "status": "PASS" if geometric_pass else "FAIL",
        "decision": "MACHINE_GEOMETRIC_SEMANTIC_SUBGATE_PASS" if
                    geometric_pass else "SEMANTIC_TRACK_NO_GO",
        "scope": "RxR-train frozen queue; directed navmesh exit regions; "
                 "first K=3 exposed prefixes; no language validity claim",
        "inputs": {
            "oracle_probe": {"path": os.path.relpath(PROBE, ROOT),
                             "sha256": sha256_file(PROBE)},
            "cost_witness": {"path": os.path.relpath(COST, ROOT),
                             "sha256": sha256_file(COST)},
            "numeric_identity": {"path": os.path.relpath(IDENTITY, ROOT),
                                 "sha256": sha256_file(IDENTITY)},
        },
        "pre_registered_contract": {
            "target": "directed post-turn reference segment tube",
            "tube_radius_m": TARGET_TUBE_M,
            "minimum_segment_progress": PROGRESS_MIN,
            "cross_target_separation_margin_m": SEPARATION_MARGIN_M,
            "persistence_k": K,
            "minimum_admitted_events": GATE_MIN_EVENTS,
            "minimum_admitted_scenes": GATE_MIN_SCENES,
            "oracle_ray_offsets_deg": list(RAY_OFFSETS_DEG),
            "fail_closed_on_any_first_k_cross_target_ambiguity": True,
        },
        "counts": {
            "provisional_events": len(audited),
            "machine_geometric_admitted": len(admitted),
            "excluded": len(audited) - len(admitted),
            "admitted_episodes": len({event["episode_id"]
                                      for event in admitted}),
            "admitted_scenes": len(admitted_scenes),
            "semantic_ambiguity_among_admitted": 0 if ambiguity_zero else 1,
            "exclusion_reasons": dict(exclusions),
        },
        "gates": {
            "admitted_event_scene_floor_pass": floor_pass,
            "semantic_ambiguity_zero_among_admitted": ambiguity_zero,
            "geometric_semantic_subgate_pass": geometric_pass,
            "language_branch_dependence_gate_pass": False,
            "automatic_candidate_to_semantic_track_gate_pass": False,
            "full_gate6_pass": False,
        },
        "events": audited,
        "non_conclusions": {
            "validated_reveal_events": 0,
            "instruction_branch_dependence_validated": False,
            "human_review_performed": False,
            "automatic_etp_candidate_semantic_link_validated": False,
            "training_authorized": False,
            "benchmark_result": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "counts": output["counts"], "gates": output["gates"],
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if geometric_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
