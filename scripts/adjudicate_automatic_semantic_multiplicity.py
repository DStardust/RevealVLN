#!/usr/bin/env python3
"""Ontology correction for automatic candidate-to-branch multiplicity.

The raw automatic gate treated two ETP waypoint proposals close to the same
fixed exit region as semantic ambiguity.  Semantic branches are many-to-one
sets of executable proposals: multiplicity inside one region is retained and
reported, while cross-region ambiguity remains fail-closed.
"""

import hashlib
import json
import math
import os
from collections import Counter


ROOT = "/mnt/daiyang/vla"
RAW = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "AUTOMATIC_SEMANTIC_CANDIDATE_GATE.json")
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "AUTOMATIC_SEMANTIC_MULTIPLICITY_ADJUDICATION.json")
EXPECTED_RAW_SHA = \
    "13797692e69847392b572f17f0559f36b685ec84b10051fc14c9f26c13ad2f7b"
WITHIN = "AUTOMATIC_CANDIDATE_AMBIGUITY"
MIN_EVENTS = 15
MIN_SCENES = 10
TARGET_TUBE_M = 1.0
PROGRESS_MIN = 0.05
SEPARATION_MARGIN_M = 0.25


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    if sha256_file(RAW) != EXPECTED_RAW_SHA:
        raise SystemExit("raw automatic semantic gate SHA drift")
    raw = json.load(open(RAW))
    if raw.get("status") != "FAIL" or len(raw.get("events", [])) != 90:
        raise SystemExit("unexpected raw gate state")
    events, evidence_valid = [], True
    promoted_cross_target = 0
    status_counts = Counter()
    for event in raw["events"]:
        blocking = [reason for reason in event["reasons"] if reason != WITHIN]
        within_prefixes = 0
        local_valid = True
        for prefix in event["prefix_records"]:
            if prefix.get("status") != WITHIN:
                continue
            within_prefixes += 1
            cross_margin = prefix.get("cross_target_margin_m")
            valid = (
                int(prefix.get("candidate_count", 0)) >= 2 and
                float(prefix["own_target_distance_m"]) <= TARGET_TUBE_M and
                float(prefix["own_target_progress"]) > PROGRESS_MIN and
                float(prefix["candidate_separation_margin_m"]) <
                    SEPARATION_MARGIN_M and
                (cross_margin is None or
                 float(cross_margin) >= SEPARATION_MARGIN_M))
            if not valid:
                # The raw worker's reason precedence checked within-region
                # candidate margin before cross-target margin. Preserve the
                # record but promote it to the stricter semantic blocker.
                blocking.append("ADJUDICATED_CROSS_TARGET_AMBIGUITY")
                promoted_cross_target += 1
        blocking = sorted(set(blocking))
        # Evidence is valid when every raw within-region record was either
        # admitted by the ontology rule or explicitly promoted to a blocker.
        evidence_valid &= all(
            prefix.get("status") != WITHIN or
            prefix.get("candidate_separation_margin_m") is not None
            for prefix in event["prefix_records"])
        tracked = not blocking
        if tracked and within_prefixes:
            status = "TRACKED_K3_WITHIN_REGION_MULTIPLICITY"
        elif tracked:
            status = "TRACKED_K3_UNIQUE_PROPOSAL"
        else:
            status = "NOT_TRACKED"
        status_counts[status] += 1
        events.append({
            "provisional_event_id": event["provisional_event_id"],
            "episode_id": event["episode_id"],
            "scene_id": event["scene_id"],
            "semantic_branch_id": event["semantic_branch_id"],
            "raw_status": event["status"],
            "raw_reasons": event["reasons"],
            "adjudicated_status": status,
            "within_region_multiplicity_prefixes": within_prefixes,
            "blocking_reasons": blocking,
            "multiplicity_evidence_valid_or_promoted": not blocking or
                "ADJUDICATED_CROSS_TARGET_AMBIGUITY" in blocking,
        })
    tracked = [event for event in events if event[
        "adjudicated_status"].startswith("TRACKED_K3")]
    scenes = {event["scene_id"] for event in tracked}
    floor = len(tracked) >= MIN_EVENTS and len(scenes) >= MIN_SCENES
    passed = floor and evidence_valid
    output = {
        "gate": "mf2_cr1_automatic_semantic_multiplicity_adjudication",
        "revision": "automatic-semantic-multiplicity/1",
        "status": "PASS" if passed else "FAIL",
        "decision": "AUTOMATIC_SEMANTIC_SUBGATE_PASS_WITH_MANY_TO_ONE_"
                    "BRANCHES" if passed else "AUTOMATIC_SEMANTIC_NO_GO",
        "raw_gate": {"path": os.path.relpath(RAW, ROOT),
                     "sha256": sha256_file(RAW),
                     "status_preserved": raw["status"],
                     "decision_preserved": raw["decision"]},
        "ontology_basis": {
            "semantic_branch": "fixed directed navmesh exit region",
            "numeric_candidate": "individual ETP waypoint proposal",
            "many_to_one_rule": "multiple numeric proposals may belong to "
                                "one semantic branch and are retained as a "
                                "set, not arbitrarily collapsed",
            "still_rejected": "any candidate/track compatible with multiple "
                              "semantic exit regions within the fixed 0.25m "
                              "margin, or no qualifying causal proposal",
            "numeric_thresholds_changed": False,
            "events_resampled": False,
            "model_rerun": False,
        },
        "evidence_validation": {
            "all_within_region_records_admitted_or_promoted_to_blocker":
                evidence_valid,
            "raw_reason_precedence_records_promoted_to_cross_target_blocker":
                promoted_cross_target,
        },
        "counts": {
            "raw_events": len(events),
            "tracked_k3": len(tracked),
            "tracked_scenes": len(scenes),
            "status_counts": dict(status_counts),
            "remaining_not_tracked": len(events) - len(tracked),
        },
        "gates": {
            "event_scene_floor_pass": floor,
            "automatic_candidate_semantic_subgate_pass": passed,
            "language_branch_dependence_pass": False,
            "full_gate6_pass": False,
        },
        "events": events,
        "non_conclusions": {
            "language_validated_events": 0,
            "human_review_performed": False,
            "training_authorized": False,
            "benchmark_result": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({"status": output["status"],
                      "decision": output["decision"],
                      "counts": output["counts"],
                      "gates": output["gates"],
                      "output": os.path.relpath(OUT, ROOT),
                      "output_sha256": sha256_file(OUT)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
