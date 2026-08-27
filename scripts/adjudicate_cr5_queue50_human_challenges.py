#!/usr/bin/env python3
"""Adjudicate three human-challenged rejects and emit one correction candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
PRIMARY = BASE / "multiview_primary"
REVIEW = BASE / "human_review_fast"
OLD_GEOMETRY = PRIMARY / "CR5_QUEUE50_DIRECTED_GEOMETRY.json"
HUMAN_REVIEW = REVIEW / "daiyang_auto_reject16.jsonl"
HUMAN_ACCEPTANCE = REVIEW / "CR5_QUEUE50_AUTO_REJECT_HUMAN_ACCEPTANCE.json"
DIAGNOSTIC = REVIEW / "CR5_QUEUE50_HUMAN_CHALLENGE_GEOMETRY_DIAGNOSTIC.json"
OUT = REVIEW / "CR5_QUEUE50_HUMAN_CHALLENGE_ADJUDICATION.json"
CORRECTED = REVIEW / "CR5_QUEUE50_Q36_CORRECTED_GEOMETRY.json"
EXPECTED = {
    OLD_GEOMETRY: "46609126537ddb9c4936bc93d683dd06243b91203d6bf2f7c03f30cca7deb850",
    HUMAN_REVIEW: "fcf17ff60bd9e07fa4e66a83741ea47136b8725bde97603cda03aed76f34f5ff",
    HUMAN_ACCEPTANCE: "4fb4bc45a2fbdd65fa80922c18c18b8ceed16a910dab67191f50e58defdfebd2",
    DIAGNOSTIC: "5cab848045e19c81bd9285b773d8eb4768ecf187b4f979c0ce9500e8f5108c8e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected):
            raise SystemExit("pinned adjudication source drift: " + str(path))
    human_rows = {row["event_id"]: row for row in (
        json.loads(line) for line in HUMAN_REVIEW.read_text().splitlines()
        if line.strip())}
    challenged = {event_id: row for event_id, row in human_rows.items()
                  if row["final_label"] == "SUSPECT_FALSE_REJECT"}
    diagnostic = {row["event_id"]: row for row in
                  load(DIAGNOSTIC)["events"]}
    old_doc = load(OLD_GEOMETRY)
    old = {row["event_id"]: row for row in old_doc["events"]}
    if set(challenged) != set(diagnostic):
        raise SystemExit("human/diagnostic challenge set drift")

    decisions = []
    for event_id in sorted(challenged):
        found = diagnostic[event_id]["selected_executable_alternative"]
        if found is None:
            decision = "ORIGINAL_REJECT_CONFIRMED_BY_3D_COUNTERFACTUAL"
        else:
            decision = "CORRECTED_GEOMETRY_CONTROLLER_REQUIRED"
        decisions.append({
            "event_id": event_id,
            "human_label": challenged[event_id]["final_label"],
            "human_comment_zh": challenged[event_id]["comment_zh"],
            "diagnostic_status": diagnostic[event_id]["diagnostic_status"],
            "decision": decision,
            "training_label": False,
        })
    if {row["event_id"] for row in decisions
            if row["decision"] == "CORRECTED_GEOMETRY_CONTROLLER_REQUIRED"} \
            != {"q36_ep1049_hv05"}:
        raise SystemExit("unexpected correction set")

    event_id = "q36_ep1049_hv05"
    old_event = old[event_id]
    selected = diagnostic[event_id]["selected_executable_alternative"]
    best = selected["best_candidate"]
    corrected_event = copy.deepcopy(old_event)
    corrected_event["status"] = "GEOMETRY_PASS_CONTROLLER_REQUIRED"
    corrected_event["failures"] = []
    corrected_event["alternative"] = {
        "branch_id": selected["branch_id"],
        "visual_descriptor": selected["visual_descriptor"],
        "horizontal_direction": selected["horizontal_direction"],
        "vertical_motion": "LEVEL",
        "Q": old_event["trace"]["Q"],
        "B_i_at_1m": best["B_i_at_1m"],
        "T_i_at_1_75m": best["T_i_at_1_75m"],
        "navmesh_shortest_path_length_m": best["path_length_m"],
        "direction_error_deg": best["direction_error_deg"],
        "vertical_delta_m": round(
            best["T_i_at_1_75m"][1] - old_event["trace"]["Q"][1], 6),
        "distinctness": best["distinctness"],
        "path_samples": best["path_samples"],
        "search_score": best["score"],
    }
    corrected_event["alternative_search"] = [{
        "branch_id": selected["branch_id"],
        "human_challenge_triggered_search": True,
        "executable_candidate_path_count": selected[
            "executable_candidate_count"],
        "distinct_candidate_found": True,
        "distinct_nonincoming_candidate_count": selected[
            "distinct_nonincoming_candidate_count"],
    }]
    corrected_event["geometry_verified"] = True
    corrected_event["correctness_revision"] = {
        "revision": "cr5-target-route-authority-correction/1",
        "reason": (
            "The original verifier skipped alternative search after a target "
            "direction-description mismatch. The target geometry is defined "
            "by the official reference future; the human challenge confirms "
            "the semantic target/alternative scene, and the alternative is "
            "now independently required to pass all frozen 3-D distinctness "
            "and nonincoming constraints."
        ),
        "original_target_direction_error_deg": old_event["target"][
            "direction_error_deg"],
        "target_direction_description_reclassified_as_diagnostic": True,
        "target_reference_route_unchanged": True,
        "human_review_path": str(HUMAN_REVIEW.relative_to(ROOT)),
        "human_review_sha256": EXPECTED[HUMAN_REVIEW],
        "diagnostic_path": str(DIAGNOSTIC.relative_to(ROOT)),
        "diagnostic_sha256": EXPECTED[DIAGNOSTIC],
    }

    corrected = {
        "manifest": "MF2-CR5 q36 corrected directed 3-D geometry",
        "revision": "cr5-q36-corrected-directed-geometry/1",
        "status": "COMPLETE_CONTROLLER_GATE_REQUIRED",
        "scope": "one human-challenged RxR-train queue50 event",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "thresholds": old_doc["thresholds"],
        "candidate_count": 1,
        "status_counts": {"GEOMETRY_PASS_CONTROLLER_REQUIRED": 1},
        "events": [corrected_event],
        "network_calls_made": 0,
        "controller_rollouts_made": 0,
        "forbidden_split_payloads_opened": 0,
        "original_geometry_artifact_modified": False,
        "training_authorized": False,
    }
    atomic_json(CORRECTED, corrected)
    adjudication = {
        "manifest": "MF2-CR5 queue50 human challenge adjudication",
        "revision": "cr5-queue50-human-challenge-adjudication/1",
        "status": "COMPLETE_Q36_CONTROLLER_REQUIRED",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "decisions": decisions,
        "counts": {
            "human_challenges": 3,
            "original_reject_confirmed": 2,
            "corrected_geometry_controller_required": 1,
        },
        "corrected_geometry": {
            "path": str(CORRECTED.relative_to(ROOT)),
            "sha256": sha256_file(CORRECTED),
        },
        "original_artifacts_modified": False,
        "training_authorized": False,
    }
    atomic_json(OUT, adjudication)
    print(json.dumps({
        "status": adjudication["status"],
        "decisions": {row["event_id"]: row["decision"]
                      for row in decisions},
        "corrected_geometry": str(CORRECTED.relative_to(ROOT)),
        "corrected_geometry_sha256": sha256_file(CORRECTED),
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
