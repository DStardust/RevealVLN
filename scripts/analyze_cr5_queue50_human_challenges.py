#!/usr/bin/env python3
"""Diagnose human-challenged queue50 geometry rejects without relabeling them."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
HABSIM = ROOT / "third_party/habitat-sim"
for value in (str(SCRIPTS), str(HABSIM)):
    if value not in sys.path:
        sys.path.insert(0, value)

import habitat_sim  # noqa: E402
import verify_phase0c_cr5_directed_geometry as geometry  # noqa: E402
from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
PRIMARY = BASE / "multiview_primary"
REVIEW = BASE / "human_review_fast"
INPUTS = PRIMARY / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
ACCEPTED = PRIMARY / "CR5_QUEUE50_PRIMARY_MULTIVIEW_ACCEPTED_RUN.json"
OLD_GEOMETRY = PRIMARY / "CR5_QUEUE50_DIRECTED_GEOMETRY.json"
HUMAN_ACCEPTANCE = REVIEW / "CR5_QUEUE50_AUTO_REJECT_HUMAN_ACCEPTANCE.json"
RXR_TRAIN = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
OUT = REVIEW / "CR5_QUEUE50_HUMAN_CHALLENGE_GEOMETRY_DIAGNOSTIC.json"
EXPECTED = {
    INPUTS: "6b70a70e5eb1e25f9522b30209eb56dc2efbf6457377a1aabefdeca6886aee72",
    ACCEPTED: "0f5b643612ad1a52b12aaa12d3d26b06b5dc7b288cfbc4f435f98fd3c5b81ead",
    OLD_GEOMETRY: "46609126537ddb9c4936bc93d683dd06243b91203d6bf2f7c03f30cca7deb850",
    HUMAN_ACCEPTANCE: "4fb4bc45a2fbdd65fa80922c18c18b8ceed16a910dab67191f50e58defdfebd2",
}


def load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or geometry.sha256_file(path) != expected):
            raise SystemExit("pinned challenge evidence drift: " + str(path))
    human = load(HUMAN_ACCEPTANCE)
    challenged = set(human["suspect_false_reject_event_ids"])
    if challenged != {
            "q17_ep34158_hv05", "q24_ep28644_hv04",
            "q36_ep1049_hv05"}:
        raise SystemExit("unexpected human challenge set")

    inputs = {row["event_id"]: row for row in load(INPUTS)["events"]}
    accepted = {row["event_id"]: row for row in load(ACCEPTED)["events"]}
    old = {row["event_id"]: row for row in load(OLD_GEOMETRY)["events"]}
    episode_ids = {inputs[event_id]["episode_id"] for event_id in challenged}
    with gzip.open(RXR_TRAIN, "rt") as handle:
        episodes = {str(row["episode_id"]): row
                    for row in json.load(handle)["episodes"]
                    if str(row["episode_id"]) in episode_ids}
    if set(episodes) != episode_ids:
        raise SystemExit("train episode closure failure")

    results = []
    for event_id in sorted(challenged):
        event = inputs[event_id]
        scene = event["scene_id"]
        pathfinder = habitat_sim.PathFinder()
        navmesh = MP3D / scene / (scene + ".navmesh")
        if (not navmesh.is_file() or navmesh.is_symlink()
                or not pathfinder.load_nav_mesh(str(navmesh))):
            raise SystemExit("navmesh closure failure: " + scene)
        centroids, _ = geometry.mesh_data(pathfinder)
        trace = build_lowlevel_trace(pathfinder, episodes[event["episode_id"]])
        center = event["positions"]["Q"]["trace_prefix"]
        q = np.asarray(trace[center]["position"], dtype=float)
        if geometry.distance(q, event["positions"]["Q"]["position_q"]) > 1e-4:
            raise SystemExit("Q reconstruction drift: " + event_id)
        route_heading = float(event["positions"]["Q"][
            "route_forward_heading_rad"])
        future = geometry.trace_polyline(
            trace, center, 1, geometry.BRANCH_TARGET_M)
        past = geometry.trace_polyline(
            trace, center, -1, geometry.BRANCH_ENTRY_M)
        if geometry.polyline_length(future) < geometry.BRANCH_TARGET_M:
            raise SystemExit("challenged target route is too short")
        incoming = (geometry.point_at(past, geometry.BRANCH_ENTRY_M)
                    if geometry.polyline_length(past)
                    >= geometry.BRANCH_ENTRY_M else None)

        proposal_path = ROOT / accepted[event_id]["accepted_proposal_path"]
        if geometry.sha256_file(proposal_path) != accepted[event_id][
                "accepted_proposal_sha256"]:
            raise SystemExit("proposal drift: " + event_id)
        proposal = load(proposal_path)["normalized_proposal"]
        target_id = proposal["target_resolution"]["target_branch_id"]
        diagnostics = []
        selected = None
        for branch in proposal["branches"]:
            if (branch["branch_id"] == target_id
                    or branch["traversability_from_images"]
                    != "LIKELY_TRAVERSABLE"):
                continue
            candidates = geometry.alternative_endpoints(
                pathfinder, q, route_heading, branch, centroids)
            passed_candidates = []
            for candidate in candidates:
                passed, distinct = geometry.distinct_evidence(
                    q, future, candidate, incoming)
                if passed:
                    passed_candidates.append({
                        "path_length_m": geometry.qfloat(
                            candidate["path_length_m"]),
                        "B_i_at_1m": geometry.qpoint(candidate["point_1m"]),
                        "T_i_at_1_75m": geometry.qpoint(
                            candidate["point_target"]),
                        "direction_error_deg": geometry.qfloat(
                            candidate["initial_direction_error_deg"], 3),
                        "distinctness": distinct,
                        "path_samples": [{
                            "offset_m": offset,
                            "position_q": geometry.qpoint(geometry.point_at(
                                candidate["path"], offset)),
                        } for offset in (0.0, 0.5, 1.0, 1.25, 1.5, 1.75)],
                        "score": geometry.qfloat(candidate["score"], 3),
                    })
            passed_candidates.sort(key=lambda row: row["score"])
            diagnostics.append({
                "branch_id": branch["branch_id"],
                "visual_descriptor": branch["visual_descriptor"],
                "horizontal_direction": branch["horizontal_direction"],
                "executable_candidate_count": len(candidates),
                "distinct_nonincoming_candidate_count": len(
                    passed_candidates),
                "best_candidate": (passed_candidates[0]
                                   if passed_candidates else None),
            })
            if passed_candidates and (selected is None
                                      or passed_candidates[0]["score"]
                                      < selected["best_candidate"]["score"]):
                selected = diagnostics[-1]

        results.append({
            "event_id": event_id,
            "old_failures": old[event_id]["failures"],
            "old_target_direction_error_deg": (
                (old[event_id].get("target") or {}).get(
                    "direction_error_deg")
            ),
            "reference_target_path_reconstructed": True,
            "alternative_search_run_despite_target_direction_diagnostic": True,
            "alternative_diagnostics": diagnostics,
            "selected_executable_alternative": selected,
            "diagnostic_status": (
                "EXECUTABLE_ALTERNATIVE_FOUND_SEMANTIC_REVIEW_REQUIRED"
                if selected else "NO_DISTINCT_NONINCOMING_ALTERNATIVE"
            ),
            "original_geometry_label_changed": False,
            "training_label": False,
        })

    output = {
        "manifest": "MF2-CR5 queue50 human challenge geometry diagnostic",
        "revision": "cr5-queue50-human-challenge-geometry-diagnostic/1",
        "status": "COMPLETE_NO_RELABELING",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "events": results,
        "counts": {
            "human_challenged": len(results),
            "executable_alternative_found": sum(
                row["selected_executable_alternative"] is not None
                for row in results),
            "no_distinct_nonincoming_alternative": sum(
                row["selected_executable_alternative"] is None
                for row in results),
        },
        "interpretation": (
            "This diagnostic separates target-direction-description failure "
            "from counterfactual executability. Any found alternative still "
            "requires semantic/visual adjudication before a versioned "
            "geometry replacement can be accepted."
        ),
        "network_calls_made": 0,
        "original_geometry_label_changed": False,
        "training_authorized": False,
    }
    temporary = OUT.with_name(OUT.name + ".part")
    temporary.write_text(json.dumps(
        output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({
        "status": output["status"],
        "counts": output["counts"],
        "dispositions": {row["event_id"]: row["diagnostic_status"]
                         for row in results},
        "output": str(OUT.relative_to(ROOT)),
        "sha256": geometry.sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
