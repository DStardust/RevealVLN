#!/usr/bin/env python3
"""Project frozen 63-degree waypoint outputs onto fixed CR5 3-D branches.

The MLLM is not involved here.  Candidate endpoints are matched using
navmesh geodesic distance to directed 3-D branch samples.  Thresholds reuse
the accepted automatic-frontend contract (1.0 m tube, 0.25 m cross-branch
margin, positive progress, K=3) and are never adjusted per event.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla")
HABSIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / "third_party/habitat-sim"))
if str(HABSIM) not in sys.path:
    sys.path.insert(0, str(HABSIM))
import habitat_sim  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_preflight/multiview_branch"
GEOMETRY = BASE / "CR5_DIRECTED_GEOMETRY_PREFLIGHT.json"
INPUTS = BASE / "CR5_MULTIVIEW_PREFLIGHT_INPUTS.json"
CONTROLLER = BASE / "CR5_CONTROLLER_EXECUTION_PREFLIGHT.json"
REVIEW = ROOT / (
    "artifacts/phase0/phase0c_cr5_human_review_v1/reviews/daiyang.jsonl"
)
SHARDS = ROOT / "artifacts/phase0/phase0c_cr5_causal_gate/frontend_shards"
OUT = ROOT / (
    "artifacts/phase0/phase0c_cr5_causal_gate/"
    "CR5_CAUSAL_CANDIDATE_ANALYSIS.json"
)
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"

EXPECTED_REVIEW_SHA256 = (
    "88eb9934cb8bc0abad3400f295e0bd1527b5d08d11189d0d4f055f61df14f1cb"
)
EXPECTED_ACCEPTED_COUNT = 9
EXPECTED_REJECTED_COUNT = 1
ACCEPTED_REVIEW_LABELS = {"ACCEPT"}
REJECTED_REVIEW_LABELS = {"REJECT"}
REVIEW_EVENT_FILTER = None
REVIEW_REQUIRED = True
OUTPUT_REVISION = "cr5-causal-candidate-analysis/1"
OUTPUT_SCOPE = "9 human-accepted RxR-train CR5 pilot events"
USE_ALL_BRANCHES = False
TARGET_TUBE_M = 1.0
CROSS_BRANCH_MARGIN_M = 0.25
MIN_BRANCH_PROGRESS_M = 0.05
K = 3


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def qfloat(value, places: int = 6):
    return round(float(value), places)


def runs(values):
    output, start = [], None
    for index, value in enumerate(list(values) + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            output.append([start, index - 1])
            start = None
    return output


def geodesic(pathfinder, start, end) -> float:
    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(start, dtype="float32")
    query.requested_end = np.asarray(end, dtype="float32")
    if not pathfinder.find_path(query):
        return math.inf
    value = float(query.geodesic_distance)
    return value if math.isfinite(value) else math.inf


def branch_score(pathfinder, endpoint, samples):
    scored = []
    for sample in samples:
        offset = float(sample["offset_m"])
        if offset + 1e-9 < MIN_BRANCH_PROGRESS_M:
            continue
        scored.append((geodesic(pathfinder, endpoint,
                                sample["position_q"]), offset))
    return min(scored, key=lambda row: (row[0], -row[1])) \
        if scored else (math.inf, 0.0)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def main() -> int:
    required_paths = [GEOMETRY, INPUTS, CONTROLLER]
    if REVIEW_REQUIRED:
        required_paths.append(REVIEW)
    for path in required_paths:
        if not path.is_file() or path.is_symlink() \
                or ROOT.resolve() not in path.resolve().parents:
            raise SystemExit("unsafe or missing causal input: " + str(path))
    geometry = {row["event_id"]: row
                for row in load_json(GEOMETRY)["events"]}
    inputs = {row["event_id"]: row
              for row in load_json(INPUTS)["events"]}
    controller = {row["event_id"]: row
                  for row in load_json(CONTROLLER)["events"]}
    if REVIEW_REQUIRED:
        if sha256_file(REVIEW) != EXPECTED_REVIEW_SHA256:
            raise SystemExit("human review drift")
        reviews = load_jsonl(REVIEW)
        if REVIEW_EVENT_FILTER is not None:
            reviews = [row for row in reviews
                       if row["event_id"] in REVIEW_EVENT_FILTER]
        accepted = {row["event_id"] for row in reviews
                    if row["final_label"] in ACCEPTED_REVIEW_LABELS}
        rejected = {row["event_id"] for row in reviews
                    if row["final_label"] in REJECTED_REVIEW_LABELS}
        if (len(accepted) != EXPECTED_ACCEPTED_COUNT
                or len(rejected) != EXPECTED_REJECTED_COUNT):
            raise SystemExit("unexpected adjudicated review split")
    else:
        accepted = {
            event_id for event_id, event in geometry.items()
            if event["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
            and event_id in controller
            and controller[event_id]["status"] ==
            "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
            and (SHARDS / ("ep" + event["episode_id"] + ".json")).is_file()
        }
        rejected = set()
    if not accepted <= geometry.keys() or not accepted <= inputs.keys() \
            or not accepted <= controller.keys():
        raise SystemExit("accepted event missing upstream evidence")

    shard_cache = {}
    pathfinders = {}
    results = []
    for event_id in sorted(accepted):
        event = geometry[event_id]
        if event["status"] != "GEOMETRY_PASS_CONTROLLER_REQUIRED" \
                or controller[event_id]["status"] != \
                "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED":
            raise SystemExit("upstream gate not passed: " + event_id)
        episode_id = event["episode_id"]
        if episode_id not in shard_cache:
            path = SHARDS / ("ep" + episode_id + ".json")
            if not path.is_file() or path.is_symlink():
                raise SystemExit("missing frontend shard: " + episode_id)
            shard_cache[episode_id] = load_json(path)
        shard = shard_cache[episode_id]
        if shard["episode_id"] != episode_id \
                or shard["network_attempts"] != 0 \
                or shard["model_contract"]["sensor_hfov_deg"] != 63 \
                or shard["model_contract"]["causal_acquired_slots"] != [0]:
            raise SystemExit("frontend contract mismatch: " + episode_id)

        scene = event["scene_id"]
        if scene not in pathfinders:
            navmesh = MP3D / scene / (scene + ".navmesh")
            finder = habitat_sim.PathFinder()
            if not finder.load_nav_mesh(str(navmesh)):
                raise RuntimeError("navmesh load failed: " + scene)
            pathfinders[scene] = finder
        finder = pathfinders[scene]
        target_id = event["target"]["branch_id"]
        if USE_ALL_BRANCHES:
            alternative_ids = [row["branch_id"]
                               for row in event["alternatives"]]
            branches = {target_id: event["target"]["path_samples"]}
            branches.update({row["branch_id"]: row["path_samples"]
                             for row in event["alternatives"]})
        else:
            alternative_ids = [event["alternative"]["branch_id"]]
            branches = {
                target_id: event["target"]["path_samples"],
                alternative_ids[0]: event["alternative"]["path_samples"],
            }
        d_prefix = int(inputs[event_id]["positions"]["D"]["trace_prefix"])
        if d_prefix >= len(shard["prefix_records"]):
            raise RuntimeError("D prefix outside frontend trace")

        prefix_records = []
        current = {branch_id: [] for branch_id in branches}
        for row in shard["prefix_records"][:d_prefix + 1]:
            branch_candidates = {branch_id: [] for branch_id in branches}
            candidate_records = []
            for candidate in row["candidates"]:
                scores = {branch_id: branch_score(
                    finder, candidate["endpoint_q"], samples)
                    for branch_id, samples in branches.items()}
                ordered = sorted(
                    (distance, -progress, branch_id, progress)
                    for branch_id, (distance, progress) in scores.items())
                best = ordered[0]
                second = ordered[1]
                margin = second[0] - best[0]
                matched = (
                    best[0] <= TARGET_TUBE_M
                    and best[3] >= MIN_BRANCH_PROGRESS_M
                    and margin >= CROSS_BRANCH_MARGIN_M
                )
                assignment = best[2] if matched else None
                if assignment is not None:
                    branch_candidates[assignment].append(
                        candidate["candidate_local_id"])
                candidate_records.append({
                    "candidate_local_id": candidate["candidate_local_id"],
                    "relative_angle_signed_deg": candidate[
                        "relative_angle_signed_deg"],
                    "distance_m": candidate["distance_m"],
                    "endpoint_q": candidate["endpoint_q"],
                    "branch_assignment": assignment,
                    "best_branch_geodesic_m": qfloat(best[0])
                        if math.isfinite(best[0]) else None,
                    "best_branch_progress_m": qfloat(best[3]),
                    "cross_branch_margin_m": qfloat(margin)
                        if math.isfinite(margin) else None,
                })
            for branch_id in current:
                current[branch_id].append(bool(branch_candidates[branch_id]))
            prefix_records.append({
                "prefix_index": row["prefix_index"],
                "action": row["action"],
                "candidate_count": row["candidate_count"],
                "branch_current": {
                    branch_id: bool(values)
                    for branch_id, values in branch_candidates.items()
                },
                "branch_candidate_local_ids": branch_candidates,
                "candidates": candidate_records,
            })

        branch_runs = {branch_id: runs(values)
                       for branch_id, values in current.items()}
        established_at = {}
        for branch_id, spans in branch_runs.items():
            qualifying = [start + K - 1 for start, end in spans
                          if end - start + 1 >= K]
            established_at[branch_id] = min(qualifying) \
                if qualifying else None

        ready = []
        for index, record in enumerate(prefix_records):
            target_current = record["branch_current"][target_id]
            competition_history = {
                branch_id: (
                    established_at[branch_id] is not None
                    and established_at[branch_id] <= index
                ) for branch_id in alternative_ids
            }
            availability = [
                record["branch_current"][branch_id]
                or competition_history[branch_id]
                for branch_id in alternative_ids
            ]
            competition_available = (
                all(availability) if USE_ALL_BRANCHES else any(availability)
            )
            ready.append(target_current and competition_available)
            if USE_ALL_BRANCHES:
                record["competition_in_causal_history"] = competition_history
                record["available_competition_branch_ids"] = [
                    branch_id for branch_id in alternative_ids
                    if record["branch_current"][branch_id]
                    or competition_history[branch_id]
                ]
            else:
                record["alternative_in_causal_history"] = (
                    competition_history[alternative_ids[0]])
            record["geometric_ready"] = ready[-1]
        ready_runs = runs(ready)
        stable_ready_runs = [span for span in ready_runs
                             if span[1] - span[0] + 1 >= K]
        status = "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED" \
            if stable_ready_runs else "TOPOLOGY_ONLY_FRONTEND_K3_FAIL"
        result = {
            "event_id": event_id,
            "episode_id": episode_id,
            "scene_id": scene,
            "status": status,
            "Q_prefix": event["trace"]["Q_prefix"],
            "D_prefix": d_prefix,
            "target_branch_id": target_id,
            "branch_current_runs": branch_runs,
            "branch_established_at_confirmation_prefix": established_at,
            "stable_geometric_ready_runs": stable_ready_runs,
            "prefix_records": prefix_records,
            "language_closure_verified": False,
            "training_label": False,
        }
        if USE_ALL_BRANCHES:
            result.update({
                "candidate_branch_ids": list(branches),
                "competition_branch_ids": alternative_ids,
                "complete_verified_candidate_set_retained": True,
            })
        else:
            result["alternative_branch_id"] = alternative_ids[0]
        results.append(result)

    counts = Counter(row["status"] for row in results)
    sources = {
        "geometry": {"path": str(GEOMETRY.relative_to(ROOT)),
                     "sha256": sha256_file(GEOMETRY)},
        "multiview_inputs": {"path": str(INPUTS.relative_to(ROOT)),
                             "sha256": sha256_file(INPUTS)},
        "controller": {"path": str(CONTROLLER.relative_to(ROOT)),
                       "sha256": sha256_file(CONTROLLER)},
        "frontend_shards": [{
            "path": str((SHARDS / ("ep" + episode + ".json")).relative_to(ROOT)),
            "sha256": sha256_file(SHARDS / ("ep" + episode + ".json")),
        } for episode in sorted(shard_cache)],
    }
    if REVIEW_REQUIRED:
        sources["human_review"] = {
            "path": str(REVIEW.relative_to(ROOT)),
            "sha256": sha256_file(REVIEW),
        }
    output = {
        "revision": OUTPUT_REVISION,
        "status": "COMPLETE_LANGUAGE_GATE_REQUIRED",
        "scope": OUTPUT_SCOPE,
        "sources": sources,
        "matching_contract": {
            "distance": "Habitat PathFinder navmesh geodesic in 3-D",
            "target_tube_m": TARGET_TUBE_M,
            "cross_branch_margin_m": CROSS_BRANCH_MARGIN_M,
            "min_branch_progress_m": MIN_BRANCH_PROGRESS_M,
            "k": K,
            "search_end": "D prefix inclusive; no post-departure rescue",
            "threshold_origin": "reused accepted automatic-frontend contract; not per-event tuned",
            "alternative_availability": "current candidate or a prior K3-established branch in candidate history",
            "full_set_availability": (
                "all competing branches must be current or prior K3-established"
                if USE_ALL_BRANCHES else "not applicable"
            ),
        },
        "event_count": len(results),
        "status_counts": dict(sorted(counts.items())),
        "events": results,
        "network_calls_made": 0,
        "future_frames_used": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "status_counts": output["status_counts"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
