#!/usr/bin/env python3
"""Deterministic 3-D directed geometry gate for CR5 preflight proposals.

The MLLM supplies semantic exit hypotheses only.  This verifier independently
reconstructs the authorized RxR-train reference trace, grounds the target on
the directed future route, searches the local MP3D navmesh for executable
counterfactual routes matching the proposed alternatives, and rejects incoming
retrace, short branches, insufficient angular separation, and early remerge.

Passing this file is still not a training label: an actual discrete-controller
rollout and causal ego-FOV projection remain separate gates.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
HABSIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / "third_party/habitat-sim"))
for value in (str(SCRIPTS), str(HABSIM)):
    if value not in sys.path:
        sys.path.insert(0, value)

import habitat_sim  # noqa: E402
from phase0c_oracle_lowlevel_probe import (  # noqa: E402
    absolute_heading,
    build_lowlevel_trace,
    signed_delta,
)


BASE = ROOT / "artifacts/phase0/phase0c_cr5_preflight/multiview_branch"
INPUT = BASE / "CR5_MULTIVIEW_PREFLIGHT_INPUTS.json"
PRESCREEN = BASE / "CR5_MULTIVIEW_MAIN_AGENT_PRESCREEN_V2.json"
PROPOSALS = BASE / "proposals_v2"
RXR_TRAIN = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
OUT = BASE / "CR5_DIRECTED_GEOMETRY_PREFLIGHT.json"

EXPECTED_INPUT_SHA256 = (
    "3d3a1d4ce468c8a54a5a61b96f340a415bad8357442ae242b0cf6b595a12f7fe"
)
EXPECTED_PRESCREEN_SHA256 = (
    "e14f1c5e61e0f725ae94fd9599455a0e32f30626400964aceb9395b3ccaad5d3"
)
EXPECTED_SELECTED_COUNT = 11
OUTPUT_MANIFEST = "MF2-CR5 deterministic directed 3-D geometry preflight"
OUTPUT_REVISION = "cr5-directed-geometry-preflight/1"
OUTPUT_SCOPE = "11 canonical candidates from six blinded RxR-train trajectories"
TARGET_DIRECTION_POLICY = "hard_gate"
RETAIN_ALL_ALTERNATIVES = False

ELIGIBLE_DISPOSITIONS = {
    "CAUSAL_CANDIDATE_TO_3D", "RELOCATE_EARLIER_THEN_3D",
}
DIRECTION_OFFSET_DEG = {
    "FRONT": 0.0,
    "FRONT_LEFT": 45.0,
    "LEFT": 90.0,
    "BACK_LEFT": 135.0,
    "BACK": 180.0,
    "BACK_RIGHT": -135.0,
    "RIGHT": -90.0,
    "FRONT_RIGHT": -45.0,
}

# Frozen within this preflight implementation.  These values instantiate the
# CR5 ranges; they are not tuned per sample.
BRANCH_ENTRY_M = 1.0
BRANCH_TARGET_M = 1.75
SEARCH_MIN_PATH_M = BRANCH_TARGET_M
SEARCH_MAX_PATH_M = 4.5
MIN_TARGET_DIRECTION_ANGLE_DEG = 0.0
MAX_TARGET_DIRECTION_ERROR_DEG = 75.0
MAX_ALTERNATIVE_DIRECTION_ERROR_DEG = 85.0
MIN_BRANCH_ANGLE_DEG = 45.0
MIN_SEPARATION_AT_1M = 0.5
MIN_SEPARATION_AT_TARGET = 0.9
MIN_INCOMING_ANGLE_DEG = 40.0
MIN_INCOMING_SEPARATION_AT_1M = 0.5
LEVEL_MAX_VERTICAL_M = 0.65
STAIR_MIN_VERTICAL_M = 0.18

TOPOLOGY_CORE_RADIUS_M = 0.8
TOPOLOGY_OUTER_RADIUS_M = 3.0
TOPOLOGY_REACH_M = 2.0


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)
                                - np.asarray(b, dtype=float)))


def polyline_length(points) -> float:
    return sum(distance(a, b) for a, b in zip(points, points[1:]))


def point_at(points, offset_m: float):
    remaining = float(offset_m)
    for a, b in zip(points, points[1:]):
        length = distance(a, b)
        if length > 1e-9 and remaining <= length + 1e-9:
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            return a + (b - a) * min(1.0, remaining / length)
        remaining -= length
    return np.asarray(points[-1], dtype=float)


def vector_angle_deg(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def qpoint(value, places: int = 6):
    return [round(float(item), places) for item in value]


def qfloat(value, places: int = 6):
    return round(float(value), places)


def prefix_number(value: str) -> int:
    if not isinstance(value, str) or not value.startswith("P") \
            or not value[1:].isdigit():
        raise ValueError("bad prefix id: " + repr(value))
    return int(value[1:])


def shortest_path(pathfinder, start, end):
    request = habitat_sim.ShortestPath()
    request.requested_start = np.asarray(start, dtype="float32")
    request.requested_end = np.asarray(end, dtype="float32")
    if not pathfinder.find_path(request) or len(request.points) < 2:
        return None
    points = [np.asarray(value, dtype=float) for value in request.points]
    length = polyline_length(points)
    if not math.isfinite(length):
        return None
    return points, length


def trace_polyline(trace, center: int, direction: int,
                   required_length: float):
    result = [np.asarray(trace[center]["position"], dtype=float)]
    index = center + direction
    while 0 <= index < len(trace):
        point = np.asarray(trace[index]["position"], dtype=float)
        if distance(result[-1], point) > 1e-7:
            result.append(point)
            if polyline_length(result) + 1e-9 >= required_length:
                break
        index += direction
    return result


def vertical_compatible(motion: str, delta_y: float) -> bool:
    if motion == "LEVEL":
        return abs(delta_y) <= LEVEL_MAX_VERTICAL_M
    if motion == "UP":
        return delta_y >= STAIR_MIN_VERTICAL_M
    if motion == "DOWN":
        return delta_y <= -STAIR_MIN_VERTICAL_M
    return motion in {"MIXED", "UNCERTAIN"}


def mesh_data(pathfinder):
    raw = np.asarray(pathfinder.build_navmesh_vertices(), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) % 3:
        raise RuntimeError("unexpected triangulated navmesh shape")
    triangles = raw.reshape(-1, 3, 3)
    centroids = triangles.mean(axis=1)
    edge_owner = {}
    adjacency = [set() for _ in triangles]
    for triangle_id, triangle in enumerate(triangles):
        keys = [tuple(np.round(point, 5)) for point in triangle]
        for a, b in ((keys[0], keys[1]), (keys[1], keys[2]),
                     (keys[2], keys[0])):
            edge = tuple(sorted((a, b)))
            if edge in edge_owner:
                neighbor = edge_owner[edge]
                adjacency[triangle_id].add(neighbor)
                adjacency[neighbor].add(triangle_id)
            else:
                edge_owner[edge] = triangle_id
    return centroids, adjacency


def topology_components(centroids, adjacency, q, route_heading):
    seed = int(np.argmin(np.linalg.norm(centroids - q, axis=1)))
    graph_distance = [float("inf")] * len(centroids)
    graph_distance[seed] = distance(centroids[seed], q)
    queue = [(graph_distance[seed], seed)]
    limit = TOPOLOGY_OUTER_RADIUS_M + 0.25
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != graph_distance[node] or current_distance > limit:
            continue
        for neighbor in adjacency[node]:
            candidate = current_distance + distance(
                centroids[node], centroids[neighbor])
            if candidate < graph_distance[neighbor] and candidate <= limit:
                graph_distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))

    active = {
        index for index, value in enumerate(graph_distance)
        if TOPOLOGY_CORE_RADIUS_M <= value <= TOPOLOGY_OUTER_RADIUS_M
    }
    components = []
    while active:
        stack = [active.pop()]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = adjacency[node] & active
            active -= neighbors
            stack.extend(neighbors)
        reach = max(graph_distance[node] for node in component)
        if reach < TOPOLOGY_REACH_M or len(component) < 3:
            continue
        shell_start = max(TOPOLOGY_REACH_M, reach - 0.4)
        shell = [node for node in component
                 if graph_distance[node] >= shell_start]
        position = np.mean(centroids[shell], axis=0)
        relative = math.degrees(signed_delta(
            absolute_heading(q, position), route_heading))
        components.append({
            "triangle_count": len(component),
            "geodesic_graph_reach_m": qfloat(reach),
            "shell_position_q": qpoint(position),
            "relative_direction_deg": qfloat(relative, 3),
            "vertical_delta_m": qfloat(position[1] - q[1]),
        })
    components.sort(key=lambda value: value["relative_direction_deg"])
    return components


def alternative_endpoints(pathfinder, q, route_heading, branch,
                          mesh_centroids):
    direction = branch["horizontal_direction"]
    if direction not in DIRECTION_OFFSET_DEG:
        return []
    offset = DIRECTION_OFFSET_DEG[direction]
    motion = branch["vertical_motion"]
    pool = []
    for point in mesh_centroids:
        horizontal = math.hypot(float(point[0] - q[0]),
                                float(point[2] - q[2]))
        if 0.7 <= horizontal <= 4.2 and distance(point, q) <= 4.8:
            pool.append(np.asarray(point, dtype=float))

    if motion == "LEVEL":
        vertical_offsets = [0.0]
    elif motion == "UP":
        vertical_offsets = [0.5, 1.0, 1.5, 2.0]
    elif motion == "DOWN":
        vertical_offsets = [-0.5, -1.0, -1.5, -2.0]
    else:
        vertical_offsets = [-1.0, 0.0, 1.0]
    for delta in range(-60, 61, 10):
        heading = route_heading + math.radians(offset + delta)
        for radial in (1.75, 2.25, 2.75, 3.25):
            for delta_y in vertical_offsets:
                desired = np.asarray([
                    q[0] - radial * math.sin(heading), q[1] + delta_y,
                    q[2] - radial * math.cos(heading),
                ], dtype="float32")
                snapped = np.asarray(pathfinder.snap_point(desired),
                                     dtype=float)
                if np.isfinite(snapped).all():
                    pool.append(snapped)

    candidates = []
    seen = set()
    desired_heading = route_heading + math.radians(offset)
    for endpoint in pool:
        key = tuple(np.round(endpoint, 3))
        if key in seen:
            continue
        seen.add(key)
        horizontal = math.hypot(float(endpoint[0] - q[0]),
                                float(endpoint[2] - q[2]))
        if horizontal < 0.7:
            continue
        endpoint_error = abs(math.degrees(signed_delta(
            absolute_heading(q, endpoint), desired_heading)))
        if endpoint_error > MAX_ALTERNATIVE_DIRECTION_ERROR_DEG:
            continue
        found = shortest_path(pathfinder, q, endpoint)
        if found is None:
            continue
        points, length = found
        if not SEARCH_MIN_PATH_M <= length <= SEARCH_MAX_PATH_M:
            continue
        point_1m = point_at(points, BRANCH_ENTRY_M)
        point_target = point_at(points, BRANCH_TARGET_M)
        delta_y = float(point_target[1] - q[1])
        if not vertical_compatible(motion, delta_y):
            continue
        initial_error = abs(math.degrees(signed_delta(
            absolute_heading(q, point_1m), desired_heading)))
        score = (initial_error + 0.25 * endpoint_error
                 + 4.0 * abs(length - 2.5))
        candidates.append({
            "score": score,
            "path": points,
            "path_length_m": length,
            "point_1m": point_1m,
            "point_target": point_target,
            "initial_direction_error_deg": initial_error,
            "endpoint_direction_error_deg": endpoint_error,
            "vertical_delta_at_target_m": delta_y,
        })
    candidates.sort(key=lambda value: (
        value["score"], tuple(qpoint(value["point_target"]))))
    return candidates


def distinct_evidence(q, target_points, alternative, incoming_point):
    target_1m = point_at(target_points, BRANCH_ENTRY_M)
    target_end = point_at(target_points, BRANCH_TARGET_M)
    alternative_1m = alternative["point_1m"]
    alternative_end = alternative["point_target"]
    angle = vector_angle_deg(target_1m - q, alternative_1m - q)
    separation_1m = distance(target_1m, alternative_1m)
    separation_end = distance(target_end, alternative_end)
    intermediate = []
    for offset in (1.0, 1.25, 1.5, 1.75):
        intermediate.append({
            "offset_m": offset,
            "separation_m": qfloat(distance(
                point_at(target_points, offset),
                point_at(alternative["path"], offset))),
        })
    if incoming_point is None:
        incoming_angle = None
        incoming_separation = None
        not_incoming = True
    else:
        incoming_angle = vector_angle_deg(
            incoming_point - q, alternative_1m - q)
        incoming_separation = distance(incoming_point, alternative_1m)
        not_incoming = (
            incoming_angle >= MIN_INCOMING_ANGLE_DEG
            and incoming_separation >= MIN_INCOMING_SEPARATION_AT_1M
        )
    passed = (
        angle >= MIN_BRANCH_ANGLE_DEG
        and separation_1m >= MIN_SEPARATION_AT_1M
        and separation_end >= MIN_SEPARATION_AT_TARGET
        and not_incoming
    )
    return passed, {
        "three_dimensional_angle_at_1m_deg": qfloat(angle, 3),
        "separation_at_1m_m": qfloat(separation_1m),
        "separation_at_1_75m_m": qfloat(separation_end),
        "separation_profile": intermediate,
        "incoming_angle_at_1m_deg": None if incoming_angle is None else
            qfloat(incoming_angle, 3),
        "incoming_separation_at_1m_m": None if incoming_separation is None
            else qfloat(incoming_separation),
        "not_incoming_retrace": not_incoming,
        "no_early_remerge": separation_end >= MIN_SEPARATION_AT_TARGET,
    }


def main() -> int:
    if TARGET_DIRECTION_POLICY not in {
            "hard_gate", "official_reference_future_diagnostic"}:
        raise SystemExit("unsupported target direction policy")
    if sha256_file(INPUT) != EXPECTED_INPUT_SHA256:
        raise SystemExit("multi-view input SHA drift")
    if sha256_file(PRESCREEN) != EXPECTED_PRESCREEN_SHA256:
        raise SystemExit("prescreen SHA drift")
    if any(token in str(RXR_TRAIN) for token in
           ("val_unseen", "test_challenge", "/test/")):
        raise SystemExit("forbidden dataset path")

    manifest = json.loads(INPUT.read_text())
    prescreen = json.loads(PRESCREEN.read_text())
    events = {row["event_id"]: row for row in manifest["events"]}
    selected_rows = [
        row for row in prescreen["events"]
        if row["prescreen_disposition"] in ELIGIBLE_DISPOSITIONS
    ]
    if len(selected_rows) != EXPECTED_SELECTED_COUNT:
        raise SystemExit(
            "expected %d canonical prescreen candidates" %
            EXPECTED_SELECTED_COUNT)
    wanted_episode_ids = {events[row["event_id"]]["episode_id"]
                          for row in selected_rows}
    with gzip.open(RXR_TRAIN, "rt") as handle:
        episodes = {
            str(row["episode_id"]): row
            for row in json.load(handle)["episodes"]
            if str(row["episode_id"]) in wanted_episode_ids
        }
    if set(episodes) != wanted_episode_ids:
        raise SystemExit("RxR-train episode closure failure")

    scene_cache = {}
    results = []
    for prescreen_row in selected_rows:
        event_id = prescreen_row["event_id"]
        event = events[event_id]
        scene = event["scene_id"]
        if scene not in scene_cache:
            pathfinder = habitat_sim.PathFinder()
            navmesh = MP3D / scene / (scene + ".navmesh")
            if (not navmesh.is_file() or navmesh.is_symlink()
                    or not pathfinder.load_nav_mesh(str(navmesh))):
                raise SystemExit("navmesh closure failure: " + scene)
            centroids, adjacency = mesh_data(pathfinder)
            scene_cache[scene] = (pathfinder, centroids, adjacency,
                                  sha256_file(navmesh))
        pathfinder, centroids, adjacency, navmesh_sha = scene_cache[scene]
        trace = build_lowlevel_trace(pathfinder,
                                     episodes[event["episode_id"]])
        center = event["positions"]["Q"]["trace_prefix"]
        q = np.asarray(trace[center]["position"], dtype=float)
        if distance(q, event["positions"]["Q"]["position_q"]) > 1e-4:
            raise SystemExit("Q position reconstruction drift: " + event_id)
        route_heading = float(event["positions"]["Q"][
            "route_forward_heading_rad"])
        future = trace_polyline(trace, center, 1, BRANCH_TARGET_M)
        past = trace_polyline(trace, center, -1, BRANCH_ENTRY_M)
        future_length = polyline_length(future)
        past_length = polyline_length(past)
        incoming_point = point_at(past, BRANCH_ENTRY_M) \
            if past_length + 1e-9 >= BRANCH_ENTRY_M else None

        proposal_path = ROOT / prescreen_row["proposal_path"] \
            if prescreen_row.get("proposal_path") else \
            PROPOSALS / (event_id + ".json")
        if sha256_file(proposal_path) != prescreen_row["proposal_sha256"]:
            raise SystemExit("proposal SHA drift: " + event_id)
        proposal = json.loads(proposal_path.read_text())[
            "normalized_proposal"]
        branch_by_id = {row["branch_id"]: row
                        for row in proposal["branches"]}
        target_id = proposal["target_resolution"]["target_branch_id"]
        target_branch = branch_by_id[target_id]

        failures = []
        target_diagnostics = []
        target_record = None
        if future_length + 1e-9 < BRANCH_TARGET_M:
            failures.append("TARGET_REFERENCE_ROUTE_SHORTER_THAN_1_75M")
        else:
            target_1m = point_at(future, BRANCH_ENTRY_M)
            target_end = point_at(future, BRANCH_TARGET_M)
            target_offset = DIRECTION_OFFSET_DEG.get(
                target_branch["horizontal_direction"])
            if target_offset is None:
                target_direction_consistent = None
                target_error = None
                if TARGET_DIRECTION_POLICY == "hard_gate":
                    failures.append("TARGET_DIRECTION_UNCERTAIN")
                else:
                    target_diagnostics.append(
                        "TARGET_DIRECTION_UNCERTAIN_REGROUNDED_TO_"
                        "OFFICIAL_REFERENCE_FUTURE")
            else:
                desired = route_heading + math.radians(target_offset)
                target_error = abs(math.degrees(signed_delta(
                    absolute_heading(q, target_1m), desired)))
                target_direction_consistent = (
                    MIN_TARGET_DIRECTION_ANGLE_DEG <= target_error
                    <= MAX_TARGET_DIRECTION_ERROR_DEG)
                if not target_direction_consistent:
                    if TARGET_DIRECTION_POLICY == "hard_gate":
                        failures.append("TARGET_DIRECTION_MISMATCH")
                    else:
                        target_diagnostics.append(
                            "TARGET_DIRECTION_MISMATCH_REGROUNDED_TO_"
                            "OFFICIAL_REFERENCE_FUTURE")
            target_delta_y = float(target_end[1] - q[1])
            if not vertical_compatible(target_branch["vertical_motion"],
                                       target_delta_y):
                failures.append("TARGET_VERTICAL_MOTION_MISMATCH")
            if incoming_point is not None:
                target_incoming_angle = vector_angle_deg(
                    incoming_point - q, target_1m - q)
                target_incoming_separation = distance(
                    incoming_point, target_1m)
                if (target_incoming_angle < MIN_INCOMING_ANGLE_DEG
                        or target_incoming_separation <
                        MIN_INCOMING_SEPARATION_AT_1M):
                    failures.append("TARGET_IS_INCOMING_RETRACE")
            else:
                target_incoming_angle = None
                target_incoming_separation = None
            target_record = {
                "branch_id": target_id,
                "visual_descriptor": target_branch["visual_descriptor"],
                "horizontal_direction": target_branch[
                    "horizontal_direction"],
                "vertical_motion": target_branch["vertical_motion"],
                "Q": qpoint(q),
                "B_star_at_1m": qpoint(target_1m),
                "T_star_at_1_75m": qpoint(target_end),
                "reference_order_verified": True,
                "reference_future_length_available_m": qfloat(future_length),
                "direction_error_deg": None if target_error is None else
                    qfloat(target_error, 3),
                "vertical_delta_m": qfloat(target_delta_y),
                "incoming_angle_deg": None if target_incoming_angle is None
                    else qfloat(target_incoming_angle, 3),
                "incoming_separation_m": None if
                    target_incoming_separation is None else
                    qfloat(target_incoming_separation),
                "path_samples": [
                    {"offset_m": offset,
                     "position_q": qpoint(point_at(future, offset))}
                    for offset in (0.0, 0.5, 1.0, 1.25, 1.5, 1.75)
                ],
            }
            if TARGET_DIRECTION_POLICY != "hard_gate":
                target_record.update({
                    "grounding_authority": "official_reference_future",
                    "proposed_direction_consistent":
                        target_direction_consistent,
                    "proposed_direction_is_diagnostic": True,
                    "direction_regrounded":
                        target_direction_consistent is not True,
                })

        selected_alternative = None
        validated_alternatives = []
        alternative_diagnostics = []
        if target_record is not None and not failures:
            alternative_branches = [
                branch for branch in proposal["branches"]
                if branch["branch_id"] != target_id
                and branch["traversability_from_images"]
                == "LIKELY_TRAVERSABLE"
            ]
            if RETAIN_ALL_ALTERNATIVES:
                alternative_branches.sort(key=lambda row: row["branch_id"])
            for branch in alternative_branches:
                candidates = alternative_endpoints(
                    pathfinder, q, route_heading, branch, centroids)
                diagnostic = {
                    "branch_id": branch["branch_id"],
                    "horizontal_direction": branch[
                        "horizontal_direction"],
                    "vertical_motion": branch["vertical_motion"],
                    "executable_candidate_path_count": len(candidates),
                    "distinct_candidate_found": False,
                }
                for candidate in candidates:
                    passed, evidence = distinct_evidence(
                        q, future, candidate, incoming_point)
                    if not passed:
                        continue
                    cross_distinct = all(
                        vector_angle_deg(
                            candidate["point_1m"] - q,
                            np.asarray(existing["B_i_at_1m"], dtype=float) - q,
                        ) >= MIN_BRANCH_ANGLE_DEG
                        and distance(
                            candidate["point_1m"], existing["B_i_at_1m"]
                        ) >= MIN_SEPARATION_AT_1M
                        and distance(
                            candidate["point_target"],
                            existing["T_i_at_1_75m"],
                        ) >= MIN_SEPARATION_AT_TARGET
                        for existing in validated_alternatives
                    )
                    if RETAIN_ALL_ALTERNATIVES and not cross_distinct:
                        continue
                    diagnostic["distinct_candidate_found"] = True
                    current = {
                        "branch_id": branch["branch_id"],
                        "visual_descriptor": branch["visual_descriptor"],
                        "horizontal_direction": branch[
                            "horizontal_direction"],
                        "vertical_motion": branch["vertical_motion"],
                        "Q": qpoint(q),
                        "B_i_at_1m": qpoint(candidate["point_1m"]),
                        "T_i_at_1_75m": qpoint(candidate["point_target"]),
                        "navmesh_shortest_path_length_m": qfloat(
                            candidate["path_length_m"]),
                        "direction_error_deg": qfloat(
                            candidate["initial_direction_error_deg"], 3),
                        "vertical_delta_m": qfloat(
                            candidate["vertical_delta_at_target_m"]),
                        "distinctness": evidence,
                        "path_samples": [
                            {"offset_m": offset,
                             "position_q": qpoint(point_at(
                                 candidate["path"], offset))}
                            for offset in
                            (0.0, 0.5, 1.0, 1.25, 1.5, 1.75)
                        ],
                        "search_score": qfloat(candidate["score"], 3),
                    }
                    if (selected_alternative is None
                            or current["search_score"] <
                            selected_alternative["search_score"]):
                        selected_alternative = current
                    if RETAIN_ALL_ALTERNATIVES:
                        validated_alternatives.append(current)
                    break
                alternative_diagnostics.append(diagnostic)
        if selected_alternative is None:
            failures.append("NO_DISTINCT_EXECUTABLE_ALTERNATIVE")

        components = topology_components(
            centroids, adjacency, q, route_heading)
        status = "GEOMETRY_PASS_CONTROLLER_REQUIRED" \
            if not failures else "GEOMETRY_REJECT"
        result = {
            "event_id": event_id,
            "episode_id": event["episode_id"],
            "scene_id": scene,
            "prescreen_disposition": prescreen_row[
                "prescreen_disposition"],
            "status": status,
            "failures": failures,
            "navmesh": {
                "path": str((MP3D / scene / (scene + ".navmesh")).relative_to(
                    ROOT)),
                "sha256": navmesh_sha,
                "local_annulus_component_count": len(components),
                "components": components,
                "component_count_is_diagnostic_not_semantic_label": True,
            },
            "trace": {
                "trace_prefix_count": len(trace),
                "Q_prefix": center,
                "Q": qpoint(q),
                "agent_heading_rad": qfloat(trace[center]["heading"]),
                "route_forward_heading_rad": qfloat(route_heading),
                "incoming_1m_available": incoming_point is not None,
                "incoming_point_at_1m": None if incoming_point is None else
                    qpoint(incoming_point),
            },
            "target": target_record,
            "alternative": selected_alternative,
            "alternative_search": alternative_diagnostics,
            "geometry_verified": not failures,
            "controller_verified": False,
            "causal_prefix_verified": False,
            "human_label": None,
            "training_label": False,
        }
        if RETAIN_ALL_ALTERNATIVES:
            result.update({
                "alternatives": validated_alternatives,
                "candidate_branch_ids": ([] if target_record is None else
                    [target_record["branch_id"]]) + [
                        row["branch_id"] for row in validated_alternatives
                    ],
                "candidate_branch_count": (0 if target_record is None else 1)
                    + len(validated_alternatives),
                "all_proposed_alternatives_evaluated": True,
            })
        if TARGET_DIRECTION_POLICY != "hard_gate":
            result["target_direction_diagnostics"] = target_diagnostics
        results.append(result)

    counts = Counter(row["status"] for row in results)
    output = {
        "manifest": OUTPUT_MANIFEST,
        "revision": OUTPUT_REVISION,
        "status": "COMPLETE_CONTROLLER_GATE_REQUIRED",
        "scope": OUTPUT_SCOPE,
        "sources": {
            "multiview_input": {
                "path": str(INPUT.relative_to(ROOT)),
                "sha256": EXPECTED_INPUT_SHA256,
            },
            "main_agent_prescreen": {
                "path": str(PRESCREEN.relative_to(ROOT)),
                "sha256": EXPECTED_PRESCREEN_SHA256,
            },
            "dataset": str(RXR_TRAIN.relative_to(ROOT)),
            "dataset_split": "RxR train only",
        },
        "thresholds": {
            "branch_entry_m": BRANCH_ENTRY_M,
            "branch_target_m": BRANCH_TARGET_M,
            "min_branch_angle_deg": MIN_BRANCH_ANGLE_DEG,
            "min_separation_at_1m_m": MIN_SEPARATION_AT_1M,
            "min_separation_at_1_75m_m": MIN_SEPARATION_AT_TARGET,
            "min_incoming_angle_deg": MIN_INCOMING_ANGLE_DEG,
            "min_incoming_separation_at_1m_m":
                MIN_INCOMING_SEPARATION_AT_1M,
            "max_target_direction_error_deg":
                MAX_TARGET_DIRECTION_ERROR_DEG,
            "max_alternative_direction_error_deg":
                MAX_ALTERNATIVE_DIRECTION_ERROR_DEG,
            "topology_core_radius_m": TOPOLOGY_CORE_RADIUS_M,
            "topology_outer_radius_m": TOPOLOGY_OUTER_RADIUS_M,
            "topology_min_reach_m": TOPOLOGY_REACH_M,
        },
        "candidate_count": len(results),
        "status_counts": dict(sorted(counts.items())),
        "events": results,
        "network_calls_made": 0,
        "controller_rollouts_made": 0,
        "forbidden_split_payloads_opened": 0,
        "training_authorized": False,
    }
    if TARGET_DIRECTION_POLICY != "hard_gate":
        output["target_direction_policy"] = {
            "name": TARGET_DIRECTION_POLICY,
            "target_geometry_authority": "official_reference_future",
            "mllm_target_direction_role": "diagnostic_proposal_only",
            "alternative_direction_role": "search_proposal_with_all_"
                "existing_geometry_thresholds_unchanged",
        }
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "candidate_count": output["candidate_count"],
        "status_counts": output["status_counts"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
