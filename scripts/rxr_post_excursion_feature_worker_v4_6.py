#!/usr/bin/env python3
"""Build causal reached-branch features and BACKTRACK costs for train events."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts", ROOT / "third_party/ETP-R1"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cr5_queue50_tx_worker as core  # noqa: E402
import rxr_branch_excursion_label_worker_v4 as v4  # noqa: E402
from automatic_semantic_candidate_worker import (  # noqa: E402
    build_models, build_sim, make_observations, set_state,
)
from revealnav_cr1.causal_frontend import (  # noqa: E402
    apply_raw_view_mask, causal_vp_feature_variable, filter_waypoint_outputs,
)


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V4 = BASE / "branch_excursion_v4"
MANIFEST = V4 / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V4.json"
PROTOCOL = BASE / (
    "post_excursion_v4_6_1/RXR_POST_EXCURSION_PROTOCOL_V4_6_1.json"
)
WRONG_COMMITMENT_COST = 5.0
MISSED_OPPORTUNITY_COST = 5.0
FAILURE_COST = 5.0


def heading(rotation) -> float:
    from habitat_sim.utils.common import quat_rotate_vector

    forward = quat_rotate_vector(rotation, np.asarray([0.0, 0.0, -1.0]))
    return math.atan2(-float(forward[0]), -float(forward[2])) % (2 * math.pi)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(part, path)


def safe_json(path: Path) -> dict:
    resolved = path.resolve()
    if ROOT not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("unsafe or missing project JSON")
    return json.loads(resolved.read_text())


def compact_route(route: dict) -> dict:
    return {key: route.get(key) for key in (
        "status", "success", "error_type", "action_count",
        "action_sequence_sha256", "leg_action_counts", "controller_call_count",
        "collision_count", "path_length_m", "position_trace_sha256",
        "start_position_q", "final_position_q", "goal_q", "final_distance_m",
        "replay_sha256",
    ) if key in route}


def bounded(route: dict, denominator: int) -> float:
    if not route.get("success") or route.get("action_count") is None:
        return FAILURE_COST
    return min(float(route["action_count"]) / denominator, FAILURE_COST)


def encode_reached_state(policy, predictor, feature_sim, position, yaw, device):
    set_state(feature_sim, position, yaw)
    observations = make_observations(feature_sim, device)
    acquired = torch.zeros((1, 12), dtype=torch.bool, device=device)
    acquired[:, 0] = True
    observations = apply_raw_view_mask(observations, acquired)
    with torch.no_grad():
        raw = policy.net(
            mode="waypoint", waypoint_predictor=predictor,
            observations=observations, in_train=False,
        )
        filtered = filter_waypoint_outputs(raw, acquired)
        inputs = causal_vp_feature_variable(filtered, device)
        inputs["mode"] = "panorama"
        panorama, panorama_mask = policy.net(**inputs)
    history = (
        panorama[0] * panorama_mask[0].unsqueeze(-1)
    ).sum(0) / panorama_mask[0].sum().clamp_min(1)
    count = len(filtered["cand_angles"][0])
    if count:
        candidates = panorama[0, :count].mean(0)
    else:
        candidates = torch.zeros_like(history)
    return (
        history.detach().cpu().float().numpy(),
        candidates.detach().cpu().float().numpy(),
        count,
    )


def one_event(event_id: str, output_dir: Path, policy, predictor, device) -> dict:
    manifest = safe_json(MANIFEST)
    rows = [row for row in manifest["records"] if row["event_id"] == event_id]
    if len(rows) != 1:
        raise RuntimeError("event identity closure failure")
    record = rows[0]
    label_path = (V4 / record["path"]).resolve()
    label = safe_json(label_path)
    if label["label_source"] != record["label_source"]:
        raise RuntimeError("label source drift")
    paths = v4.bundle(record["label_source"])
    geometry = v4.witness.unique(safe_json(paths["geometry"])["events"], event_id)
    causal = v4.witness.unique(safe_json(paths["causal"])["events"], event_id)
    trace = safe_json(paths["shards"] / f"ep{causal['episode_id']}.json")
    state = trace["prefix_records"][label["decision_prefix"]]
    branch_ids = label["candidate_branch_ids"]
    target_id = label["target_branch_id"]
    checkpoint = geometry["trace"]["Q"]
    target_goal = v4.primary.branch_goal(geometry, target_id)
    denominator = int(label["normalization_denominator_actions"])
    feature_path = (ROOT / label["online_feature"]["path"]).resolve()
    if (
        ROOT not in feature_path.parents or feature_path.is_symlink()
        or not feature_path.is_file()
        or core.sha256_file(feature_path) != label["online_feature"]["sha256"]
    ):
        raise RuntimeError("causal prefix feature provenance drift")
    with np.load(feature_path, allow_pickle=False) as shard:
        instruction = shard["instruction_embedding"].astype(np.float32)
        offset = int(label["online_feature_relative_step"])
        pre_history = shard["history_embeddings"][:offset + 1].astype(np.float32)
        selected = shard["candidate_embeddings"][offset].astype(np.float32)
        mask = shard["candidate_mask"][offset].astype(np.bool_)
    if selected.shape[0] != len(branch_ids) or not mask.all():
        raise RuntimeError("decision branch embedding alignment failure")

    route_sim = core.make_sim(causal["scene_id"], 0)
    post_history = np.zeros((len(branch_ids), pre_history.shape[-1]), np.float32)
    post_candidates = np.zeros_like(post_history)
    candidate_counts = np.zeros(len(branch_ids), np.int64)
    elapsed = np.zeros(len(branch_ids), np.float32)
    reachable = np.zeros(len(branch_ids), np.bool_)
    evidence = []
    reached_states: list[tuple[np.ndarray, float] | None] = [
        None for _ in branch_ids
    ]
    try:
        for index, branch_id in enumerate(branch_ids):
            goal = v4.primary.branch_goal(geometry, branch_id)
            outbound = core.route(
                route_sim, "frozen_shortest_path_compat", state["position_q"],
                float(state["heading_rad"]), [goal],
            )
            source = sorted(label["labels"], key=lambda row: row["branch_index"])[index]
            matches_source = (
                outbound["action_sequence_sha256"]
                == source["commit_route"].get("action_sequence_sha256")
                and outbound["replay_sha256"]
                == source["commit_route"].get("replay_sha256")
                and bool(outbound["success"]) == bool(source["commit_route"].get("success"))
            )
            if not matches_source:
                raise RuntimeError("outbound frozen-controller replay drift")
            row = {
                "branch_id": branch_id, "branch_index": index,
                "is_target": branch_id == target_id,
                "outbound_route": compact_route(outbound),
                "outbound_matches_v4": True,
                "trainable": False,
            }
            if outbound["success"]:
                reached = route_sim.get_agent(0).get_state()
                reached_position = np.asarray(reached.position, np.float32).copy()
                reached_heading = heading(reached.rotation)
                reached_states[index] = (reached_position, reached_heading)
                reachable[index] = True
                elapsed[index] = float(outbound["action_count"]) / denominator
                return_route = core.route(
                    route_sim, "frozen_shortest_path_compat", reached_position,
                    reached_heading, [checkpoint, target_goal],
                )
                continue_cost = (
                    0.0 if branch_id == target_id else WRONG_COMMITMENT_COST
                )
                backtrack_cost = bounded(return_route, denominator) + (
                    MISSED_OPPORTUNITY_COST if branch_id == target_id else 0.0
                )
                row.update({
                    "trainable": True,
                    "continue_cost": core.qfloat(continue_cost),
                    "backtrack_cost": core.qfloat(backtrack_cost),
                    "preferred_action": (
                        "CONTINUE" if continue_cost < backtrack_cost
                        else "BACKTRACK" if backtrack_cost < continue_cost
                        else "TIE"
                    ),
                    "return_route": compact_route(return_route),
                    "post_input_cutoff": "before_return_rollout",
                })
            else:
                row.update({
                    "failure_disposition": "RETAINED_UNREACHABLE_NOT_TRAINABLE",
                    "continue_cost": None, "backtrack_cost": None,
                    "preferred_action": None, "return_route": None,
                })
            evidence.append(row)
    finally:
        route_sim.close()

    # Habitat-Sim 0.1.7 does not reliably support two live EGL simulators in
    # one process. Close the route simulator before opening the frozen-feature
    # renderer; the reached pose is ordinary online executor state.
    feature_sim = build_sim(causal["scene_id"])
    try:
        for index, reached_state in enumerate(reached_states):
            if reached_state is None:
                continue
            post_history[index], post_candidates[index], candidate_counts[index] = (
                encode_reached_state(
                    policy, predictor, feature_sim, reached_state[0],
                    reached_state[1], device,
                )
            )
    finally:
        feature_sim.close()

    feature_output = output_dir / f"{event_id}.npz"
    atomic_npz(feature_output, {
        "instruction_embedding": instruction,
        "pre_history_embeddings": pre_history,
        "checkpoint_embedding": pre_history[0],
        "selected_branch_embeddings": selected,
        "post_history_embeddings": post_history,
        "post_candidate_embeddings": post_candidates,
        "post_candidate_counts": candidate_counts,
        "normalized_excursion_elapsed": elapsed,
        "reachable_mask": reachable,
    })
    label_output = output_dir / f"{event_id}.json"
    value = {
        "schema_version": "revealnav-mf2-post-excursion-event/4.6.1",
        "status": "POST_EXCURSION_EVENT_COMPLETE",
        "event_id": event_id, "scene_id": causal["scene_id"],
        "label_source": record["label_source"],
        "candidate_branch_ids": branch_ids,
        "target_branch_id": target_id,
        "checkpoint_prefix": label["checkpoint_prefix"],
        "decision_prefix": label["decision_prefix"],
        "normalization_denominator_actions": denominator,
        "branches": evidence,
        "feature": {
            "path": str(feature_output.relative_to(ROOT)),
            "bytes": feature_output.stat().st_size,
            "sha256": core.sha256_file(feature_output),
        },
        "causal_input_excludes_target_truth": True,
        "future_frames_used_for_input": 0,
        "return_rollout_used_for_input": False,
        "offline_truth_used_for_labels_only": True,
        "protocol_sha256": core.sha256_file(PROTOCOL),
        "gold_payload_read": False, "paper_result": False,
    }
    value["event_sha256"] = core.stable_sha(value)
    core.atomic_json(label_output, value)
    return {
        "event_id": event_id,
        "feature_path": str(feature_output.relative_to(ROOT)),
        "feature_bytes": feature_output.stat().st_size,
        "feature_sha256": core.sha256_file(feature_output),
        "label_path": str(label_output.relative_to(ROOT)),
        "label_bytes": label_output.stat().st_size,
        "label_sha256": core.sha256_file(label_output),
        "branches": len(branch_ids),
        "trainable": int(reachable.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lane-result", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    lane_result = args.lane_result.resolve()
    if ROOT not in output_dir.parents or ROOT not in lane_result.parents:
        raise SystemExit("output outside project")
    if core.sha256_file(PROTOCOL) != os.environ.get("POST_EXCURSION_PROTOCOL_SHA256"):
        raise RuntimeError("post-excursion protocol drift")
    event_ids = json.loads(args.event_list.read_text())
    attempts = core.install_network_guard()
    os.chdir(ROOT / "third_party/ETP-R1")
    torch.manual_seed(20260827)
    torch.cuda.manual_seed_all(20260827)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    policy, predictor, _ = build_models()
    started = time.monotonic()
    records = []
    for event_id in event_ids:
        records.append(one_event(
            event_id, output_dir, policy, predictor, device,
        ))
        print(event_id, "POST_EXCURSION_PASS", flush=True)
    if attempts:
        raise RuntimeError("network attempt observed")
    value = {
        "schema_version": "revealnav-mf2-post-excursion-lane/4.6.1",
        "status": "POST_EXCURSION_LANE_COMPLETE",
        "physical_gpu": args.physical_gpu,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "records": records, "network_attempts": 0,
        "future_frames_used": 0, "raw_images_written": 0,
        "wall_clock_s": core.qfloat(time.monotonic() - started),
    }
    core.atomic_json(lane_result, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
