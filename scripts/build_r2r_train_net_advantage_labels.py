#!/usr/bin/env python3
"""Build route-consistent counterfactual labels from R2R train only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ETPR1 = ROOT / "third_party/ETP-R1"
HABITAT_SIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / ".remote_runtime/habitat-sim"
)).resolve()
DATASET = ETPR1 / (
    "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz"
)
MP3D = ETPR1 / "data/scene_datasets/mp3d"
for path in (ROOT, ROOT / "scripts", HABITAT_SIM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import habitat_sim  # noqa: E402


MARGIN_M = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(part, path)


def load_episodes() -> dict[str, dict]:
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["episodes"]
    return {str(row["episode_id"]): row for row in rows}


def make_pathfinder(scene: str):
    glb = (MP3D / scene / f"{scene}.glb").resolve()
    navmesh = (MP3D / scene / f"{scene}.navmesh").resolve()
    if (
        ROOT not in glb.parents or ROOT not in navmesh.parents
        or glb.is_symlink() or navmesh.is_symlink()
        or not glb.is_file() or not navmesh.is_file()
    ):
        raise RuntimeError("scene provenance closure failed")
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh)):
        raise RuntimeError("navmesh load failed")
    return pathfinder


def geodesic(pathfinder, start, goal) -> float:
    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(start, dtype=np.float32)
    query.requested_end = np.asarray(goal, dtype=np.float32)
    if not pathfinder.find_path(query):
        return math.inf
    value = float(query.geodesic_distance)
    return value if math.isfinite(value) else math.inf


def causal_distance(start, goal) -> float:
    """Distance derivable from the two positions already present online."""
    value = float(np.linalg.norm(
        np.asarray(start, dtype=np.float32) - np.asarray(goal, dtype=np.float32)
    ))
    if not math.isfinite(value):
        raise RuntimeError("non-finite online candidate distance")
    return value


def route_suffix(pathfinder, reference: list[list[float]]) -> list[float]:
    suffix = [0.0 for _ in reference]
    for index in range(len(reference) - 2, -1, -1):
        segment = geodesic(pathfinder, reference[index], reference[index + 1])
        suffix[index] = suffix[index + 1] + segment
    return suffix


def merge_cost(
    pathfinder, checkpoint, candidate, reference, progress_index, suffix,
) -> tuple[float, float, int | None]:
    outbound = geodesic(pathfinder, checkpoint, candidate)
    choices = [
        (geodesic(pathfinder, candidate, reference[index]) + suffix[index], index)
        for index in range(progress_index, len(reference))
    ]
    finite = [row for row in choices if math.isfinite(row[0])]
    if not math.isfinite(outbound) or not finite:
        return math.inf, outbound, None
    merge, index = min(finite)
    return outbound + merge, outbound, index


def scene_partition(scenes: list[str]) -> dict[str, str]:
    ordered = sorted(scenes, key=lambda value: stable_hash({"scene": value}))
    dev_count = max(1, round(len(ordered) * 0.15))
    calibration_count = max(1, round(len(ordered) * 0.15))
    dev = set(ordered[:dev_count])
    calibration = set(ordered[dev_count:dev_count + calibration_count])
    return {
        scene: (
            "dev" if scene in dev else
            "calibration" if scene in calibration else "train"
        )
        for scene in scenes
    }


def load_summaries(
    runs: Path, summary_pattern: str = "ep_*/RUN_SUMMARY.json",
) -> list[dict]:
    summaries = []
    for path in sorted(runs.glob(summary_pattern)):
        value = json.loads(path.read_text())
        if value.get("status") != "PASS" or value.get("split") != "train":
            raise RuntimeError(f"invalid feature worker summary: {path}")
        if value.get("task_metric_payload_read") is not False:
            raise RuntimeError("task metrics entered train feature collection")
        summaries.append(value)
    if not summaries:
        raise RuntimeError("no completed R2R train feature summaries")
    return summaries


def build(
    runs: Path, output_dir: Path,
    summary_pattern: str = "ep_*/RUN_SUMMARY.json",
) -> dict:
    runs = runs.resolve()
    output_dir = output_dir.resolve()
    if ROOT not in runs.parents or ROOT not in output_dir.parents:
        raise RuntimeError("input or output escapes the project")
    episodes = load_episodes()
    summaries = load_summaries(runs, summary_pattern)
    source_policies = {summary.get("source_policy") for summary in summaries}
    if source_policies not in ({None}, {"V5.6 shadow proposals"}):
        raise RuntimeError("mixed or unsupported feature source policies")
    policy_induced = source_policies == {"V5.6 shadow proposals"}
    events = [event for summary in summaries for event in summary["feature_events"]]
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_scene[event["scene_id"]].append(event)
    partition = scene_partition(sorted({summary["scene_id"] for summary in summaries}))

    arrays = defaultdict(list)
    records = []
    unreachable = []
    for scene in sorted(by_scene):
        pathfinder = make_pathfinder(scene)
        try:
            for event in by_scene[scene]:
                if policy_induced and (
                    event.get("policy_induced") is not True
                    or event.get("proposed_branch_id")
                    not in event.get("candidate_branch_ids", ())
                    or event.get("proposed_branch_id")
                    == event.get("native_branch_id")
                    or len(event.get("candidate_branch_ids", ())) != 2
                ):
                    raise RuntimeError("policy-induced proposal identity drift")
                episode = episodes[event["episode_id"]]
                reference = [list(map(float, point)) for point in episode["reference_path"]]
                if len(reference) < 2:
                    unreachable.append({"event_id": event["event_id"], "reason": "short_reference"})
                    continue
                checkpoint = event["checkpoint_position"]
                progress_rows = [
                    (geodesic(pathfinder, checkpoint, point), index)
                    for index, point in enumerate(reference)
                ]
                finite_progress = [row for row in progress_rows if math.isfinite(row[0])]
                if not finite_progress:
                    unreachable.append({"event_id": event["event_id"], "reason": "checkpoint_unreachable"})
                    continue
                _, progress_index = min(finite_progress)
                suffix = route_suffix(pathfinder, reference)
                feature_path = (ROOT / event["feature_path"]).resolve()
                if (
                    ROOT not in feature_path.parents or feature_path.is_symlink()
                    or feature_path.stat().st_size != event["feature_bytes"]
                    or sha256_file(feature_path) != event["feature_sha256"]
                ):
                    raise RuntimeError("feature provenance drift")
                with np.load(feature_path, allow_pickle=False) as feature:
                    instruction = feature["instruction_embedding"].astype(np.float32)
                    history = feature["history_embeddings"].astype(np.float32)
                    candidates = feature["candidate_embeddings"].astype(np.float32)
                    mask = feature["candidate_mask"].astype(bool)
                branches = event["candidate_branch_ids"]
                native = event["native_branch_id"]
                native_index = branches.index(native)
                native_cost, native_outbound, native_merge = merge_cost(
                    pathfinder, checkpoint, event["candidate_positions"][native],
                    reference, progress_index, suffix,
                )
                if not math.isfinite(native_cost):
                    unreachable.append({"event_id": event["event_id"], "reason": "native_unreachable"})
                    continue
                native_causal_distance = causal_distance(
                    checkpoint, event["candidate_positions"][native]
                )
                for alternative_index, alternative in enumerate(branches):
                    if alternative == native:
                        continue
                    if (
                        policy_induced
                        and alternative != event["proposed_branch_id"]
                    ):
                        continue
                    alternative_cost, alternative_outbound, alternative_merge = merge_cost(
                        pathfinder, checkpoint, event["candidate_positions"][alternative],
                        reference, progress_index, suffix,
                    )
                    if not math.isfinite(alternative_cost):
                        unreachable.append({
                            "event_id": event["event_id"], "alternative": alternative,
                            "reason": "alternative_unreachable",
                        })
                        continue
                    alternative_causal_distance = causal_distance(
                        checkpoint, event["candidate_positions"][alternative]
                    )
                    gain = native_cost - alternative_cost
                    better = gain > MARGIN_M
                    round_trip = 2.0 * alternative_outbound
                    realized_trial_net = gain if better else -round_trip
                    arrays["instruction"].append(instruction)
                    arrays["current_history"].append(history[-1])
                    arrays["temporal_history"].append(history.mean(0))
                    arrays["native"].append(candidates[-1, native_index])
                    arrays["alternative"].append(candidates[-1, alternative_index])
                    arrays["immediate_costs"].append([
                        native_causal_distance, alternative_causal_distance
                    ])
                    arrays["better"].append(float(better))
                    arrays["positive_gain"].append(max(0.0, gain))
                    arrays["signed_gain"].append(gain)
                    arrays["round_trip_cost"].append(round_trip)
                    arrays["realized_trial_net"].append(realized_trial_net)
                    records.append({
                        "row_index": len(records),
                        "event_id": event["event_id"],
                        "episode_id": event["episode_id"],
                        "trajectory_id": event["trajectory_id"],
                        "scene_id": scene,
                        "partition": partition[scene],
                        "native_branch_id": native,
                        "alternative_branch_id": alternative,
                        "reference_progress_index": progress_index,
                        "native_merge_index": native_merge,
                        "alternative_merge_index": alternative_merge,
                        "native_route_merge_cost_m": round(native_cost, 6),
                        "alternative_route_merge_cost_m": round(alternative_cost, 6),
                        "native_causal_distance_m": round(native_causal_distance, 6),
                        "alternative_causal_distance_m": round(
                            alternative_causal_distance, 6
                        ),
                        "signed_gain_m": round(gain, 6),
                        "round_trip_cost_m": round(round_trip, 6),
                        "realized_trial_net_m": round(realized_trial_net, 6),
                        "better_by_margin": better,
                        **({
                            "controller_seed": event["controller_seed"],
                            "proposal_action": event["proposal_action"],
                            "proposed_branch_id": event["proposed_branch_id"],
                            "policy_induced": True,
                        } if policy_induced else {}),
                    })
        finally:
            del pathfinder

    if not records:
        raise RuntimeError("no finite counterfactual training rows")
    array_path = output_dir / "R2R_TRAIN_NET_ADVANTAGE_DATASET.npz"
    tensor_arrays = {
        key: np.asarray(value, dtype=(np.float16 if key in {
            "instruction", "current_history", "temporal_history", "native", "alternative"
        } else np.float32))
        for key, value in arrays.items()
    }
    atomic_npz(array_path, tensor_arrays)
    value = {
        "schema_version": (
            "revealnav-r2r-train-policy-induced-net-advantage-dataset/1"
            if policy_induced else
            "revealnav-r2r-train-net-advantage-dataset/1"
        ),
        "status": (
            "R2R_TRAIN_POLICY_INDUCED_NET_ADVANTAGE_DATASET_READY"
            if policy_induced else "R2R_TRAIN_NET_ADVANTAGE_DATASET_READY"
        ),
        "source_runs": str(runs.relative_to(ROOT)),
        "completed_episodes": len(summaries),
        "completed_runs": len(summaries),
        "unique_episodes": len({summary["episode_id"] for summary in summaries}),
        "source_policy": (
            "V5.6 shadow proposals" if policy_induced else
            "all aligned ETP alternatives"
        ),
        "controller_seeds": sorted({
            summary["controller_seed"] for summary in summaries
            if "controller_seed" in summary
        }),
        "source_feature_events": len(events),
        "training_rows": len(records),
        "positive_rows": sum(row["better_by_margin"] for row in records),
        "negative_rows": sum(not row["better_by_margin"] for row in records),
        "train_rows": sum(row["partition"] == "train" for row in records),
        "calibration_rows": sum(row["partition"] == "calibration" for row in records),
        "dev_rows": sum(row["partition"] == "dev" for row in records),
        "scenes": len(partition),
        "train_scenes": sorted(scene for scene, split in partition.items() if split == "train"),
        "calibration_scenes": sorted(scene for scene, split in partition.items() if split == "calibration"),
        "dev_scenes": sorted(scene for scene, split in partition.items() if split == "dev"),
        "unreachable_rows": unreachable,
        "records": records,
        "arrays": {
            "path": str(array_path.relative_to(ROOT)),
            "bytes": array_path.stat().st_size,
            "sha256": sha256_file(array_path),
            "shapes": {key: list(value.shape) for key, value in tensor_arrays.items()},
        },
        "label_definition": {
            "objective": "shortest geodesic merge back into the remaining R2R reference path",
            "positive_margin_m": MARGIN_M,
            "wrong_trial_cost": "two times checkpoint-to-alternative geodesic",
            "model_distance_input": (
                "Euclidean checkpoint-to-candidate distance derived only from "
                "positions available in the online ETP graph"
            ),
            "offline_geodesic_used_as_model_input": False,
            "offline_reference_path_used_for_labels_only": True,
            "future_frames_used_for_online_inputs": 0,
        },
        "task_metric_payload_read": False,
        "split": "train_only_with_scene_disjoint_internal_partition",
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    value["dataset_sha256"] = stable_hash(value)
    manifest = output_dir / "R2R_TRAIN_NET_ADVANTAGE_MANIFEST.json"
    atomic_json(manifest, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--summary-pattern", default="ep_*/RUN_SUMMARY.json",
    )
    args = parser.parse_args()
    value = build(args.runs, args.output_dir, args.summary_pattern)
    print(json.dumps({key: value[key] for key in (
        "status", "completed_episodes", "source_feature_events", "training_rows",
        "positive_rows", "negative_rows", "train_rows", "calibration_rows",
        "dev_rows", "scenes",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
