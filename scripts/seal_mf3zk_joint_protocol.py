#!/usr/bin/env python3
"""Seal the train-only, cross-benchmark MF3ZK protocol before collection.

The protocol deliberately separates a deterministic R2R-train confirmation
cohort from the routes used to collect new exact MF3V switches.  The public
R2R val_seen split is not reused: earlier work evaluated all 778 episodes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R2R_TRAIN = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz"
)
RXR_CORE = ROOT / (
    "artifacts/phase1/mf3zd_direct_switch_returns_v1/"
    "MF3ZD_DIRECT_SWITCH_MANIFEST.json"
)
RXR_EXPANSION = ROOT / (
    "artifacts/phase1/mf3zf_expanded_direct_switch_returns_v1/"
    "MF3ZF_DIRECT_SWITCH_MANIFEST.json"
)
OUT = ROOT / "artifacts/training/mf3zk_joint_v1"
PROTOCOL = OUT / "MF3ZK_JOINT_PROTOCOL.json"
SCHEMA = "revealnav-mf3zk-joint-train-protocol/1"
SALT = "revealnav-mf3zk-r2r-train-scene-role/1"
COLLECTION_LIMIT = 1200
CONFIRM_PER_SCENE = 4
EXPECTED_R2R_EPISODES = 10819
EXPECTED_R2R_SCENES = 61


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


def scene_id(value: str) -> str:
    scene = Path(value).stem
    if len(scene) != 11:
        raise RuntimeError(f"invalid MP3D scene id: {value}")
    return scene


def load_r2r_rows() -> list[dict]:
    if (
        R2R_TRAIN.is_symlink() or not R2R_TRAIN.is_file()
        or ROOT not in R2R_TRAIN.resolve().parents
    ):
        raise RuntimeError("R2R train payload is outside the project")
    with gzip.open(R2R_TRAIN, "rt", encoding="utf-8") as stream:
        rows = json.load(stream).get("episodes")
    if not isinstance(rows, list) or len(rows) != EXPECTED_R2R_EPISODES:
        raise RuntimeError("R2R train inventory drift")
    ids = [str(row["episode_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("R2R train episode IDs are not unique")
    return rows


def route_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory_id"])].append(row)
    routes = []
    for trajectory, members in grouped.items():
        selected = min(members, key=lambda row: stable_hash({
            "trajectory": trajectory,
            "episode": str(row["episode_id"]),
        }))
        reference = selected.get("reference_path")
        # Four waypoints still provide a non-trivial causal prefix while
        # retaining the one small R2R train scene whose routes have length 4.
        if not isinstance(reference, list) or len(reference) < 4:
            continue
        routes.append({
            "episode_id": str(selected["episode_id"]),
            "trajectory_id": trajectory,
            "scene_id": scene_id(str(selected["scene_id"])),
            "reference_points": len(reference),
            "selection_digest": stable_hash({
                "salt": "mf3zk-route/1",
                "episode": str(selected["episode_id"]),
                "trajectory": trajectory,
            }),
        })
    return sorted(routes, key=lambda row: stable_hash(row))


def role_scenes(scenes: list[str]) -> tuple[set[str], set[str]]:
    ordered = sorted(scenes, key=lambda value: stable_hash({
        "salt": SALT, "scene": value,
    }))
    confirmation_count = max(1, math.ceil(len(ordered) * 0.20))
    confirm = set(ordered[:confirmation_count])
    return set(ordered) - confirm, confirm


def balanced_take(routes: list[dict], scenes: set[str], limit: int) -> list[dict]:
    by_scene: dict[str, deque[dict]] = defaultdict(deque)
    for row in routes:
        if row["scene_id"] in scenes:
            by_scene[row["scene_id"]].append(row)
    for members in by_scene.values():
        members = sorted(members, key=lambda row: stable_hash({
            "salt": "mf3zk-collection/1", "route": row,
        }))
    queue = deque(sorted(by_scene, key=lambda value: stable_hash({
        "salt": "mf3zk-scene-queue/1", "scene": value,
    })))
    selected = []
    while queue and len(selected) < limit:
        scene = queue.popleft()
        # Re-sort here because deque assignment above intentionally avoids
        # mutating the caller's route list.
        members = sorted(by_scene[scene], key=lambda row: stable_hash({
            "salt": "mf3zk-collection/1", "route": row,
        }))
        row = members.pop(0)
        selected.append(row)
        by_scene[scene] = deque(members)
        if members:
            queue.append(scene)
    if len(selected) != min(limit, sum(len(v) for v in by_scene.values()) + len(selected)):
        raise RuntimeError("deterministic R2R collection cohort is too small")
    return selected


def confirmation_routes(routes: list[dict], scenes: set[str]) -> list[dict]:
    result = []
    for scene in sorted(scenes):
        members = sorted(
            (row for row in routes if row["scene_id"] == scene),
            key=lambda row: stable_hash({
                "salt": "mf3zk-confirm/1", "route": row,
            }),
        )
        result.extend(members[:CONFIRM_PER_SCENE])
    return result


def load_rxr_manifest(path: Path, tier: str) -> list[dict]:
    value = json.loads(path.read_text())
    if value.get("status") != "DIRECT_SWITCH_RETURN_DATASET_READY":
        raise RuntimeError(f"RxR source status drift: {path}")
    if value.get("unseen_or_test_read") is not False:
        raise RuntimeError(f"RxR source crossed public split: {path}")
    rows = value.get("records")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"empty RxR source: {path}")
    result = []
    for row in rows:
        if not all(key in row for key in (
            "scene_id", "episode_id", "feature", "delta", "decision",
        )):
            raise RuntimeError(f"RxR source schema drift: {path}")
        result.append({
            "dataset": "RxR",
            "tier": tier,
            "scene_id": str(row["scene_id"]),
            "episode_id": str(row["episode_id"]),
            "source_row_index": int(row["row_index"]),
            "source_manifest": str(path.relative_to(ROOT)),
        })
    return result


def build() -> dict:
    rows = load_r2r_rows()
    routes = route_rows(rows)
    r2r_scenes = {scene_id(str(row["scene_id"])) for row in rows}
    route_scenes = {row["scene_id"] for row in routes}
    if len(r2r_scenes) != EXPECTED_R2R_SCENES or route_scenes != r2r_scenes:
        raise RuntimeError("R2R train route scene inventory drift")
    fit_scenes, r2r_confirm_scenes = role_scenes(sorted(r2r_scenes))
    collection = balanced_take(routes, fit_scenes, COLLECTION_LIMIT)
    confirm = confirmation_routes(routes, r2r_confirm_scenes)
    if set(row["scene_id"] for row in collection) & r2r_confirm_scenes:
        raise RuntimeError("R2R collection/confirmation scene overlap")
    if set(row["trajectory_id"] for row in collection) & {
        row["trajectory_id"] for row in confirm
    }:
        raise RuntimeError("R2R collection/confirmation trajectory overlap")

    rxr_core = load_rxr_manifest(RXR_CORE, "core")
    rxr_expansion = load_rxr_manifest(RXR_EXPANSION, "expansion")
    rxr_rows = rxr_core + rxr_expansion
    rxr_scenes = {row["scene_id"] for row in rxr_rows}
    # Strict cross-benchmark holdout: a scene selected for R2R confirmation
    # is excluded from the joint fitting rows even if it appears in RxR.
    rxr_confirm = [
        row for row in rxr_rows if row["scene_id"] in r2r_confirm_scenes
        or int(stable_hash({"salt": "mf3zk-rxr-confirm/1", "scene": row["scene_id"]}), 16) % 5 == 0
    ]
    confirm_scenes = r2r_confirm_scenes | {row["scene_id"] for row in rxr_confirm}
    rxr_fit = [row for row in rxr_rows if row["scene_id"] not in confirm_scenes]
    if not rxr_fit or not rxr_confirm:
        raise RuntimeError("RxR joint fit/confirmation partition is empty")
    for tier in ("core", "expansion"):
        if not any(row["tier"] == tier for row in rxr_fit):
            raise RuntimeError(f"RxR fit has no {tier} rows")

    value = {
        "schema_version": SCHEMA,
        "status": "SEALED_BEFORE_MF3ZK_JOINT_TRAINING",
        "revision": "mf3zk",
        "mainline": "one shared action-aligned return/harm gate trained on balanced RxR+R2R train data",
        "proposal_ranker": "MF3V frozen; no proposal/backbone fine-tuning in this revision",
        "joint_sampling": {
            "effective_dataset_weight": {"RxR": 0.5, "R2R": 0.5},
            "benchmark_identifier_as_model_input": False,
        },
        "r2r_train": {
            "payload": {
                "path": str(R2R_TRAIN.relative_to(ROOT)),
                "bytes": R2R_TRAIN.stat().st_size,
                "sha256": sha256_file(R2R_TRAIN),
            },
            "available_episodes": len(rows),
            "available_routes": len(routes),
            "available_scenes": len(r2r_scenes),
            "fit_scenes": sorted(fit_scenes),
            "confirmation_scenes": sorted(r2r_confirm_scenes),
            "collection_routes": collection,
            "confirmation_routes": confirm,
            "collection_route_limit": COLLECTION_LIMIT,
            "confirmation_routes_per_scene": CONFIRM_PER_SCENE,
        },
        "rxr_sources": {
            "core": {
                "path": str(RXR_CORE.relative_to(ROOT)),
                "bytes": RXR_CORE.stat().st_size,
                "sha256": sha256_file(RXR_CORE),
                "rows": len(rxr_core),
            },
            "expansion": {
                "path": str(RXR_EXPANSION.relative_to(ROOT)),
                "bytes": RXR_EXPANSION.stat().st_size,
                "sha256": sha256_file(RXR_EXPANSION),
                "rows": len(rxr_expansion),
            },
            "fit_rows": rxr_fit,
            "confirmation_rows": rxr_confirm,
        },
        "strict_scene_holdout": {
            "confirmation_scenes": sorted(confirm_scenes),
            "fit_scene_count": len({row["scene_id"] for row in rxr_fit} | fit_scenes),
            "confirmation_scene_count": len(confirm_scenes),
            "raw_scene_id_shared_across_benchmarks_is_same_group": True,
        },
        "consumed_public_evaluation": {
            "r2r_val_seen_total_episodes": 778,
            "r2r_val_seen_fresh_confirmation_available": False,
            "policy": "never reopen R2R val_seen for 3ZK tuning; use train-scene confirmation then sealed val_unseen",
        },
        "labels": {
            "exact_one_switch": True,
            "causal_online_inputs": True,
            "future_frames": 0,
            "task_metrics_used_for_training_selection": False,
            "route_geometry_used_only_for_train_targets": True,
        },
        "public_split_access": {
            "r2r_val_seen": False,
            "r2r_val_unseen": False,
            "rxr_val_seen": False,
            "rxr_val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
    }
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("MF3ZK protocol already exists with different contents")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "inspect"))
    args = parser.parse_args()
    value = build()
    if args.command == "inspect":
        print(json.dumps({
            "status": value["status"],
            "r2r_collection_routes": len(value["r2r_train"]["collection_routes"]),
            "r2r_confirmation_routes": len(value["r2r_train"]["confirmation_routes"]),
            "rxr_fit_rows": len(value["rxr_sources"]["fit_rows"]),
            "rxr_confirmation_rows": len(value["rxr_sources"]["confirmation_rows"]),
            "confirmation_scenes": value["strict_scene_holdout"]["confirmation_scene_count"],
        }, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "status": value["status"],
            "protocol": str(PROTOCOL.relative_to(ROOT)),
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
