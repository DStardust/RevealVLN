#!/usr/bin/env python3
"""Freeze train-only conditional secondary events for data expansion."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
HINDSIGHT = BASE / "hindsight_factory/RXR_HINDSIGHT_EVENT_CANDIDATES.json"
INDEX = BASE / "multibranch_v2/RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
AUTHORIZATION = BASE / (
    "multibranch_v2/RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"
)
OUT_DIR = BASE / "secondary_expansion_v1"
OUT = OUT_DIR / "RXR_SECONDARY_EXPANSION_SELECTION.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def main() -> int:
    hindsight = json.loads(HINDSIGHT.read_text())
    index = json.loads(INDEX.read_text())
    authorization = json.loads(AUTHORIZATION.read_text())
    if not (
        hindsight.get("status") == "PASS_PENDING_MULTIVIEW_AND_3D_GATES"
        and hindsight.get("conditional_secondary_count") == 1929
        and index.get("status") == "FEATURE_AND_TX_GENERATION_REQUIRED"
        and authorization.get("status") == "TRAINING_AUTHORIZATION_PASS"
    ):
        raise RuntimeError("secondary selection precondition failed")
    split_by_scene = {
        scene: split
        for split, scenes in index["scene_split"]["scenes"].items()
        for scene in scenes
    }
    candidates = {
        row["hindsight_candidate_id"]: row for row in hindsight["candidates"]
    }
    admitted_primary_ids = {row["event_id"] for row in index["records"]}
    selected = []
    for plan in hindsight["cascade_review_plan"]:
        secondary_id = plan["secondary_candidate_id"]
        primary_id = plan["primary_candidate_id"]
        if secondary_id is None:
            continue
        candidate = candidates[secondary_id]
        if (
            split_by_scene.get(candidate["scene_id"]) != "train"
            or primary_id in admitted_primary_ids
        ):
            continue
        source_path = ROOT / candidate["source"]["proposal_path"]
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or sha256_file(source_path) != candidate["source"]["proposal_sha256"]
        ):
            raise RuntimeError("secondary source provenance failure")
        selected.append({
            "secondary_order": len(selected),
            "event_id": secondary_id,
            "conditional_primary_event_id": primary_id,
            "primary_absent_from_v2_index": True,
            "expansion_order": candidate["expansion_order"],
            "episode_id": candidate["episode_id"],
            "trajectory_id": candidate["trajectory_id"],
            "scene_id": candidate["scene_id"],
            "instruction_id": candidate["instruction_id"],
            "candidate_kind": candidate["candidate_kind"],
            "candidate_interval": candidate["interval"],
            "source": candidate["source"],
            "processing_status": "PENDING_SECONDARY_MULTIVIEW_RENDER",
        })
    event_ids = [row["event_id"] for row in selected]
    trajectory_ids = [row["trajectory_id"] for row in selected]
    if (
        len(selected) != 903
        or len(set(event_ids)) != len(selected)
        or len(set(trajectory_ids)) != len(selected)
        or any(row["scene_id"] not in index["scene_split"]["scenes"]["train"]
               for row in selected)
    ):
        raise RuntimeError("secondary selection closure failure")
    output = {
        "schema_version": "revealnav-rxr-secondary-expansion-selection/1",
        "status": "FROZEN_TRAIN_ONLY_SECONDARY_SELECTION",
        "scope": (
            "training augmentation only; conditional secondaries frozen before "
            "primary geometry/causal outcomes and activated only when the primary "
            "did not enter the v2 index"
        ),
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (HINDSIGHT, INDEX, AUTHORIZATION)
        },
        "selection_rule": {
            "scene_split": "train only",
            "secondary_candidate_was_precommitted": True,
            "conditional_primary_absent_from_v2_index": True,
            "one_secondary_per_trajectory": True,
            "development_or_gold_events_selected": 0,
            "not_an_unbiased_event_rate_sample": True,
        },
        "counts": {
            "selected_events": len(selected),
            "unique_trajectories": len(set(trajectory_ids)),
            "unique_scenes": len({row["scene_id"] for row in selected}),
            "by_candidate_kind": dict(sorted(Counter(
                row["candidate_kind"] for row in selected
            ).items())),
        },
        "items": selected,
        "gold_payload_read": False,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "counts": output["counts"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
