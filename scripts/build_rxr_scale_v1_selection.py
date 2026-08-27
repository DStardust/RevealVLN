#!/usr/bin/env python3
"""Freeze the unconsumed R3 event-scale and new-Gold candidate population."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
HINDSIGHT = BASE / "hindsight_factory/RXR_HINDSIGHT_EVENT_CANDIDATES.json"
PRIMARY_INDEX = BASE / "multibranch_v2/RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
SECONDARY_SELECTION = (
    BASE / "secondary_expansion_v1/RXR_SECONDARY_EXPANSION_SELECTION.json"
)
CURRENT_GATE = BASE / "expiry_r3/RXR_EXPIRY_R3_FEATURE_GATE.json"
OUT_DIR = BASE / "scale_v1"
OUT = OUT_DIR / "RXR_SCALE_V1_SELECTION.json"
GOLD_SEED = "revealnav-new-gold-wave1/1"
GOLD_WAVE1_SIZE = 900


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
    primary = json.loads(PRIMARY_INDEX.read_text())
    secondary = json.loads(SECONDARY_SELECTION.read_text())
    current = json.loads(CURRENT_GATE.read_text())
    if not (
        hindsight.get("status") == "PASS_PENDING_MULTIVIEW_AND_3D_GATES"
        and primary.get("status") == "FEATURE_AND_TX_GENERATION_REQUIRED"
        and secondary.get("status") == "FROZEN_TRAIN_ONLY_SECONDARY_SELECTION"
        and current.get("status") == "EXPIRY_R3_FEATURE_GATE_PASS"
        and current.get("counts", {}).get("events") == 492
    ):
        raise RuntimeError("scale selection precondition failed")

    scene_split = {
        scene: split
        for split, scenes in primary["scene_split"]["scenes"].items()
        for scene in scenes
    }
    consumed = {row["event_id"] for row in primary["records"]}
    attempted_secondary = {row["event_id"] for row in secondary["items"]}
    if consumed & attempted_secondary:
        raise RuntimeError("primary and secondary populations overlap")
    excluded = consumed | attempted_secondary

    eligible = [
        row
        for row in hindsight["candidates"]
        if row["candidate_kind"] != "LIKELY_NO_CHOICE_HARD_NEGATIVE"
        and not row["conflicting_kind_votes"]
        and row["hindsight_candidate_id"] not in excluded
    ]
    automatic_source = [
        row for row in eligible if scene_split.get(row["scene_id"])
        in {"train", "development"}
    ]
    gold_source = [
        row for row in eligible if scene_split.get(row["scene_id"]) == "gold"
    ]
    unassigned = [row for row in eligible if row["scene_id"] not in scene_split]

    automatic_source.sort(
        key=lambda row: (
            scene_split[row["scene_id"]],
            row["scene_id"],
            row["expansion_order"],
            row["hindsight_candidate_id"],
        )
    )
    gold_source.sort(
        key=lambda row: hashlib.sha256(
            f"{GOLD_SEED}|{row['hindsight_candidate_id']}".encode()
        ).hexdigest()
    )

    def record(row: dict, order: int, lane: str) -> dict:
        split = scene_split[row["scene_id"]]
        value = {
            "scale_order": order,
            "event_id": row["hindsight_candidate_id"],
            "expansion_order": row["expansion_order"],
            "episode_id": row["episode_id"],
            "trajectory_id": row["trajectory_id"],
            "scene_id": row["scene_id"],
            "instruction_id": row["instruction_id"],
            "scene_split": split,
            "lane": lane,
            "candidate_kind": row["candidate_kind"],
            "candidate_interval": row["interval"],
            "source": row["source"],
            "processing_status": "PENDING_SCALE_MULTIVIEW",
        }
        if lane == "new_gold":
            value["gold_wave"] = 1 if order < GOLD_WAVE1_SIZE else 2
            value["gold_rank_commitment"] = hashlib.sha256(
                f"{GOLD_SEED}|{value['event_id']}".encode()
            ).hexdigest()
        return value

    automatic = [
        record(row, order, "automatic")
        for order, row in enumerate(automatic_source)
    ]
    new_gold = [
        record(row, order, "new_gold") for order, row in enumerate(gold_source)
    ]
    all_ids = [row["event_id"] for row in automatic + new_gold]
    gates = {
        "source_populations_disjoint": len(all_ids) == len(set(all_ids)),
        "all_prior_primary_events_excluded": not (
            set(all_ids) & consumed
        ),
        "all_prior_secondary_attempts_excluded": not (
            set(all_ids) & attempted_secondary
        ),
        "automatic_uses_train_or_development_only": all(
            row["scene_split"] in {"train", "development"}
            for row in automatic
        ),
        "new_gold_uses_frozen_gold_scenes_only": all(
            row["scene_split"] == "gold" for row in new_gold
        ),
        "new_gold_wave1_has_900_candidates": sum(
            row["gold_wave"] == 1 for row in new_gold
        ) == GOLD_WAVE1_SIZE,
        "old_gold_payload_not_read": True,
    }
    counts = {
        "current_train_development_events": 492,
        "automatic_candidates": len(automatic),
        "automatic_by_split": dict(sorted(Counter(
            row["scene_split"] for row in automatic
        ).items())),
        "new_gold_candidates": len(new_gold),
        "new_gold_wave1": sum(row["gold_wave"] == 1 for row in new_gold),
        "new_gold_reserve": sum(row["gold_wave"] == 2 for row in new_gold),
        "unassigned_scene_candidates_excluded": len(unassigned),
        "previous_primary_events_excluded": len(consumed),
        "previous_secondary_attempts_excluded": len(attempted_secondary),
    }
    output = {
        "schema_version": "revealnav-rxr-scale-selection/1",
        "status": "SCALE_V1_SELECTION_FROZEN" if all(gates.values()) else "FAIL",
        "scope": (
            "unconsumed hindsight candidates; automatic train/development "
            "expansion and event-disjoint new three-reviewer Gold construction"
        ),
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (HINDSIGHT, PRIMARY_INDEX, SECONDARY_SELECTION, CURRENT_GATE)
        },
        "selection_rule": {
            "eligible_hindsight_candidates_only": True,
            "prior_primary_and_every_secondary_attempt_excluded": True,
            "automatic_split": ["train", "development"],
            "new_gold_split": ["gold"],
            "new_gold_event_ids_disjoint_from_prior_events": True,
            "new_gold_scene_disjoint_from_train_and_development": True,
            "old_gold_labels_or_statistics_used": False,
            "gold_wave1_rank": "sha256(seed|event_id)",
            "gold_wave1_seed": GOLD_SEED,
            "gold_wave1_size": GOLD_WAVE1_SIZE,
            "gold_reserve_activated_only_for_rejected_wave1_items": True,
        },
        "counts": counts,
        "automatic": automatic,
        "new_gold": new_gold,
        "gates": gates,
        "human_labels_created": 0,
        "old_gold_payload_read": False,
        "training_authorized": False,
        "paper_result": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "counts": counts,
        "gates": gates,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

