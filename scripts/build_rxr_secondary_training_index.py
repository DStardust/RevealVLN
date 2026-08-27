#!/usr/bin/env python3
"""Assemble train-only secondary labels before feature and T_X extraction."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
MULTIBRANCH = BASE / "multibranch"
GEOMETRY = MULTIBRANCH / "RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json"
CONTROLLER = MULTIBRANCH / "RXR_SECONDARY_MULTIBRANCH_CONTROLLER.json"
CAUSAL = MULTIBRANCH / "RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json"
LANGUAGE = MULTIBRANCH / "RXR_SECONDARY_CAUSAL_PREFIX_LANGUAGE_GATE.json"
INPUTS = BASE / "multiview_factory/RXR_SECONDARY_MULTIVIEW_INPUTS.json"
SELECTION = BASE / "RXR_SECONDARY_EXPANSION_SELECTION.json"
PRIMARY_INDEX = ROOT / (
    "artifacts/phase1/rxr_train_expansion/multibranch_v2/"
    "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
)
OUT = MULTIBRANCH / "RXR_SECONDARY_TRAINING_INDEX.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    if (
        not path.is_file()
        or path.is_symlink()
        or ROOT.resolve() not in path.resolve().parents
    ):
        raise RuntimeError("unsafe or missing input: " + str(path))
    return json.loads(path.read_text())


def unique(rows):
    output = {}
    for row in rows:
        if row["event_id"] in output:
            raise RuntimeError("duplicate event_id: " + row["event_id"])
        output[row["event_id"]] = row
    return output


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def main() -> int:
    geometry_doc = load(GEOMETRY)
    controller_doc = load(CONTROLLER)
    causal_doc = load(CAUSAL)
    language_doc = load(LANGUAGE)
    inputs_doc = load(INPUTS)
    selection_doc = load(SELECTION)
    primary_index = load(PRIMARY_INDEX)
    if not (
        geometry_doc.get("status") == "COMPLETE_CONTROLLER_GATE_REQUIRED"
        and controller_doc.get("status")
        == "COMPLETE_CAUSAL_AND_HUMAN_GATES_REQUIRED"
        and causal_doc.get("status") == "COMPLETE_LANGUAGE_GATE_REQUIRED"
        and language_doc.get("status") == "COMPLETE_CAUSAL_CONTROLS_REQUIRED"
        and language_doc.get("full_candidate_sets") is True
        and language_doc.get("future_frames_used") == 0
        and selection_doc.get("status")
        == "FROZEN_TRAIN_ONLY_SECONDARY_SELECTION"
    ):
        raise RuntimeError("secondary upstream gate status mismatch")
    geometry = unique(geometry_doc["events"])
    controller = unique(controller_doc["events"])
    causal = unique(causal_doc["events"])
    language = unique(language_doc["events"])
    inputs = unique(inputs_doc["events"])
    train_scenes = set(primary_index["scene_split"]["scenes"]["train"])
    primary_ids = {row["event_id"] for row in primary_index["records"]}

    records = []
    for event_id, language_row in sorted(language.items()):
        if language_row["status"] != "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED":
            continue
        geometry_row = geometry[event_id]
        controller_row = controller[event_id]
        causal_row = causal[event_id]
        input_row = inputs[event_id]
        branch_ids = causal_row["candidate_branch_ids"]
        if not (
            geometry_row["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
            and controller_row["status"]
            == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
            and causal_row["status"]
            == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
            and geometry_row["candidate_branch_ids"] == branch_ids
            and controller_row["candidate_branch_ids"] == branch_ids
            and controller_row["all_candidate_branches_executed"] is True
            and causal_row["complete_verified_candidate_set_retained"] is True
            and causal_row["scene_id"] in train_scenes
            and input_row.get("cascade_role") == "CONDITIONAL_SECONDARY"
        ):
            raise RuntimeError("secondary candidate-set closure failed: " + event_id)
        controller_branches = {
            row["branch_id"]: row for row in controller_row["branches"]
        }
        if set(controller_branches) != set(branch_ids) or not all(
            row["pass"] and row["deterministic_exact"]
            for row in controller_branches.values()
        ):
            raise RuntimeError("secondary branch controller closure failed")
        target = causal_row["target_branch_id"]
        reveal_start, reveal_end = language_row["reveal_interval"]
        if not (
            target in branch_ids
            and language_row["confirmation_prefix"] == reveal_end
            and reveal_end - reveal_start + 1 == 3
        ):
            raise RuntimeError("secondary reveal interval closure failed")
        records.append({
            "event_id": event_id,
            "episode_id": causal_row["episode_id"],
            "scene_id": causal_row["scene_id"],
            "split": "train",
            "cascade_role": "CONDITIONAL_SECONDARY",
            "conditional_primary_event_id": input_row[
                "conditional_primary_event_id"
            ],
            "candidate_branch_ids": branch_ids,
            "candidate_branch_count": len(branch_ids),
            "target_branch_id": target,
            "target_index": branch_ids.index(target),
            "Q_prefix": causal_row["Q_prefix"],
            "D_prefix": causal_row["D_prefix"],
            "strict_reveal_interval": [reveal_start, reveal_end],
            "branch_established_at_prefix": causal_row[
                "branch_established_at_confirmation_prefix"
            ],
            "branch_rollout_cost_actions_at_Q": {
                branch_id: controller_branches[branch_id]["replays"][0][
                    "action_count"
                ]
                for branch_id in branch_ids
            },
            "label_contract": {
                "candidate_mask": "persistent after K3 branch establishment",
                "target_index": "masked before strict reveal, target thereafter",
                "separation": "true only against every available competitor",
                "evidence_complete": "true from strict reveal start",
                "reveal_hazard": "one only at strict reveal start",
                "option_cost": "pending per-branch resource-conditioned T_X",
                "checkpoint_value": "pending per-branch T_X",
            },
            "feature_status": "PENDING_FROZEN_ETP_PREFIX_EXTRACTION",
            "resource_label_status": "PENDING_MULTIBRANCH_TX",
            "human_audit_status": "NOT_PERFORMED_AUTOMATIC_TRAIN_ONLY",
            "training_label": False,
        })

    candidate_counts = Counter(row["candidate_branch_count"] for row in records)
    gates = {
        "at_least_one_secondary_event": bool(records),
        "all_scenes_in_frozen_train_split": all(
            row["scene_id"] in train_scenes for row in records
        ),
        "all_records_train_only": all(row["split"] == "train" for row in records),
        "event_ids_disjoint_from_primary_index": not (
            {row["event_id"] for row in records} & primary_ids
        ),
        "all_events_have_two_to_four_branches": all(
            2 <= row["candidate_branch_count"] <= 4 for row in records
        ),
        "no_future_frames_used_online": True,
    }
    sources = (GEOMETRY, CONTROLLER, CAUSAL, LANGUAGE, INPUTS, SELECTION, PRIMARY_INDEX)
    output = {
        "schema_version": "revealnav-mf2-secondary-training-index/1",
        "status": "FEATURE_AND_TX_GENERATION_REQUIRED"
        if all(gates.values()) else "INDEX_FAIL",
        "scope": "conditional secondary augmentation in frozen train scenes only",
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in sources
        },
        "scene_split": {
            "source_primary_index": str(PRIMARY_INDEX.relative_to(ROOT)),
            "train_scenes": sorted(train_scenes),
            "development_scenes_used": [],
            "gold_scenes_used": [],
        },
        "counts": {
            "events": len(records),
            "train": len(records),
            "development": 0,
            "gold": 0,
            "events_by_candidate_count": {
                str(key): value for key, value in sorted(candidate_counts.items())
            },
        },
        "records": records,
        "gates": gates,
        "feature_generation_authorized": all(gates.values()),
        "resource_label_generation_authorized": all(gates.values()),
        "training_authorized": False,
        "training_blockers": [
            "frozen ETP causal-prefix features not generated",
            "per-branch resource-conditioned T_X not generated",
        ],
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "counts": output["counts"],
        "gates": gates,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
