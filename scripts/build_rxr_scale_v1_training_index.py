#!/usr/bin/env python3
"""Assemble automatic train/development scale-v1 causal event labels."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1"
AUTO = BASE / "automatic"
MULTI = AUTO / "multibranch"
GEOMETRY = MULTI / "RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
CONTROLLER = MULTI / "RXR_SCALE_MULTIBRANCH_CONTROLLER.json"
CAUSAL = MULTI / "RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json"
LANGUAGE = MULTI / "RXR_SCALE_CAUSAL_PREFIX_LANGUAGE_GATE.json"
INPUTS = AUTO / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
SELECTION = BASE / "RXR_SCALE_V1_SELECTION.json"
PRIMARY_INDEX = ROOT / (
    "artifacts/phase1/rxr_train_expansion/multibranch_v2/"
    "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
)
SECONDARY_SELECTION = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1/"
    "RXR_SECONDARY_EXPANSION_SELECTION.json"
)
OUT = MULTI / "RXR_SCALE_TRAINING_INDEX.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in path.resolve().parents:
        raise RuntimeError("unsafe or missing input: " + str(path))
    return json.loads(path.read_text())


def unique(rows: list[dict]) -> dict[str, dict]:
    output = {}
    for row in rows:
        event_id = row["event_id"]
        if event_id in output:
            raise RuntimeError("duplicate event_id: " + event_id)
        output[event_id] = row
    return output


def atomic_json(path: Path, value: dict) -> None:
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
    primary = load(PRIMARY_INDEX)
    secondary = load(SECONDARY_SELECTION)
    if not (
        geometry_doc.get("status") == "COMPLETE_CONTROLLER_GATE_REQUIRED"
        and controller_doc.get("status") == "COMPLETE_CAUSAL_AND_HUMAN_GATES_REQUIRED"
        and causal_doc.get("status") == "COMPLETE_LANGUAGE_GATE_REQUIRED"
        and language_doc.get("status") == "COMPLETE_CAUSAL_CONTROLS_REQUIRED"
        and language_doc.get("full_candidate_sets") is True
        and language_doc.get("future_frames_used") == 0
        and selection_doc.get("status") == "SCALE_V1_SELECTION_FROZEN"
    ):
        raise RuntimeError("scale index upstream status mismatch")

    geometry = unique(geometry_doc["events"])
    controller = unique(controller_doc["events"])
    causal = unique(causal_doc["events"])
    language = unique(language_doc["events"])
    inputs = unique(inputs_doc["events"])
    scene_split = {
        scene: split
        for split, scenes in primary["scene_split"]["scenes"].items()
        for scene in scenes
    }
    prior_ids = {row["event_id"] for row in primary["records"]}
    prior_ids.update(row["event_id"] for row in secondary["items"])

    records = []
    for event_id, language_row in sorted(language.items()):
        if language_row["status"] != "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED":
            continue
        geometry_row = geometry[event_id]
        controller_row = controller[event_id]
        causal_row = causal[event_id]
        input_row = inputs[event_id]
        branch_ids = causal_row["candidate_branch_ids"]
        split = scene_split.get(causal_row["scene_id"])
        if not (
            geometry_row["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
            and controller_row["status"] == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
            and causal_row["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
            and geometry_row["candidate_branch_ids"] == branch_ids
            and controller_row["candidate_branch_ids"] == branch_ids
            and controller_row["all_candidate_branches_executed"] is True
            and causal_row["complete_verified_candidate_set_retained"] is True
            and split in {"train", "development"}
            and input_row.get("cascade_role") == "SCALE_UNCONSUMED_CANDIDATE"
            and input_row.get("scale_lane") == "automatic"
            and event_id not in prior_ids
        ):
            raise RuntimeError("scale candidate-set closure failed: " + event_id)
        controller_branches = {
            row["branch_id"]: row for row in controller_row["branches"]
        }
        if set(controller_branches) != set(branch_ids) or not all(
            row["pass"] and row["deterministic_exact"]
            for row in controller_branches.values()
        ):
            raise RuntimeError("scale branch controller closure failed")
        target = causal_row["target_branch_id"]
        reveal_start, reveal_end = language_row["reveal_interval"]
        if not (
            target in branch_ids
            and language_row["confirmation_prefix"] == reveal_end
            and reveal_end - reveal_start + 1 == 3
        ):
            raise RuntimeError("scale reveal interval closure failed")
        records.append({
            "event_id": event_id,
            "episode_id": causal_row["episode_id"],
            "scene_id": causal_row["scene_id"],
            "split": split,
            "scale_lane": "automatic",
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
                branch_id: controller_branches[branch_id]["replays"][0]["action_count"]
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
            "human_audit_status": "NOT_PERFORMED_AUTOMATIC_SCALE",
            "training_label": False,
        })

    candidate_counts = Counter(row["candidate_branch_count"] for row in records)
    split_counts = Counter(row["split"] for row in records)
    ids = {row["event_id"] for row in records}
    gates = {
        "at_least_one_scale_event": bool(records),
        "all_scenes_in_frozen_train_or_development_split": all(
            scene_split[row["scene_id"]] == row["split"]
            and row["split"] in {"train", "development"}
            for row in records
        ),
        "no_gold_records": not split_counts.get("gold", 0),
        "event_ids_disjoint_from_all_prior_attempts": not (ids & prior_ids),
        "all_events_have_two_to_four_branches": all(
            2 <= row["candidate_branch_count"] <= 4 for row in records
        ),
        "no_future_frames_used_online": True,
    }
    sources = (
        GEOMETRY, CONTROLLER, CAUSAL, LANGUAGE, INPUTS, SELECTION,
        PRIMARY_INDEX, SECONDARY_SELECTION,
    )
    output = {
        "schema_version": "revealnav-mf2-scale-training-index/1",
        "status": "FEATURE_AND_TX_GENERATION_REQUIRED" if all(gates.values()) else "INDEX_FAIL",
        "scope": "automatic event-scale augmentation in frozen train/development scenes",
        "sources": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        "counts": {
            "events": len(records),
            "train": split_counts.get("train", 0),
            "development": split_counts.get("development", 0),
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
        "gold_payload_read": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"], "counts": output["counts"],
        "gates": gates, "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

