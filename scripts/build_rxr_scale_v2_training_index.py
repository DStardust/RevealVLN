#!/usr/bin/env python3
"""Assemble strict automatic scale-v2 train/development event labels."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2"
AUTO = BASE / "automatic"
MULTI = AUTO / "multibranch"
GEOMETRY = MULTI / "RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
CONTROLLER = MULTI / "RXR_SCALE_MULTIBRANCH_CONTROLLER.json"
CAUSAL = MULTI / "RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json"
LANGUAGE = MULTI / "RXR_SCALE_CAUSAL_PREFIX_LANGUAGE_GATE.json"
INPUTS = AUTO / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
SELECTION = BASE / "RXR_SCALE_V2_SELECTION.json"
ROUTES = BASE / "RXR_SCALE_V2_ROUTE_CENSUS.json"
OUT = MULTI / "RXR_SCALE_TRAINING_INDEX.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in path.resolve().parents:
        raise RuntimeError("unsafe or missing scale-v2 input: " + str(path))
    return json.loads(path.read_text())


def by_id(rows: list[dict]) -> dict[str, dict]:
    output = {}
    for row in rows:
        if row["event_id"] in output:
            raise RuntimeError("duplicate event_id: " + row["event_id"])
        output[row["event_id"]] = row
    return output


def main() -> int:
    geometry_doc = load(GEOMETRY)
    controller_doc = load(CONTROLLER)
    causal_doc = load(CAUSAL)
    language_doc = load(LANGUAGE)
    inputs_doc = load(INPUTS)
    selection_doc = load(SELECTION)
    routes_doc = load(ROUTES)
    if not (
        geometry_doc.get("status") == "COMPLETE_CONTROLLER_GATE_REQUIRED"
        and controller_doc.get("status") == "COMPLETE_CAUSAL_AND_HUMAN_GATES_REQUIRED"
        and causal_doc.get("status") == "COMPLETE_LANGUAGE_GATE_REQUIRED"
        and language_doc.get("status") == "COMPLETE_CAUSAL_CONTROLS_REQUIRED"
        and language_doc.get("full_candidate_sets") is True
        and language_doc.get("future_frames_used") == 0
        and selection_doc.get("status") == "SCALE_V2_SELECTION_FROZEN"
        and routes_doc.get("status") == "SCALE_V2_ROUTE_CENSUS_FROZEN"
    ):
        raise RuntimeError("scale-v2 index upstream status mismatch")

    geometry = by_id(geometry_doc["events"])
    controller = by_id(controller_doc["events"])
    causal = by_id(causal_doc["events"])
    language = by_id(language_doc["events"])
    inputs = by_id(inputs_doc["events"])
    route_by_order = {
        row["expansion_order"]: row for row in routes_doc["candidates"]
    }
    records = []
    for event_id, language_row in sorted(language.items()):
        if language_row["status"] != "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED":
            continue
        geometry_row = geometry[event_id]
        controller_row = controller[event_id]
        causal_row = causal[event_id]
        input_row = inputs[event_id]
        route = route_by_order[input_row["expansion_order"]]
        branch_ids = causal_row["candidate_branch_ids"]
        if not (
            geometry_row["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
            and controller_row["status"] == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
            and causal_row["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
            and geometry_row["candidate_branch_ids"] == branch_ids
            and controller_row["candidate_branch_ids"] == branch_ids
            and controller_row["all_candidate_branches_executed"] is True
            and causal_row["complete_verified_candidate_set_retained"] is True
            and route["scene_split"] in {"train", "development"}
            and route["scene_id"] == causal_row["scene_id"]
            and input_row.get("scale_lane") == "automatic"
        ):
            raise RuntimeError("scale-v2 candidate-set closure failed: " + event_id)
        controller_branches = {row["branch_id"]: row for row in controller_row["branches"]}
        if set(controller_branches) != set(branch_ids) or not all(
            row["pass"] and row["deterministic_exact"]
            for row in controller_branches.values()
        ):
            raise RuntimeError("scale-v2 branch controller closure failed")
        reveal_start, reveal_end = language_row["reveal_interval"]
        target = causal_row["target_branch_id"]
        if not (
            target in branch_ids
            and language_row["confirmation_prefix"] == reveal_end
            and reveal_end - reveal_start + 1 == 3
        ):
            raise RuntimeError("scale-v2 reveal interval closure failed")
        records.append(
            {
                "event_id": event_id,
                "episode_id": causal_row["episode_id"],
                "scene_id": causal_row["scene_id"],
                "split": route["scene_split"],
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
                "human_audit_status": "NOT_PERFORMED_AUTOMATIC_SCALE",
                "training_label": False,
            }
        )

    split_counts = Counter(row["split"] for row in records)
    candidate_counts = Counter(row["candidate_branch_count"] for row in records)
    gates = {
        "at_least_one_scale_v2_event": bool(records),
        "all_records_train_or_development": all(
            row["split"] in {"train", "development"} for row in records
        ),
        "all_events_have_two_to_four_branches": all(
            2 <= row["candidate_branch_count"] <= 4 for row in records
        ),
        "all_event_ids_use_scale_v2_namespace": all(
            row["event_id"].startswith("v2x") for row in records
        ),
        "no_future_frames_used_online": True,
    }
    sources = (GEOMETRY, CONTROLLER, CAUSAL, LANGUAGE, INPUTS, SELECTION, ROUTES)
    output = {
        "schema_version": "revealnav-mf2-scale-v2-training-index/1",
        "status": "FEATURE_AND_TX_GENERATION_REQUIRED" if all(gates.values()) else "INDEX_FAIL",
        "scope": "automatic event-scale augmentation from the remaining route census",
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
        "old_gold_payload_read": False,
    }
    part = OUT.with_name(OUT.name + ".part")
    part.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(part, OUT)
    print(json.dumps({"status": output["status"], "counts": output["counts"], "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
