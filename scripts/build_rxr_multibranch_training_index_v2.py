#!/usr/bin/env python3
"""Assemble deterministic full-set labels before feature extraction."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
GEOMETRY = V2 / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json"
CONTROLLER = V2 / "RXR_MULTIBRANCH_CONTROLLER_EXECUTION_V2.json"
CAUSAL = V2 / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json"
LANGUAGE = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_GATE_V2.json"
LEGACY_AUDIT = BASE / (
    "human_pilot_300/RXR_HUMAN_PILOT_300_SELECTION.json"
)
LEGACY_MANIFEST = BASE / (
    "human_pilot_300/RXR_HUMAN_PILOT_300_MANIFEST.json"
)
LEGACY_ACCEPTANCE = BASE / (
    "human_pilot_300/RXR_HUMAN_PILOT_300_LABEL_ACCEPTANCE.json"
)
LEGACY_LABELS = BASE / "human_pilot_300/daiyang_rxr300.jsonl"
OUT = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
SPLIT_SEED = "mf2-cr6-scene-split/1"
TRAIN_SCENES = 36
DEVELOPMENT_SCENES = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    if not path.is_file() or path.is_symlink() \
            or ROOT not in path.resolve().parents:
        raise RuntimeError("unsafe or missing input: " + str(path))
    return json.loads(path.read_text())


def unique(rows):
    output = {}
    for row in rows:
        event_id = row["event_id"]
        if event_id in output:
            raise RuntimeError("duplicate event_id: " + event_id)
        output[event_id] = row
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
    legacy_doc = load(LEGACY_AUDIT)
    legacy_manifest_doc = load(LEGACY_MANIFEST)
    legacy_acceptance = load(LEGACY_ACCEPTANCE)
    if legacy_acceptance.get("status") != "HUMAN_LABELS_PASS_TX_JOIN_REQUIRED":
        raise RuntimeError("legacy pairwise human-label gate failed")
    labels = {}
    with LEGACY_LABELS.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["event_id"] in labels:
                raise RuntimeError("duplicate legacy human label")
            labels[row["event_id"]] = row
    legacy_manifest = {row["event_id"]: row
                       for row in legacy_manifest_doc["items"]}
    if (
        geometry_doc.get("status") != "COMPLETE_CONTROLLER_GATE_REQUIRED"
        or controller_doc.get("status") != "COMPLETE_CAUSAL_AND_HUMAN_GATES_REQUIRED"
        or causal_doc.get("status") != "COMPLETE_LANGUAGE_GATE_REQUIRED"
        or language_doc.get("status") != "COMPLETE_CAUSAL_CONTROLS_REQUIRED"
        or language_doc.get("full_candidate_sets") is not True
        or language_doc.get("future_frames_used") != 0
    ):
        raise RuntimeError("full-set upstream gate status mismatch")
    geometry = unique(geometry_doc["events"])
    controller = unique(controller_doc["events"])
    causal = unique(causal_doc["events"])
    language = unique(language_doc["events"])
    legacy_ids = {row["event_id"] for row in legacy_doc["items"]}

    scene_ids = sorted({row["scene_id"] for row in geometry.values()
                        if row["geometry_verified"]})
    if len(scene_ids) < TRAIN_SCENES + DEVELOPMENT_SCENES + 1:
        raise RuntimeError("insufficient scenes for the frozen split")
    ranked_scenes = sorted(scene_ids, key=lambda scene: hashlib.sha256(
        (SPLIT_SEED + ":" + scene).encode()
    ).hexdigest())
    split_by_scene = {}
    for index, scene in enumerate(ranked_scenes):
        if index < TRAIN_SCENES:
            split = "train"
        elif index < TRAIN_SCENES + DEVELOPMENT_SCENES:
            split = "development"
        else:
            split = "gold"
        split_by_scene[scene] = split

    records = []
    excluded_legacy_rejects = []
    for event_id, language_row in sorted(language.items()):
        if language_row["status"] != "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED":
            continue
        geometry_row = geometry[event_id]
        controller_row = controller[event_id]
        causal_row = causal[event_id]
        branch_ids = causal_row["candidate_branch_ids"]
        if not (
            geometry_row["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
            and controller_row["status"] == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
            and causal_row["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
            and geometry_row["candidate_branch_ids"] == branch_ids
            and controller_row["candidate_branch_ids"] == branch_ids
            and controller_row["all_candidate_branches_executed"] is True
            and causal_row["complete_verified_candidate_set_retained"] is True
        ):
            raise RuntimeError("event candidate-set closure failed: " + event_id)
        controller_branches = {row["branch_id"]: row
                               for row in controller_row["branches"]}
        if set(controller_branches) != set(branch_ids) or not all(
            row["pass"] and row["deterministic_exact"]
            for row in controller_branches.values()
        ):
            raise RuntimeError("branch controller closure failed: " + event_id)
        target = causal_row["target_branch_id"]
        reveal_start, reveal_end = language_row["reveal_interval"]
        if (
            target not in branch_ids
            or language_row["confirmation_prefix"] != reveal_end
            or reveal_end - reveal_start + 1 != 3
        ):
            raise RuntimeError("reveal interval closure failed: " + event_id)
        legacy_label = labels.get(event_id)
        legacy_item = legacy_manifest.get(event_id)
        pairwise_applicable = bool(
            legacy_label is not None and legacy_item is not None
            and legacy_item["target_branch_id"] == target
            and legacy_item["alternative_branch_id"] in branch_ids
        )
        if (pairwise_applicable
                and legacy_label["final_label"] == "REJECT"):
            excluded_legacy_rejects.append(event_id)
            continue
        pairwise_review = None
        if legacy_label is not None:
            pairwise_review = {
                "reviewer_id": legacy_label["reviewer_id"],
                "final_label": legacy_label["final_label"],
                "two_distinct_executable_exits": legacy_label[
                    "two_distinct_executable_exits"
                ],
                "alternative_is_not_incoming_closed_or_duplicate":
                    legacy_label[
                        "alternative_is_not_incoming_closed_or_duplicate"
                    ],
                "instruction_uniquely_selects_target": legacy_label[
                    "instruction_uniquely_selects_target"
                ],
                "decision_center_and_temporal_order_are_reasonable":
                    legacy_label[
                        "decision_center_and_temporal_order_are_reasonable"
                    ],
                "causal_prefix_supports_reveal_without_future_frames":
                    legacy_label[
                        "causal_prefix_supports_reveal_without_future_frames"
                    ],
                "reason_codes": legacy_label["reason_codes"],
                "applies_to_current_selected_pair": pairwise_applicable,
                "scope": (
                    "selected pair validity only; does not certify full-set "
                    "branch completeness"
                ),
            }
        records.append({
            "event_id": event_id,
            "episode_id": causal_row["episode_id"],
            "scene_id": causal_row["scene_id"],
            "split": split_by_scene[causal_row["scene_id"]],
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
                ] for branch_id in branch_ids
            },
            "legacy_pairwise_audited": event_id in legacy_ids,
            "legacy_pairwise_review": pairwise_review,
            "label_contract": {
                "candidate_mask": (
                    "persistent after K3 branch establishment; no future branch injection"
                ),
                "target_index": (
                    "masked before strict_reveal_interval start; target thereafter"
                ),
                "target_in_set": "derived from causal persistent candidate set",
                "separation": "true only against every available competitor",
                "evidence_complete": "true from strict reveal start",
                "reveal_hazard": "one only at strict reveal start",
                "option_cost": "pending per-branch resource-conditioned T_X v2",
                "checkpoint_value": "pending per-branch resource-conditioned T_X v2",
            },
            "feature_status": "PENDING_FROZEN_ETP_PREFIX_EXTRACTION",
            "resource_label_status": "PENDING_MULTIBRANCH_TX_V2",
            "human_audit_status": "PENDING_FRESH_FULLSET_AUDIT",
            "training_label": False,
        })

    counts = Counter(row["split"] for row in records)
    candidate_counts = Counter(row["candidate_branch_count"] for row in records)
    scenes_by_split = {split: sorted(
        scene for scene, value in split_by_scene.items() if value == split
    ) for split in ("train", "development", "gold")}
    disjoint = not any(
        set(scenes_by_split[left]) & set(scenes_by_split[right])
        for left, right in (
            ("train", "development"), ("train", "gold"),
            ("development", "gold"),
        )
    )
    gates = {
        "at_least_one_fullset_event": bool(records),
        "all_events_have_two_to_four_branches": all(
            2 <= row["candidate_branch_count"] <= 4 for row in records
        ),
        "all_candidate_ids_unique": all(
            len(row["candidate_branch_ids"])
            == len(set(row["candidate_branch_ids"])) for row in records
        ),
        "scene_splits_disjoint": disjoint,
        "no_future_frames_used_online": True,
        "legacy_pairwise_audit_not_promoted_to_v2_gold": all(
            not (row["split"] == "gold" and row["legacy_pairwise_audited"]
                 and row["human_audit_status"] != "PENDING_FRESH_FULLSET_AUDIT")
            for row in records
        ),
    }
    output = {
        "schema_version": "revealnav-mf2-multibranch-training-index/2",
        "status": "FEATURE_AND_TX_GENERATION_REQUIRED" if all(gates.values())
                  else "INDEX_FAIL",
        "sources": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            GEOMETRY, CONTROLLER, CAUSAL, LANGUAGE, LEGACY_AUDIT,
            LEGACY_MANIFEST, LEGACY_ACCEPTANCE, LEGACY_LABELS,
        )},
        "scene_split": {
            "seed": SPLIT_SEED,
            "rank_allocation": {
                "train": TRAIN_SCENES,
                "development": DEVELOPMENT_SCENES,
                "gold": len(scene_ids) - TRAIN_SCENES - DEVELOPMENT_SCENES,
            },
            "scenes": scenes_by_split,
        },
        "counts": {
            "events": len(records),
            "events_by_split": dict(sorted(counts.items())),
            "events_by_candidate_count": {
                str(key): value for key, value in sorted(candidate_counts.items())
            },
            "legacy_pairwise_audited_events": sum(
                row["legacy_pairwise_audited"] for row in records
            ),
            "legacy_pairwise_accepts_retained": sum(
                row["legacy_pairwise_review"] is not None
                and row["legacy_pairwise_review"]["final_label"] == "ACCEPT"
                for row in records
            ),
            "legacy_pairwise_rejects_excluded_when_pair_still_applies": len(
                excluded_legacy_rejects
            ),
        },
        "excluded_legacy_pairwise_reject_event_ids": excluded_legacy_rejects,
        "records": records,
        "gates": gates,
        "feature_generation_authorized": all(gates.values()),
        "resource_label_generation_authorized": all(gates.values()),
        "training_authorized": False,
        "training_blockers": [
            "frozen ETP causal-prefix features not generated",
            "per-branch resource-conditioned T_X v2 not generated",
            "fresh independent full-set human audit not passed",
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
