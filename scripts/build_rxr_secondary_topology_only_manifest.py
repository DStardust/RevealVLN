#!/usr/bin/env python3
"""Preserve causal-language failures as neutral exploration topology nodes."""

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
INPUTS = BASE / "multiview_factory/RXR_SECONDARY_MULTIVIEW_INPUTS.json"
PRESCREEN = BASE / "branch_factory/RXR_SECONDARY_MACHINE_PRESCREEN.json"
GEOMETRY = BASE / "multibranch/RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json"
CONTROLLER = BASE / "multibranch/RXR_SECONDARY_MULTIBRANCH_CONTROLLER.json"
ANALYSIS = BASE / "multibranch/RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json"
OUT = BASE / "multibranch/RXR_SECONDARY_TOPOLOGY_ONLY_MANIFEST.json"


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
        raise RuntimeError("unsafe or missing topology source: " + str(path))
    return json.loads(path.read_text())


def unique(rows):
    result = {}
    for row in rows:
        if row["event_id"] in result:
            raise RuntimeError("duplicate topology event")
        result[row["event_id"]] = row
    return result


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def geometry_branches(event):
    return [event["target"], *event["alternatives"]]


def branch_endpoint(branch):
    return branch.get("T_star_at_1_75m") or branch["T_i_at_1_75m"]


def main() -> int:
    inputs_doc = load(INPUTS)
    prescreen_doc = load(PRESCREEN)
    geometry_doc = load(GEOMETRY)
    controller_doc = load(CONTROLLER)
    analysis_doc = load(ANALYSIS)
    inputs = unique(inputs_doc["events"])
    prescreen = unique(prescreen_doc["events"])
    geometry = unique(geometry_doc["events"])
    controller = unique(controller_doc["events"])
    topology_only = [
        row for row in analysis_doc["events"]
        if row["status"] == "TOPOLOGY_ONLY_FRONTEND_K3_FAIL"
    ]
    records = []
    for analysis_row in sorted(topology_only, key=lambda row: row["event_id"]):
        event_id = analysis_row["event_id"]
        input_row = inputs[event_id]
        geometry_row = geometry[event_id]
        controller_row = controller[event_id]
        prescreen_row = prescreen[event_id]
        proposal_path = ROOT / prescreen_row["proposal_path"]
        if sha256_file(proposal_path) != prescreen_row["proposal_sha256"]:
            raise RuntimeError("topology proposal drift: " + event_id)
        proposal = load(proposal_path)["normalized_proposal"]
        proposal_branches = {
            row["branch_id"]: row for row in proposal["branches"]
        }
        controller_branches = {
            row["branch_id"]: row for row in controller_row["branches"]
        }
        physical_branches = geometry_branches(geometry_row)
        branch_ids = analysis_row["candidate_branch_ids"]
        if not (
            geometry_row["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
            and controller_row["status"]
            == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
            and controller_row["all_candidate_branches_executed"] is True
            and geometry_row["candidate_branch_ids"] == branch_ids
            and controller_row["candidate_branch_ids"] == branch_ids
            and {row["branch_id"] for row in physical_branches}
            == set(branch_ids)
            and set(proposal_branches) >= set(branch_ids)
            and set(controller_branches) == set(branch_ids)
            and all(
                row["pass"] and row["deterministic_exact"]
                for row in controller_branches.values()
            )
        ):
            raise RuntimeError("topology branch closure failed: " + event_id)
        q_frame = input_row["positions"]["Q"]["frame_id"]
        branches = []
        q_observed = []
        for physical in physical_branches:
            branch_id = physical["branch_id"]
            proposed = proposal_branches[branch_id]
            q_views = sorted(
                view for view in proposed["supporting_view_ids"]
                if view.startswith("Q_V")
            )
            current_q_support = bool(q_views) and set(
                proposed["supporting_frame_ids"]
            ) == {q_frame}
            if current_q_support:
                q_observed.append(branch_id)
            branches.append({
                "branch_id": branch_id,
                "q_panorama_observed": current_q_support,
                "q_supporting_view_ids": q_views if current_q_support else [],
                "endpoint_q": branch_endpoint(physical),
                "controller_action_count_at_q": controller_branches[
                    branch_id
                ]["replays"][0]["action_count"],
                "controller_exactly_reproduced": True,
            })
        complete = set(q_observed) == set(branch_ids)
        topology_ready = len(q_observed) >= 2
        if complete:
            disposition = "PANORAMIC_ONLINE_COMPLETE_TOPOLOGY"
        elif topology_ready:
            disposition = "PANORAMIC_ONLINE_PARTIAL_TOPOLOGY"
        else:
            disposition = "HINDSIGHT_ONLY_AT_Q"
        records.append({
            "event_id": event_id,
            "episode_id": analysis_row["episode_id"],
            "scene_id": analysis_row["scene_id"],
            "split": "train",
            "Q_prefix": analysis_row["Q_prefix"],
            "D_prefix": analysis_row["D_prefix"],
            "candidate_branch_ids": branch_ids,
            "candidate_branch_count": len(branch_ids),
            "q_observed_branch_ids": q_observed,
            "q_candidate_set_complete": complete,
            "topology_checkpoint_label": topology_ready,
            "semantic_evidence_state": "U",
            "semantic_target_label": None,
            "disposition": disposition,
            "branches": branches,
            "q_panorama": input_row["positions"]["Q"]["contact_sheet"],
            "online_observation_contract": (
                "only the current Q panorama may supervise branch visibility"
            ),
            "future_supported_branches_are_masked_until_observed": True,
            "training_label": False,
        })
    counts = Counter(row["disposition"] for row in records)
    output = {
        "schema_version": "revealnav-mf2-topology-only-exploration/1",
        "status": "READY_FOR_TOPOLOGY_AUXILIARY_FEATURES",
        "scope": "train-only topology nodes excluded from semantic Reveal labels",
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (INPUTS, PRESCREEN, GEOMETRY, CONTROLLER, ANALYSIS)
        },
        "label_contract": {
            "topology_checkpoint": (
                "positive only when at least two verified executable branches "
                "are supported by current-time Q panorama views"
            ),
            "semantic_state": "U for every record",
            "semantic_target": "omitted; never inherited from offline instruction",
            "partial_topology": (
                "create node with the currently observed candidate mask and "
                "append later branches only after online discovery"
            ),
        },
        "counts": {
            "topology_only_events": len(records),
            "topology_checkpoint_positive": sum(
                row["topology_checkpoint_label"] for row in records
            ),
            "by_disposition": dict(sorted(counts.items())),
            "three_or_four_branch": sum(
                row["candidate_branch_count"] >= 3 for row in records
            ),
        },
        "records": records,
        "semantic_reveal_training_authorized": False,
        "topology_auxiliary_feature_generation_authorized": True,
        "topology_auxiliary_training_authorized": False,
        "remaining_work": (
            "extract current-Q panoramic features and add matched non-junction negatives"
        ),
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "counts": output["counts"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
