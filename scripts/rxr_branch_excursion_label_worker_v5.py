#!/usr/bin/env python3
"""Generate one V5 train-only branch-excursion label from frozen evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cr5_queue50_tx_worker as core  # noqa: E402
import run_rxr_branch_excursion_witness_v4 as witness  # noqa: E402
import rxr_multibranch_tx_v2_worker as branch_tx  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SOURCE_MANIFEST = BASE / "RXR_SCALE_AUTOMATIC_TRAINING_MANIFEST.json"
PROTOCOL = BASE / (
    "branch_excursion_v5_1/RXR_BRANCH_EXCURSION_LABEL_PROTOCOL_V5_1.json"
)
FAILURE_COST = 5.0
WRONG_COMMITMENT_COST = 5.0


def bundle(label_source: str) -> dict[str, Path]:
    if label_source == "primary_human_audited":
        multi = BASE / "multibranch_v2"
        return {
            "geometry": multi / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json",
            "causal": multi / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json",
            "shards": BASE / "causal_frontend/frontend_shards",
            "tx": multi / "tx_runs/round1",
        }
    if label_source == "automatic_secondary_pseudolabel":
        wave = BASE / "secondary_expansion_v1"
        multi = wave / "multibranch"
        return {
            "geometry": multi / "RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json",
            "causal": multi / "RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json",
            "shards": wave / "causal_frontend/frontend_shards",
            "tx": multi / "tx_runs/round1",
        }
    wave_names = {
        "automatic_scale_pseudolabel": "scale_v1",
        "automatic_scale_v2_pseudolabel": "scale_v2",
    }
    if label_source in wave_names:
        wave = BASE / wave_names[label_source] / "automatic"
        multi = wave / "multibranch"
        return {
            "geometry": multi / "RXR_SCALE_MULTIBRANCH_GEOMETRY.json",
            "causal": multi / "RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json",
            "shards": wave / "causal_frontend/frontend_shards",
            "tx": multi / "tx_runs/round1",
        }
    raise ValueError(f"unsupported label source: {label_source}")


def safe_json(path: Path) -> dict:
    resolved = path.resolve()
    if ROOT not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("unsafe or missing project JSON")
    return json.loads(resolved.read_text())


def bounded(route: dict, denominator: int) -> float:
    if not route.get("success") or route.get("action_count") is None:
        return FAILURE_COST
    return min(float(route["action_count"]) / denominator, FAILURE_COST)


def compact_existing_route(route: dict) -> dict:
    keys = (
        "status", "success", "error_type", "action_count",
        "action_sequence_sha256", "leg_action_counts", "controller_call_count",
        "collision_count", "path_length_m", "position_trace_sha256",
        "start_position_q", "final_position_q", "goal_q", "final_distance_m",
        "replay_sha256",
    )
    return {key: route.get(key) for key in keys if key in route}


def source_record(event_id: str) -> dict:
    rows = [
        row for row in safe_json(SOURCE_MANIFEST)["records"]
        if row["event_id"] == event_id
    ]
    if len(rows) != 1 or rows[0]["split"] != "train":
        raise RuntimeError("worker accepts exactly one authorized train event")
    return rows[0]


def validate_feature(record: dict, decision_step: int, candidates: int) -> Path:
    feature = (BASE / record["path"]).resolve()
    if (
        BASE not in feature.parents
        or feature.is_symlink()
        or not feature.is_file()
        or feature.stat().st_size != record["bytes"]
        or core.sha256_file(feature) != record["sha256"]
    ):
        raise RuntimeError("online feature provenance drift")
    with np.load(feature, allow_pickle=False) as shard:
        mask = shard["candidate_mask"]
        embeddings = shard["candidate_embeddings"]
        if (
            not 0 <= decision_step < mask.shape[0]
            or embeddings.shape[:2] != mask.shape
            or mask.shape[1] != candidates
            or not bool(mask[decision_step].all())
        ):
            raise RuntimeError("decision-time feature/branch alignment failure")
    return feature


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.is_symlink():
        raise SystemExit("unsafe output path")
    expected_protocol = os.environ.get("RXR_BRANCH_EXCURSION_V5_1_PROTOCOL_SHA256")
    if not expected_protocol or core.sha256_file(PROTOCOL) != expected_protocol:
        raise RuntimeError("branch-excursion V5 protocol drift")

    network = core.install_network_guard()
    started = time.monotonic()
    record = source_record(args.event_id)
    paths = bundle(record["label_source"])
    geometry = witness.unique(safe_json(paths["geometry"])["events"], args.event_id)
    causal = witness.unique(safe_json(paths["causal"])["events"], args.event_id)
    existing = safe_json(paths["tx"] / f"{args.event_id}.json")["evidence"]
    branch_ids = causal["candidate_branch_ids"]
    target_id = causal["target_branch_id"]
    if not (
        geometry["candidate_branch_ids"] == branch_ids
        and existing["candidate_branch_ids"] == branch_ids
        and existing["target_branch_id"] == target_id
        and record["candidate_count"] == len(branch_ids)
    ):
        raise RuntimeError("branch identity closure failure")

    q_prefix = int(existing["checkpoint"]["prefix_index"])
    decision_prefix = max(
        q_prefix,
        max(int(causal["branch_established_at_confirmation_prefix"][branch_id])
            for branch_id in branch_ids),
    )
    decision_step = decision_prefix - q_prefix
    feature = validate_feature(record, decision_step, len(branch_ids))
    trace = safe_json(
        paths["shards"] / f"ep{causal['episode_id']}.json"
    )
    state = trace["prefix_records"][decision_prefix]
    checkpoint = geometry["trace"]["Q"]
    target_controller = existing["branches"][target_id]["controllers"][
        "frozen_shortest_path_compat"
    ]
    denominator = int(target_controller["normalization_denominator_actions"])
    if denominator < 1:
        raise RuntimeError("invalid normalization denominator")

    sim = core.make_sim(causal["scene_id"], args.gpu)
    try:
        labels = []
        for branch_index, branch_id in enumerate(branch_ids):
            old_controller = existing["branches"][branch_id]["controllers"][
                "frozen_shortest_path_compat"
            ]
            prefix_rows = [
                row for row in old_controller["prefix_costs"]
                if int(row["prefix_index"]) == decision_prefix
            ]
            if len(prefix_rows) != 1:
                raise RuntimeError("existing direct-cost prefix mismatch")
            commit_route = prefix_rows[0]["direct"]
            commit_cost = bounded(commit_route, denominator) + (
                0.0 if branch_id == target_id else WRONG_COMMITMENT_COST
            )
            if branch_id == target_id:
                macro_route = compact_existing_route(commit_route)
            else:
                goals = [
                    branch_tx.branch_goal(geometry, branch_id), checkpoint,
                    branch_tx.branch_goal(geometry, target_id),
                ]
                macro_route = witness.compact(core.route(
                    sim, "frozen_shortest_path_compat", state["position_q"],
                    float(state["heading_rad"]), goals,
                ))
            macro_cost = bounded(macro_route, denominator)
            labels.append({
                "branch_id": branch_id,
                "branch_index": branch_index,
                "is_target": branch_id == target_id,
                "commit_cost": core.qfloat(commit_cost),
                "checkpointed_excursion_cost": core.qfloat(macro_cost),
                "option_preservation_gain": core.qfloat(commit_cost - macro_cost),
                "commit_route": compact_existing_route(commit_route),
                "checkpointed_excursion_route": macro_route,
            })
    finally:
        sim.close()
    if network:
        raise RuntimeError("network attempt observed")

    value = {
        "schema_version": "revealnav-mf2-branch-excursion-label/5.1",
        "status": "BRANCH_EXCURSION_LABEL_COMPLETE",
        "event_id": args.event_id,
        "scene_id": causal["scene_id"],
        "label_source": record["label_source"],
        "candidate_branch_ids": branch_ids,
        "target_branch_id": target_id,
        "checkpoint_prefix": q_prefix,
        "decision_prefix": decision_prefix,
        "online_feature_relative_step": decision_step,
        "normalization_denominator_actions": denominator,
        "labels": labels,
        "online_feature": {
            "path": str(feature.relative_to(ROOT)),
            "bytes": feature.stat().st_size,
            "sha256": record["sha256"],
        },
        "network_attempts": 0,
        "future_information_used_for_online_input": 0,
        "offline_target_truth_used_for_macro_cost_labels": True,
        "runtime": {
            "physical_gpu": args.gpu,
            "wall_clock_s": core.qfloat(time.monotonic() - started),
        },
        "protocol_sha256": expected_protocol,
        "gold_payload_read": False,
        "paper_result": False,
    }
    value["label_sha256"] = core.stable_sha(value)
    core.atomic_json(output, value)
    print(json.dumps({
        "event_id": args.event_id,
        "labels": len(labels),
        "route_failures": sum(
            not row["checkpointed_excursion_route"].get("success", False)
            for row in labels
        ),
        "output": str(output.relative_to(ROOT)),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
