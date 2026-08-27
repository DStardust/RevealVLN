#!/usr/bin/env python3
"""Generate checkpointed branch-excursion macro labels for one train event."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cr5_queue50_tx_worker as core  # noqa: E402
import run_rxr_branch_excursion_witness_v4 as witness  # noqa: E402
import rxr_multibranch_tx_v2_worker as primary  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
Q_ROOT = BASE / "expiry_r3_qpair"
Q_MANIFEST = Q_ROOT / "RXR_EXPIRY_R3_Q_FEATURE_MANIFEST.json"
PROTOCOL = BASE / (
    "branch_excursion_v4/RXR_BRANCH_EXCURSION_LABEL_PROTOCOL_V4.json"
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
        secondary = BASE / "secondary_expansion_v1"
        multi = secondary / "multibranch"
        return {
            "geometry": multi / "RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json",
            "causal": multi / "RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json",
            "shards": secondary / "causal_frontend/frontend_shards",
            "tx": multi / "tx_runs/round1",
        }
    raise ValueError("unsupported label source")


def safe_json(path: Path) -> dict:
    resolved = path.resolve()
    if ROOT not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("unsafe or missing project JSON")
    return json.loads(resolved.read_text())


def bounded(route: dict, denominator: int) -> float:
    if not route.get("success") or route.get("action_count") is None:
        return FAILURE_COST
    return min(float(route["action_count"]) / denominator, FAILURE_COST)


def existing_route(route: dict) -> dict:
    keys = (
        "status", "success", "error_type", "action_count",
        "action_sequence_sha256", "leg_action_counts", "controller_call_count",
        "collision_count", "path_length_m", "start_position_q",
        "final_position_q", "goal_q", "final_distance_m", "replay_sha256",
    )
    return {key: route.get(key) for key in keys if key in route}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.is_symlink():
        raise SystemExit("unsafe output path")
    expected_protocol = os.environ.get("RXR_BRANCH_EXCURSION_PROTOCOL_SHA256")
    if not expected_protocol or core.sha256_file(PROTOCOL) != expected_protocol:
        raise RuntimeError("branch-excursion protocol drift")
    attempts = core.install_network_guard()
    started = time.monotonic()
    manifest = safe_json(Q_MANIFEST)
    rows = [row for row in manifest["records"] if row["event_id"] == args.event_id]
    if len(rows) != 1 or rows[0]["split"] != "train":
        raise RuntimeError("worker accepts exactly one train event")
    record = rows[0]
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
    relative_step = decision_prefix - q_prefix
    if not 0 <= relative_step < int(record["steps"]):
        raise RuntimeError("decision prefix is outside the online feature horizon")
    shard = paths["shards"] / f"ep{causal['episode_id']}.json"
    trace = safe_json(shard)
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
                macro_route = existing_route(commit_route)
            else:
                goals = [
                    primary.branch_goal(geometry, branch_id), checkpoint,
                    primary.branch_goal(geometry, target_id),
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
                "commit_route": existing_route(commit_route),
                "checkpointed_excursion_route": macro_route,
            })
    finally:
        sim.close()
    if attempts:
        raise RuntimeError("network attempt observed")
    feature = (Q_ROOT / record["path"]).resolve()
    if (
        Q_ROOT not in feature.parents or feature.is_symlink()
        or feature.stat().st_size != record["bytes"]
        or core.sha256_file(feature) != record["sha256"]
    ):
        raise RuntimeError("online feature provenance drift")
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-label/4",
        "status": "BRANCH_EXCURSION_LABEL_COMPLETE",
        "event_id": args.event_id,
        "scene_id": causal["scene_id"],
        "label_source": record["label_source"],
        "candidate_branch_ids": branch_ids,
        "target_branch_id": target_id,
        "checkpoint_prefix": q_prefix,
        "decision_prefix": decision_prefix,
        "online_feature_relative_step": relative_step,
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
