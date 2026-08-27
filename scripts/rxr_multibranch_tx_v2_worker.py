#!/usr/bin/env python3
"""Generate deterministic per-branch resource labels for one MF2-CR6 event."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla").resolve()
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cr5_queue50_tx_worker as core  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
PLAN = V2 / "RXR_MULTIBRANCH_TX_V2_PLAN.json"
GEOMETRY = V2 / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json"
CONTROLLER = V2 / "RXR_MULTIBRANCH_CONTROLLER_EXECUTION_V2.json"
CAUSAL = V2 / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json"
LANGUAGE = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_GATE_V2.json"
FOLLOWER = ROOT / "third_party/ETP-R1/habitat_extensions/shortest_path_follower.py"
NORMALIZED_BUDGETS = (1.5, 2.0, 3.0, 4.0)
CONTROLLERS = ("oracle_greedy", "frozen_shortest_path_compat")


def select(rows, event_id: str):
    found = [row for row in rows if row["event_id"] == event_id]
    if len(found) != 1:
        raise RuntimeError("event identity closure failure")
    return found[0]


def compact_route(value):
    return {key: value[key] for key in (
        "status", "success", "error_type", "action_count", "actions",
        "action_sequence_sha256", "leg_action_counts", "controller_call_count",
        "collision_count", "path_length_m", "start_position_q",
        "final_position_q", "goal_q", "final_distance_m", "replay_sha256",
    )}


def branch_goal(geometry, branch_id: str):
    if geometry["target"]["branch_id"] == branch_id:
        return geometry["target"]["T_star_at_1_75m"]
    alternatives = [row for row in geometry["alternatives"]
                    if row["branch_id"] == branch_id]
    if len(alternatives) != 1:
        raise RuntimeError("branch goal closure failure")
    return alternatives[0]["T_i_at_1_75m"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents:
        raise SystemExit("output outside project")
    attempts = core.install_network_guard()
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    expected_plan_sha = os.environ.get("RXR_MULTIBRANCH_TX_PLAN_SHA256")
    if not expected_plan_sha or core.sha256_file(PLAN) != expected_plan_sha:
        raise RuntimeError("multi-branch T_X plan drift")
    if args.event_id not in plan["eligible_event_ids"]:
        raise RuntimeError("event is outside the sealed plan")
    source_paths = {
        str(path.relative_to(ROOT)): path
        for path in (GEOMETRY, CONTROLLER, CAUSAL, LANGUAGE, FOLLOWER)
    }
    for relative, path in source_paths.items():
        if core.sha256_file(core.project_file(path)) != plan[
                "source_sha256"][relative]:
            raise RuntimeError("source drift: " + relative)

    geometry_doc = json.loads(GEOMETRY.read_text())
    controller_doc = json.loads(CONTROLLER.read_text())
    causal_doc = json.loads(CAUSAL.read_text())
    language_doc = json.loads(LANGUAGE.read_text())
    geometry = select(geometry_doc["events"], args.event_id)
    controller = select(controller_doc["events"], args.event_id)
    causal = select(causal_doc["events"], args.event_id)
    language = select(language_doc["events"], args.event_id)
    branch_ids = causal["candidate_branch_ids"]
    if not (
        geometry["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
        and controller["status"] == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
        and causal["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
        and language["status"] == "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED"
        and geometry["candidate_branch_ids"] == branch_ids
        and controller["candidate_branch_ids"] == branch_ids
        and len(branch_ids) == len(set(branch_ids)) >= 2
    ):
        raise RuntimeError("upstream full-set closure failure")

    episode_id = causal["episode_id"]
    shard = BASE / "causal_frontend/frontend_shards" / (
        "ep" + episode_id + ".json"
    )
    shard_sources = {
        row["path"]: row["sha256"]
        for row in causal_doc["sources"]["frontend_shards"]
    }
    relative_shard = str(shard.relative_to(ROOT))
    if (relative_shard not in shard_sources
            or core.sha256_file(core.project_file(shard))
            != shard_sources[relative_shard]):
        raise RuntimeError("frontend shard provenance failure")
    trace = json.loads(shard.read_text())
    records = trace["prefix_records"]
    checkpoint_prefix = int(causal["Q_prefix"])
    checkpoint = geometry["trace"]["Q"]
    checkpoint_state = records[checkpoint_prefix]
    checkpoint_error = float(np.linalg.norm(
        np.asarray(checkpoint_state["position_q"], dtype=float)
        - np.asarray(checkpoint, dtype=float)
    ))
    if checkpoint_error > 1e-4:
        raise RuntimeError("checkpoint does not match causal trace")
    established = causal["branch_established_at_confirmation_prefix"]
    if any(established[branch_id] is None for branch_id in branch_ids):
        raise RuntimeError("full-set branch lacks K3 establishment")

    sim = core.make_sim(causal["scene_id"], args.gpu)
    try:
        branches = {}
        for branch_id in branch_ids:
            goal = branch_goal(geometry, branch_id)
            branch_evidence = {}
            for controller_name in CONTROLLERS:
                normalization = core.route(
                    sim, controller_name, checkpoint,
                    float(checkpoint_state["heading_rad"]), [goal]
                )
                denominator = max(
                    normalization["action_count"],
                    core.DENOMINATOR_FLOOR_ACTIONS,
                )
                prefix_indices = list(range(checkpoint_prefix, len(records)))
                rows = []
                parent_hash = None
                for prefix in prefix_indices:
                    state = records[prefix]
                    available = prefix >= int(established[branch_id])
                    if available:
                        direct = core.route(
                            sim, controller_name, state["position_q"],
                            float(state["heading_rad"]), [goal]
                        )
                        saved = core.route(
                            sim, controller_name, state["position_q"],
                            float(state["heading_rad"]), [checkpoint, goal]
                        )
                        direct = compact_route(direct)
                        saved = compact_route(saved)
                    else:
                        direct = {"status": "BRANCH_NOT_K3_ESTABLISHED",
                                  "success": False}
                        saved = {"status": "BRANCH_NOT_K3_ESTABLISHED",
                                 "success": False}
                    options = []
                    if direct["success"]:
                        options.append((direct["action_count"], 0, "direct",
                                        direct))
                    if saved["success"]:
                        options.append((saved["action_count"], 1, "saved",
                                        saved))
                    best = min(options) if options else None
                    row = {
                        "prefix_index": prefix,
                        "source_prefix_sha256": core.stable_sha(state),
                        "parent_cost_prefix_sha256": parent_hash,
                        "branch_k3_established": available,
                        "direct": direct,
                        "saved_via_checkpoint": saved,
                        "cstar_action_count": best[0] if best else None,
                        "normalized_cstar": (
                            core.qfloat(best[0] / denominator) if best else None
                        ),
                        "cstar_source": best[2] if best else None,
                        "cstar_replay_sha256": (
                            best[3]["replay_sha256"] if best else None
                        ),
                    }
                    row["cost_prefix_sha256"] = core.stable_sha(row)
                    parent_hash = row["cost_prefix_sha256"]
                    rows.append(row)
                frontiers = {}
                for budget in NORMALIZED_BUDGETS:
                    absolute = budget * denominator
                    feasible = [
                        row["cstar_action_count"] is not None
                        and row["cstar_action_count"] <= absolute
                        for row in rows
                    ]
                    status, last_safe, transitions = core.classify_frontier(
                        feasible, prefix_indices
                    )
                    frontiers[str(budget)] = {
                        "normalized_budget": budget,
                        "absolute_action_budget": absolute,
                        "status": status,
                        "last_safe_prefix": last_safe,
                        "feasibility_transition_count": transitions,
                        "feasibility_sha256": core.stable_sha(feasible),
                    }
                branch_evidence[controller_name] = {
                    "checkpoint_to_branch_normalization": compact_route(
                        normalization
                    ),
                    "normalization_denominator_actions": denominator,
                    "prefix_costs": rows,
                    "frontiers": frontiers,
                    "complete_prefix_evidence": len(rows) == len(prefix_indices),
                }
            branches[branch_id] = {
                "branch_id": branch_id,
                "goal_q": core.qpoint(goal),
                "established_prefix": int(established[branch_id]),
                "controllers": branch_evidence,
            }
    finally:
        sim.close()
    if attempts:
        raise RuntimeError("network attempt observed")
    evidence = {
        "schema_version": "revealnav-mf2-multibranch-tx-event/2",
        "event_id": args.event_id,
        "episode_id": episode_id,
        "scene_id": causal["scene_id"],
        "candidate_branch_ids": branch_ids,
        "target_branch_id": causal["target_branch_id"],
        "strict_reveal_interval": language["reveal_interval"],
        "checkpoint": {
            "prefix_index": checkpoint_prefix,
            "position_q": core.qpoint(checkpoint),
            "causal_trace_position_error_m": core.qfloat(checkpoint_error),
        },
        "normalized_budgets": list(NORMALIZED_BUDGETS),
        "branches": branches,
        "source_manifest": {
            **plan["source_sha256"], relative_shard: shard_sources[relative_shard]
        },
        "cost_semantics": (
            "branch option enters the feasible set only after causal K3 "
            "establishment; semantic target commit remains gated by Reveal"
        ),
        "future_information_used_for_online_input": 0,
        "offline_future_used_only_for_cost_and_last_passage_labels": True,
        "network_attempts": 0,
    }
    result = {
        "schema_version": "revealnav-mf2-multibranch-tx-worker-run/2",
        "event_id": args.event_id,
        "evidence": evidence,
        "event_evidence_sha256": core.stable_sha(evidence),
        "runtime": {
            "physical_gpu": args.gpu,
            "wall_clock_s": core.qfloat(time.monotonic() - started),
            "pid": os.getpid(),
        },
    }
    core.atomic_json(output, result)
    print(json.dumps({
        "event_id": args.event_id,
        "candidate_branch_count": len(branch_ids),
        "evidence_sha256": result["event_evidence_sha256"],
        "output": str(output.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
