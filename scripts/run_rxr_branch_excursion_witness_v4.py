#!/usr/bin/env python3
"""Feasibility witness for checkpointed wrong-branch excursion labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cr5_queue50_tx_worker as core  # noqa: E402
import rxr_multibranch_tx_v2_worker as tx  # noqa: E402


EVENT_IDS = ("x0024_ep11427_hv03", "x0006_ep7289_hv03", "x0727_ep21293_hv03")
Q_MANIFEST = ROOT / (
    "artifacts/phase1/rxr_train_expansion/expiry_r3_qpair/"
    "RXR_EXPIRY_R3_Q_FEATURE_MANIFEST.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_witness_v4"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_WITNESS_PROTOCOL_V4.json"
RESULT = OUT / "RXR_BRANCH_EXCURSION_WITNESS_RESULT_V4.json"
def source_paths() -> tuple[Path, ...]:
    return (
        Q_MANIFEST,
        tx.PLAN,
        tx.GEOMETRY,
        tx.CONTROLLER,
        tx.CAUSAL,
        tx.LANGUAGE,
        ROOT / "scripts/cr5_queue50_tx_worker.py",
        ROOT / "scripts/rxr_multibranch_tx_v2_worker.py",
        tx.FOLLOWER,
    )


def protocol_value() -> dict:
    manifest = json.loads(Q_MANIFEST.read_text())
    selected = [row for row in manifest["records"] if row["event_id"] in EVENT_IDS]
    if not (
        len(selected) == len(EVENT_IDS)
        and {row["event_id"] for row in selected} == set(EVENT_IDS)
        and all(row["split"] == "train" for row in selected)
        and all(row["label_source"] == "primary_human_audited" for row in selected)
        and sorted(row["candidate_count"] for row in selected) == [2, 3, 3]
    ):
        raise RuntimeError("branch-excursion witness selection drift")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-witness/4",
        "status": "SEALED_BEFORE_BRANCH_EXCURSION_WITNESS",
        "event_ids": list(EVENT_IDS),
        "selection": "fixed train-only events spanning 2 and 3 persistent branches",
        "decision_prefix": "first causal prefix where every branch is K=3 established",
        "macro_actions": {
            "commit_branch": "current prefix -> selected branch goal",
            "checkpointed_excursion": (
                "current prefix -> selected non-target branch goal -> frozen "
                "checkpoint Q -> target branch goal"
            ),
        },
        "controller": "frozen_shortest_path_compat",
        "repeat_count": 2,
        "success_gates": {
            "all_routes_succeed": True,
            "all_wrong_branch_excursions_have_three_legs": True,
            "all_target_commits_have_one_leg": True,
            "repeated_action_and_replay_hashes_match": True,
            "only_train_events_used": True,
        },
        "scope_limit": (
            "This witness validates macro-action label feasibility only. It does "
            "not provide post-excursion observation features and therefore does "
            "not authorize a claim of fully online BACKTRACK-state Q learning."
        ),
        "sources": {
            str(path.relative_to(ROOT)): core.sha256_file(path)
            for path in source_paths()
        },
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed branch-excursion protocol drift")
    if not PROTOCOL.exists():
        core.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": core.sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def unique(rows: list[dict], event_id: str) -> dict:
    selected = [row for row in rows if row["event_id"] == event_id]
    if len(selected) != 1:
        raise RuntimeError("event identity closure failure")
    return selected[0]


def compact(route: dict) -> dict:
    return {key: route[key] for key in (
        "status", "success", "error_type", "action_count",
        "action_sequence_sha256", "leg_action_counts",
        "controller_call_count", "collision_count", "path_length_m",
        "position_trace_sha256", "start_position_q", "final_position_q",
        "goal_q", "final_distance_m", "replay_sha256",
    )}


def run_event(event_id: str, gpu: int, documents: dict[str, dict]) -> dict:
    geometry = unique(documents["geometry"]["events"], event_id)
    causal = unique(documents["causal"]["events"], event_id)
    if geometry["candidate_branch_ids"] != causal["candidate_branch_ids"]:
        raise RuntimeError("branch identity drift")
    branch_ids = causal["candidate_branch_ids"]
    target_id = causal["target_branch_id"]
    decision_prefix = max(
        int(causal["branch_established_at_confirmation_prefix"][branch_id])
        for branch_id in branch_ids
    )
    episode_id = str(causal["episode_id"])
    shard = ROOT / (
        "artifacts/phase1/rxr_train_expansion/causal_frontend/"
        f"frontend_shards/ep{episode_id}.json"
    )
    trace = json.loads(core.project_file(shard).read_text())
    state = trace["prefix_records"][decision_prefix]
    checkpoint = geometry["trace"]["Q"]
    goals = {
        branch_id: tx.branch_goal(geometry, branch_id) for branch_id in branch_ids
    }
    sim = core.make_sim(causal["scene_id"], gpu)
    try:
        actions = []
        for branch_id in branch_ids:
            route_goals = (
                [goals[branch_id]] if branch_id == target_id
                else [goals[branch_id], checkpoint, goals[target_id]]
            )
            repeats = [
                compact(core.route(
                    sim, "frozen_shortest_path_compat", state["position_q"],
                    float(state["heading_rad"]), route_goals,
                ))
                for _ in range(2)
            ]
            actions.append({
                "branch_id": branch_id,
                "is_target": branch_id == target_id,
                "macro_action": (
                    "commit_branch" if branch_id == target_id
                    else "checkpointed_excursion"
                ),
                "goal_count": len(route_goals),
                "repeat_1": repeats[0],
                "repeat_2": repeats[1],
                "deterministic": (
                    repeats[0]["action_sequence_sha256"]
                    == repeats[1]["action_sequence_sha256"]
                    and repeats[0]["replay_sha256"] == repeats[1]["replay_sha256"]
                ),
            })
    finally:
        sim.close()
    return {
        "event_id": event_id,
        "scene_id": causal["scene_id"],
        "candidate_branch_ids": branch_ids,
        "target_branch_id": target_id,
        "decision_prefix": decision_prefix,
        "checkpoint_position_q": core.qpoint(checkpoint),
        "actions": actions,
    }


def run(gpu: int) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("branch-excursion protocol must be sealed")
    attempts = core.install_network_guard()
    started = time.monotonic()
    documents = {
        "geometry": json.loads(tx.GEOMETRY.read_text()),
        "causal": json.loads(tx.CAUSAL.read_text()),
    }
    events = [run_event(event_id, gpu, documents) for event_id in EVENT_IDS]
    if attempts:
        raise RuntimeError("network attempt observed")
    action_rows = [action for event in events for action in event["actions"]]
    wrong = [row for row in action_rows if not row["is_target"]]
    target = [row for row in action_rows if row["is_target"]]
    gates = {
        "all_events_complete": len(events) == len(EVENT_IDS),
        "all_routes_succeed": all(
            row[repeat]["success"]
            for row in action_rows for repeat in ("repeat_1", "repeat_2")
        ),
        "all_wrong_branch_excursions_have_three_legs": all(
            row["goal_count"] == 3
            and len(row["repeat_1"]["leg_action_counts"]) == 3
            for row in wrong
        ),
        "all_target_commits_have_one_leg": all(
            row["goal_count"] == 1
            and len(row["repeat_1"]["leg_action_counts"]) == 1
            for row in target
        ),
        "repeated_action_and_replay_hashes_match": all(
            row["deterministic"] for row in action_rows
        ),
        "only_train_events_used": True,
        "no_network_attempts": not attempts,
        "no_gold_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-witness-result/4",
        "status": (
            "BRANCH_EXCURSION_LABEL_WITNESS_PASS" if passed
            else "BRANCH_EXCURSION_LABEL_WITNESS_FAIL"
        ),
        "events": events,
        "counts": {
            "events": len(events),
            "macro_actions": len(action_rows),
            "wrong_branch_excursions": len(wrong),
            "target_commits": len(target),
        },
        "gates": gates,
        "runtime": {
            "physical_gpu": gpu,
            "wall_clock_s": core.qfloat(time.monotonic() - started),
        },
        "protocol_sha256": core.sha256_file(PROTOCOL),
        "scope_limit": protocol_value()["scope_limit"],
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": "scale branch-excursion macros on train-only events",
    }
    core.atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"],
        "counts": value["counts"],
        "gates": gates,
        "wall_clock_s": value["runtime"]["wall_clock_s"],
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    return seal() if args.seal else run(args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
