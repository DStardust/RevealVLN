#!/usr/bin/env python3
"""Adjudicate deterministic frozen-controller failures in the V4 witness."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cr5_queue50_tx_worker as core  # noqa: E402
import run_rxr_branch_excursion_witness_v4 as witness  # noqa: E402
import rxr_multibranch_tx_v2_worker as tx  # noqa: E402


V4 = witness.RESULT
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_witness_v4_1"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_ADJUDICATION_PROTOCOL_V4_1.json"
RESULT = OUT / "RXR_BRANCH_EXCURSION_ADJUDICATION_RESULT_V4_1.json"
EXPECTED_FAILURE = ("x0727_ep21293_hv03", "BR03")


def frozen_failures(value: dict) -> list[tuple[str, str]]:
    return [
        (event["event_id"], action["branch_id"])
        for event in value["events"]
        for action in event["actions"]
        if not action["repeat_1"]["success"]
    ]


def protocol_value() -> dict:
    failed = json.loads(V4.read_text())
    if not (
        failed.get("status") == "BRANCH_EXCURSION_LABEL_WITNESS_FAIL"
        and frozen_failures(failed) == [EXPECTED_FAILURE]
        and failed["gates"]["repeated_action_and_replay_hashes_match"] is True
        and failed.get("gold_payload_read") is False
    ):
        raise RuntimeError("V4.1 adjudication precondition failed")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-adjudication/4.1",
        "status": "SEALED_AFTER_V4_FAILURE_BEFORE_ORACLE_ADJUDICATION",
        "failure": {"event_id": EXPECTED_FAILURE[0], "branch_id": EXPECTED_FAILURE[1]},
        "question": (
            "Is the frozen-controller collision a deterministic bounded-cost "
            "label on a geometrically valid three-leg route?"
        ),
        "adjudication": (
            "Repeat the identical three-leg route twice with oracle_greedy. "
            "The original frozen failure remains unchanged."
        ),
        "pass_rule": (
            "oracle succeeds deterministically in three legs; V4 frozen failure "
            "is deterministic; all other V4 macros succeed"
        ),
        "label_rule_if_pass": (
            "frozen success uses normalized controller cost; frozen failure uses "
            "the predeclared bounded failure cost. Do not discard failures."
        ),
        "sources": {
            str(V4.relative_to(ROOT)): core.sha256_file(V4),
            str(witness.PROTOCOL.relative_to(ROOT)): core.sha256_file(witness.PROTOCOL),
            "scripts/cr5_queue50_tx_worker.py": core.sha256_file(
                ROOT / "scripts/cr5_queue50_tx_worker.py"
            ),
        },
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V4.1 protocol drift")
    if not PROTOCOL.exists():
        core.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": core.sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def run(gpu: int) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("V4.1 protocol must be sealed")
    attempts = core.install_network_guard()
    started = time.monotonic()
    geometry_doc = json.loads(tx.GEOMETRY.read_text())
    causal_doc = json.loads(tx.CAUSAL.read_text())
    event_id, branch_id = EXPECTED_FAILURE
    geometry = witness.unique(geometry_doc["events"], event_id)
    causal = witness.unique(causal_doc["events"], event_id)
    target = causal["target_branch_id"]
    prefix = max(
        int(causal["branch_established_at_confirmation_prefix"][value])
        for value in causal["candidate_branch_ids"]
    )
    shard = ROOT / (
        "artifacts/phase1/rxr_train_expansion/causal_frontend/frontend_shards/"
        f"ep{causal['episode_id']}.json"
    )
    state = json.loads(core.project_file(shard).read_text())["prefix_records"][prefix]
    goals = [
        tx.branch_goal(geometry, branch_id),
        geometry["trace"]["Q"],
        tx.branch_goal(geometry, target),
    ]
    sim = core.make_sim(causal["scene_id"], gpu)
    try:
        repeats = [
            witness.compact(core.route(
                sim, "oracle_greedy", state["position_q"],
                float(state["heading_rad"]), goals,
            ))
            for _ in range(2)
        ]
    finally:
        sim.close()
    if attempts:
        raise RuntimeError("network attempt observed")
    deterministic = (
        repeats[0]["action_sequence_sha256"] == repeats[1]["action_sequence_sha256"]
        and repeats[0]["replay_sha256"] == repeats[1]["replay_sha256"]
    )
    failed = json.loads(V4.read_text())
    action_rows = [
        action for event in failed["events"] for action in event["actions"]
    ]
    gates = {
        "exactly_one_deterministic_frozen_failure": (
            frozen_failures(failed) == [EXPECTED_FAILURE]
            and all(row["deterministic"] for row in action_rows)
        ),
        "all_other_frozen_macros_succeed": sum(
            row["repeat_1"]["success"] for row in action_rows
        ) == len(action_rows) - 1,
        "oracle_route_succeeds_twice": all(row["success"] for row in repeats),
        "oracle_route_has_three_legs": all(
            len(row["leg_action_counts"]) == 3 for row in repeats
        ),
        "oracle_repeats_are_deterministic": deterministic,
        "no_network_attempts": not attempts,
        "no_gold_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-adjudication-result/4.1",
        "status": (
            "BRANCH_EXCURSION_LABEL_FEASIBILITY_ADJUDICATED_PASS" if passed
            else "BRANCH_EXCURSION_LABEL_FEASIBILITY_ADJUDICATED_FAIL"
        ),
        "failure": {"event_id": event_id, "branch_id": branch_id},
        "original_v4_status_preserved": failed["status"],
        "oracle_repeats": repeats,
        "gates": gates,
        "interpretation": (
            "The route is geometrically valid; the frozen controller collision "
            "is a valid deterministic failure-cost target, not a corrupt event."
            if passed else "The frozen failure cannot yet be used as a bounded label."
        ),
        "runtime": {"physical_gpu": gpu, "wall_clock_s": core.qfloat(time.monotonic() - started)},
        "protocol_sha256": core.sha256_file(PROTOCOL),
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": "train-only branch-excursion label generation",
    }
    core.atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "oracle_action_count": repeats[0]["action_count"],
        "oracle_leg_action_counts": repeats[0]["leg_action_counts"],
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
