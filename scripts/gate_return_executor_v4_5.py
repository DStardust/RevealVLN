#!/usr/bin/env python3
"""Seal and witness the V4.5 return-executor state contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r3 import OptionStatus  # noqa: E402
from revealnav_mf2r4 import CheckpointReturnExecutor, ExecutorPhase  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


CONFIRMATION = ROOT / (
    "artifacts/evaluation/mf2_r2r_unseen_fusion_v4_4_2/"
    "R2R_UNSEEN_FUSION_RESULT_V4_4_2.json"
)
DESIGN = ROOT / "artifacts/design/MF2_STATE_CONDITIONED_RETURN_EXECUTOR_V4_5.md"
SOURCE = ROOT / "revealnav_mf2r4/executor.py"
OUT = ROOT / "artifacts/evaluation/mf2_return_executor_v4_5"
PROTOCOL = OUT / "MF2_RETURN_EXECUTOR_PROTOCOL_V4_5.json"
RESULT = OUT / "MF2_RETURN_EXECUTOR_RESULT_V4_5.json"


def protocol_value() -> dict:
    confirmation = json.loads(CONFIRMATION.read_text())
    if not (
        confirmation.get("status") == "R2R_UNSEEN_FUSION_CONFIRMATION_PASS"
        and confirmation.get("shadow_actions_executed") == 0
        and confirmation.get("test_or_test_challenge_accessed") is False
    ):
        raise RuntimeError("return-executor gate precondition failed")
    return {
        "schema_version": "revealnav-mf2-return-executor-protocol/4.5",
        "status": "SEALED_BEFORE_RETURN_EXECUTOR_WITNESS",
        "contract": {
            "success_path": [
                "at_checkpoint", "exploring", "returning",
                "at_checkpoint", "committed",
            ],
            "failure_path": ["returning", "return_failed", "returning"],
            "return_target": "the exact stored public controller reference",
            "successful_excursion_branch_status": "exhausted",
            "failed_return_is_fail_closed": True,
        },
        "gates": [
            "success_path_exact",
            "failed_return_blocks_commit",
            "retry_uses_identical_return_reference",
            "unknown_branch_rejected",
            "executor_makes_no_policy_choice",
        ],
        "sources": {
            str(CONFIRMATION.relative_to(ROOT)): sha256_file(CONFIRMATION),
            str(DESIGN.relative_to(ROOT)): sha256_file(DESIGN),
            str(SOURCE.relative_to(ROOT)): sha256_file(SOURCE),
        },
        "post_excursion_training_data_read": False,
        "gold_payload_read": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed return-executor protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def run() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("return-executor protocol must be sealed")
    executor = CheckpointReturnExecutor(
        "checkpoint-7", "frozen-public-controller:checkpoint-7",
        ("branch-a", "branch-b", "branch-c"),
    )
    phases = [executor.phase.value]
    executor.start_excursion("branch-a")
    phases.append(executor.phase.value)
    first = executor.request_backtrack()
    phases.append(executor.phase.value)
    executor.report_return(False)
    failure_phase = executor.phase.value
    commit_blocked = False
    try:
        executor.commit("branch-b")
    except RuntimeError:
        commit_blocked = True
    retry = executor.retry_return()
    retry_phase = executor.phase.value
    executor.report_return(True)
    phases.append(executor.phase.value)
    exhausted = executor.branch_status["branch-a"] is OptionStatus.EXHAUSTED
    executor.commit("branch-b")
    phases.append(executor.phase.value)
    unknown_rejected = False
    fresh = CheckpointReturnExecutor("cp", "controller", ("left", "right"))
    try:
        fresh.start_excursion("unknown")
    except ValueError:
        unknown_rejected = True
    gates = {
        "success_path_exact": phases == [
            "at_checkpoint", "exploring", "returning",
            "at_checkpoint", "committed",
        ] and exhausted,
        "failed_return_blocks_commit": (
            failure_phase == "return_failed" and commit_blocked
        ),
        "retry_uses_identical_return_reference": (
            retry_phase == "returning" and retry == first
        ),
        "unknown_branch_rejected": unknown_rejected,
        "executor_makes_no_policy_choice": (
            fresh.phase is ExecutorPhase.AT_CHECKPOINT
            and fresh.active_branch is None
            and all(status is OptionStatus.UNTRIED
                    for status in fresh.branch_status.values())
        ),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-return-executor-result/4.5",
        "status": (
            "RETURN_EXECUTOR_ENGINEERING_GATE_PASS" if passed
            else "RETURN_EXECUTOR_ENGINEERING_GATE_FAIL"
        ),
        "gates": gates,
        "success_path": phases,
        "failure_phase": failure_phase,
        "retry_phase": retry_phase,
        "return_command": {
            "checkpoint_id": first.checkpoint_id,
            "controller_ref": first.controller_ref,
            "branch_id": first.branch_id,
        },
        "protocol_sha256": sha256_file(PROTOCOL),
        "post_excursion_training_data_read": False,
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": (
            "generate train-only post-excursion observations and BACKTRACK labels"
            if passed else "executor diagnosis"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps(value, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "run"))
    args = parser.parse_args()
    return seal() if args.mode == "seal" else run()


if __name__ == "__main__":
    raise SystemExit(main())
