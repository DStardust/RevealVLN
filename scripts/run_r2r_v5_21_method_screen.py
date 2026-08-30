#!/usr/bin/env python3
"""Seal, execute, and verify the frozen V5.21 val-seen method screen."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_full_opp_gate_v5_6 as common  # noqa: E402
import run_r2r_v5_10_paired_seen_gate as paired  # noqa: E402
import run_r2r_v5_11_paired_seen_gate as v511  # noqa: E402
import run_r2r_v5_17_method_screen as v517  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_consensus_exploration_worker_v5_21.py"
TEST = ROOT / "tests/test_r2r_consensus_exploration_v5_21.py"
METHOD = ROOT / "artifacts/design/R2R_V5_21_METHOD_PROTOCOL.json"
COHORT = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_12_reversible_dev_gate/"
    "R2R_V5_12_REVERSIBLE_DEV_PROTOCOL_V2.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_21_method_screen"
PROTOCOL = OUT / "R2R_V5_21_METHOD_SCREEN_PROTOCOL.json"
RESULT = OUT / "R2R_V5_21_METHOD_SCREEN_RESULT.json"
SEEDS = common.SEEDS
MAX_LEN = 15
FORBIDDEN_SPLITS = {"val_unseen", "test", "test_challenge"}


def _locked_method() -> dict:
    value = json.loads(METHOD.read_text())
    sources = [value["parent_spec"], value["implementation"], value["test"]]
    sources.extend(value["locked_parents"].values())
    if not (
        value.get("status")
        == "FROZEN_AFTER_ENGINEERING_DIAGNOSIS_BEFORE_V5_21_COHORT_SCREEN"
        and all(common.sha256_file(ROOT / row["path"]) == row["sha256"] for row in sources)
        and value.get("threshold_change") is False
        and value.get("new_training") is False
        and value.get("new_data") is False
    ):
        raise RuntimeError("V5.21 frozen method closure drift")
    return value


def protocol_value() -> dict:
    method = _locked_method()
    cohort = json.loads(COHORT.read_text())
    if not (
        cohort.get("status") == "SEALED_V5_12_METHOD_DEVELOPMENT_GATE_ADJUDICATED"
        and cohort.get("treatment_runs") == 72
        and cohort.get("unseen_or_test_allowed") is False
    ):
        raise RuntimeError("V5.21 fixed cohort drift")
    return {
        "schema_version": "revealnav-r2r-v5.21-method-screen-protocol/1",
        "status": "SEALED_BEFORE_V5_21_FIXED_COHORT_SCREEN_OUTCOMES",
        "selection": cohort["selection"],
        "selection_provenance": cohort["selection_provenance"],
        "distinct_scenes": len({row["scene_id"] for row in cohort["selection"]}),
        "seeds": list(SEEDS),
        "treatment_runs": len(cohort["selection"]) * len(SEEDS),
        "baseline": "identical deterministic frozen ETP-R1 trajectory",
        "paired_unit": "episode averaged across three locked controller seeds",
        "uncertainty": {
            "replicates": 10000,
            "unit": "episode bootstrap",
            "rng_seed": 20260827,
        },
        "success_gate": (
            "zero engineering failures; causal-credit trace valid; at least "
            "one executed alternative; mean SPL>0; mean nDTW>0; mean Success>=0"
        ),
        "method": method["intervention_rule"],
        "sources": {
            str(RUNNER.relative_to(ROOT)): common.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): common.sha256_file(WORKER),
            str(TEST.relative_to(ROOT)): common.sha256_file(TEST),
            str(METHOD.relative_to(ROOT)): common.sha256_file(METHOD),
            str(COHORT.relative_to(ROOT)): common.sha256_file(COHORT),
        },
        "engineering_outcomes_already_opened": method[
            "engineering_outcomes_opened_during_revision"
        ],
        "training_or_tuning_during_screen": False,
        "fresh_confirmation_claim": False,
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def configure() -> None:
    paired.WORKER = WORKER
    paired.OUT = OUT
    paired.PROTOCOL = PROTOCOL
    paired.RESULT = RESULT
    paired.SEEDS = SEEDS
    paired.protocol_value = protocol_value
    paired.baseline_summary = v511.baseline_summary
    paired.configure_executor()


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.21 method-screen protocol drift")
    if not PROTOCOL.exists():
        common.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "runs": value["treatment_runs"],
        "episodes": len(value["selection"]),
        "scenes": value["distinct_scenes"],
        "sha256": common.sha256_file(PROTOCOL),
    }))


def _post_valid(event: dict) -> bool:
    if event.get("event") != "post_decision":
        return False
    try:
        keep = float(event["median_continue_cost"])
        backtrack = float(event["median_backtrack_cost"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (keep, backtrack)):
        return False
    robust_return = backtrack < keep and event.get("ree_closed_selected_branch") is False
    if "evidence_dominance_accept" in event:
        evidence_accept = (
            event.get("selected_is_post_target_argmax") is True
            and event.get("discriminability_nondecreasing") is True
        )
        if event.get("evidence_dominance_accept") is not evidence_accept:
            return False
        expected_return = robust_return or not evidence_accept
    else:
        expected_return = robust_return
    return (
        event.get("robust_estimator") == "three_head_coordinatewise_median"
        and event.get("robust_median_backtrack") is (backtrack < keep)
        and event.get("executed_return") is expected_return
        and event.get("policy_action")
        == ("backtrack" if expected_return else "continue")
    )


def _credit_trace_valid(rows: list[dict]) -> bool:
    credit = False
    last_gate = None
    expected_credit_grant = False
    for event in rows:
        kind = event.get("event")
        if kind == "consensus_exploration_gate":
            consensus = event.get("action_consensus") is True
            if event.get("evidence_credit_available") is not credit:
                return False
            expected = (
                "consensus_information_probe" if consensus
                else "evidence_conditioned_action_update" if credit
                else "native_first_no_direct_replacement"
            )
            if event.get("decision") != expected:
                return False
            if expected == "evidence_conditioned_action_update":
                credit = False
            last_gate = event
        elif kind == "alternative_first_trial_created":
            if last_gate is None or last_gate.get("step") != event.get("step"):
                return False
            if last_gate.get("decision") not in (
                "consensus_information_probe",
                "evidence_conditioned_action_update",
            ):
                return False
        elif kind == "post_decision":
            if not _post_valid(event):
                return False
            expected_credit_grant = bool(
                last_gate is not None
                and last_gate.get("decision") == "consensus_information_probe"
                and event.get("executed_return") is False
            )
        elif kind == "evidence_credit_granted":
            if not expected_credit_grant:
                return False
            credit = True
            expected_credit_grant = False
    return not expected_credit_grant


def _return_transactions_valid(rows: list[dict]) -> bool:
    phase = "idle"
    for event in rows:
        kind = event.get("event")
        if kind == "post_decision" and event.get("executed_return") is True:
            if phase != "idle" or not _post_valid(event):
                return False
            phase = "post"
        elif kind == "return_scheduled":
            if phase != "post":
                return False
            phase = "scheduled"
        elif kind == "return_complete":
            if phase != "scheduled" or event.get("success") is not True:
                return False
            phase = "returned"
        elif kind == "remaining_set_rerank_armed":
            if phase != "returned" or event.get("return_verified") is not True:
                return False
            phase = "armed"
        elif kind == "checkpoint_topology_restored":
            if phase != "armed" or event.get("transient_current_id_rewritten") is not True:
                return False
            phase = "restored"
        elif kind in ("remaining_set_probe_created", "remaining_set_rerank_committed"):
            if phase != "restored" or not v517._rerank_event_valid(event):
                return False
            phase = "idle"
    return phase == "idle"


def verify() -> None:
    paired.verify()
    result = json.loads(RESULT.read_text())
    summaries = [
        json.loads(path.read_text())
        for path in sorted((OUT / "runs").glob("*/RUN_SUMMARY.json"))
    ]
    traces = [
        common.load_jsonl(path)
        for path in sorted((OUT / "runs").glob("*/controller_trace.jsonl"))
    ]
    events = [event for trace in traces for event in trace]
    protocol = protocol_value()
    sealed = json.loads(PROTOCOL.read_text())
    source_hashes_valid = sealed == protocol and all(
        common.sha256_file(ROOT / path) == digest
        for path, digest in sealed["sources"].items()
    )
    trials = [row for row in events if row.get("event") == "alternative_first_trial_created"]
    returns = [
        row for row in events
        if row.get("event") == "post_decision" and row.get("executed_return") is True
    ]
    controller = {
        key: sum(row["controller"][key] for row in summaries)
        for key in (
            "checkpointed_excursions", "continue_decisions", "backtrack_decisions",
            "successful_returns", "failed_returns", "terminal_unresolved_excursions",
        )
    }
    engineering = result["engineering_gates"]
    engineering.update({
        "all_v5_21_workers": all(
            row.get("schema_version") == "revealnav-r2r-worker/5.21"
            for row in summaries
        ),
        "executed_alternatives_present": bool(trials),
        "all_declared_actions_match_execution": all(
            row.get("executed_action_validation", {}).get("all_equal") is True
            for row in summaries
        ),
        "all_post_decisions_recomputed": all(
            _post_valid(row) for row in events if row.get("event") == "post_decision"
        ),
        "all_causal_credit_traces_valid": (
            len(traces) == len(summaries) and all(_credit_trace_valid(rows) for rows in traces)
        ),
        "all_return_transactions_ordered": all(
            _return_transactions_valid(rows) for rows in traces
        ),
        "returns_and_counters_exact": (
            len(returns) == controller["backtrack_decisions"]
            == controller["successful_returns"]
            and controller["failed_returns"] == 0
        ),
        "no_terminal_pending_transaction": controller[
            "terminal_unresolved_excursions"
        ] == 0,
        "locked_sources_unchanged": source_hashes_valid,
        "no_unseen_or_test_payload": all(
            row.get("split") == "val_seen"
            and not any(value in FORBIDDEN_SPLITS for value in row.get("argv", []))
            for row in summaries
        ),
    })
    directional = result["scientific_gates"]["directional_positive"]
    passed = all(engineering.values()) and directional
    result.update({
        "schema_version": "revealnav-r2r-v5.21-method-screen-result/1",
        "status": "V5_21_METHOD_SCREEN_PASS" if passed else "V5_21_METHOD_SCREEN_FAIL",
        "engineering_gates": engineering,
        "controller_activity": controller,
        "alternative_trials": len(trials),
        "executed_return_events": len(returns),
        "method_screen_pass": passed,
        "task_metrics_already_opened_for_method_development": True,
        "fresh_confirmation_claim": False,
        "paper_result": False,
        "unseen_or_test_accessed": False,
    })
    common.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "run", "resume", "verify", "all"))
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain unique GPU indices")
    if args.command == "seal":
        seal()
    elif args.command in ("run", "resume"):
        paired.executor.execute(gpus, args.command == "resume")
    elif args.command == "verify":
        verify()
    else:
        seal()
        paired.executor.execute(gpus, (OUT / "runs").exists())
        verify()


if __name__ == "__main__":
    main()
