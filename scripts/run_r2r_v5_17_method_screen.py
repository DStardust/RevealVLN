#!/usr/bin/env python3
"""Run and independently verify the frozen V5.17 development screen."""

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


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_remaining_set_rerank_worker_v5_17.py"
METHOD = ROOT / "artifacts/design/R2R_V5_17_METHOD_PROTOCOL.json"
COHORT = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_12_reversible_dev_gate/"
    "R2R_V5_12_REVERSIBLE_DEV_PROTOCOL_V2.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_17_method_screen"
PROTOCOL = OUT / "R2R_V5_17_METHOD_SCREEN_PROTOCOL.json"
RESULT = OUT / "R2R_V5_17_METHOD_SCREEN_RESULT.json"
SEEDS = common.SEEDS
MAX_LEN = 15
FORBIDDEN_SPLITS = {"val_unseen", "test", "test_challenge"}


def protocol_value() -> dict:
    method = json.loads(METHOD.read_text())
    cohort = json.loads(COHORT.read_text())
    if not (
        method.get("status")
        == "FROZEN_AFTER_ENGINEERING_SMOKE_BEFORE_V5_17_COHORT_SCREEN"
        and method["implementation"]["sha256"] == common.sha256_file(WORKER)
        and method["test"]["sha256"] == common.sha256_file(
            ROOT / method["test"]["path"]
        )
        and cohort.get("status")
        == "SEALED_V5_12_METHOD_DEVELOPMENT_GATE_ADJUDICATED"
        and cohort.get("treatment_runs") == 72
        and cohort.get("unseen_or_test_allowed") is False
    ):
        raise RuntimeError("V5.17 method or fixed cohort drift")
    return {
        "schema_version": "revealnav-r2r-v5.17-method-screen-protocol/1",
        "status": "SEALED_BEFORE_V5_17_FIXED_COHORT_SCREEN_OUTCOMES",
        "selection": cohort["selection"],
        "selection_provenance": cohort["selection_provenance"],
        "distinct_scenes": len({row["scene_id"] for row in cohort["selection"]}),
        "seeds": list(SEEDS),
        "treatment_runs": 72,
        "baseline": "identical deterministic frozen ETP-R1 trajectory",
        "paired_unit": "episode averaged across three locked controller seeds",
        "uncertainty": {
            "replicates": 10000,
            "unit": "episode bootstrap",
            "rng_seed": 20260827
        },
        "success_gate": (
            "zero engineering failures; >=1 non-STOP return-conditioned "
            "remaining-set action; mean SPL>0; mean nDTW>0; mean Success>=0"
        ),
        "method": (
            "native-first; strict coordinatewise-median post-Q rejection; "
            "verified return; frozen-ETP argmax over STOP plus unexhausted "
            "checkpoint options; each option probed at most once"
        ),
        "transaction_budget": {
            "max_len": MAX_LEN,
            "probe_allowed_iff": "navigation_step < max_len - 3"
        },
        "sources": {
            str(RUNNER.relative_to(ROOT)): common.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): common.sha256_file(WORKER),
            str(METHOD.relative_to(ROOT)): common.sha256_file(METHOD),
            str(COHORT.relative_to(ROOT)): common.sha256_file(COHORT),
        },
        "training_or_tuning": False,
        "new_data": False,
        "engineering_smoke_episode_351_already_opened": True,
        "task_metrics_already_opened_for_method_development": True,
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
        raise RuntimeError("sealed V5.17 screen protocol drift")
    if not PROTOCOL.exists():
        common.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "runs": value["treatment_runs"],
        "episodes": len(value["selection"]),
        "scenes": value["distinct_scenes"],
        "sha256": common.sha256_file(PROTOCOL),
    }))


def _median_return_valid(event: dict) -> bool:
    costs = event.get("predicted_costs")
    if not isinstance(costs, list) or len(costs) != 3:
        return False
    try:
        keep = sorted(float(row["continue"]) for row in costs)[1]
        backtrack = sorted(float(row["backtrack"]) for row in costs)[1]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in (keep, backtrack))
        and backtrack < keep
        and event.get("robust_estimator")
        == "three_head_coordinatewise_median"
        and event.get("robust_median_backtrack") is True
        and event.get("ree_closed_selected_branch") is False
        and event.get("forced_stress_return") is False
        and event.get("policy_action") == "backtrack"
        and math.isclose(
            float(event.get("median_continue_cost")), keep,
            rel_tol=0.0, abs_tol=1e-7,
        )
        and math.isclose(
            float(event.get("median_backtrack_cost")), backtrack,
            rel_tol=0.0, abs_tol=1e-7,
        )
    )


def _rerank_event_valid(event: dict) -> bool:
    rows = event.get("eligible_frozen_etp_scores")
    candidates = {str(value) for value in event.get("candidate_ids", [])}
    exhausted = {str(value) for value in event.get("exhausted_option_ids", [])}
    if not isinstance(rows, list) or not rows or not candidates or not exhausted:
        return False
    finite = []
    seen_indices = set()
    for row in rows:
        index = row.get("index")
        branch = row.get("branch_id")
        if not isinstance(index, int) or index in seen_indices:
            return False
        seen_indices.add(index)
        if branch is not None and (
            str(branch) not in candidates or str(branch) in exhausted
        ):
            return False
        if row.get("finite") is True:
            score = row.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                return False
            finite.append(row)
        elif row.get("score") is not None:
            return False
    if not finite:
        return False
    chosen = max(finite, key=lambda row: (float(row["score"]), -row["index"]))
    return (
        chosen["index"] == event.get("selected_global_index")
        and chosen.get("branch_id") == event.get("branch_id")
        and event.get("selection_rule")
        == "frozen_ETP_argmax_over_STOP_and_unexhausted_options"
    )


def _trace_transaction_valid(rows: list[dict]) -> bool:
    phase = "idle"
    for event in rows:
        kind = event.get("event")
        if kind == "post_decision" and event.get("executed_return") is True:
            if phase != "idle" or not _median_return_valid(event):
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
            if event.get("rejected_native_branch") not in set(
                event.get("exhausted_option_ids", [])
            ):
                return False
            phase = "armed"
        elif kind == "checkpoint_topology_restored":
            if phase != "armed" or event.get("transient_current_id_rewritten") is not True:
                return False
            phase = "restored"
        elif kind in (
            "remaining_set_probe_created", "remaining_set_rerank_committed",
        ):
            if phase != "restored" or not _rerank_event_valid(event):
                return False
            navigation_step = event.get("navigation_step")
            if not isinstance(navigation_step, int):
                return False
            if kind == "remaining_set_probe_created":
                if navigation_step >= MAX_LEN - 3 or event.get("reversible") is not True:
                    return False
            elif event.get("branch_id") is not None:
                if (
                    event.get("commit_reason")
                    != "option_expiry_no_complete_probe_budget"
                    or navigation_step < MAX_LEN - 3
                ):
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
    activity_keys = (
        "native_first_trials", "native_first_shadow_trials",
        "switch_budget_suppressions", "robust_median_returns",
        "robust_median_disagreements", "ree_closed_return_vetoes",
        "remaining_set_probe_count", "remaining_set_rerank_commits",
        "remaining_set_stop_commits", "remaining_set_rerank_cancellations",
        "return_schedule_failures", "topology_snapshots", "topology_restores",
    )
    activity = {
        key: sum(row["safety_funnel"][key] for row in summaries)
        for key in activity_keys
    }
    controller_keys = (
        "checkpointed_excursions", "continue_decisions", "backtrack_decisions",
        "successful_returns", "failed_returns", "terminal_unresolved_excursions",
    )
    controller = {
        key: sum(row["controller"][key] for row in summaries)
        for key in controller_keys
    }
    flat_events = [event for rows in traces for event in rows]
    returns = [
        event for event in flat_events
        if event.get("event") == "post_decision"
        and event.get("executed_return") is True
    ]
    reranks = [
        event for event in flat_events
        if event.get("event") in (
            "remaining_set_probe_created", "remaining_set_rerank_committed",
        )
    ]
    non_stop_reranks = [row for row in reranks if row.get("branch_id") is not None]
    terminal_pending = [
        event for event in flat_events
        if event.get("event") == "terminal_remaining_set_rerank_not_executed"
    ]
    post_events = [
        event for event in flat_events if event.get("event") == "post_decision"
    ]
    probe_events = [
        event for event in flat_events
        if event.get("event") == "remaining_set_probe_created"
    ]
    protocol = protocol_value()
    sealed = json.loads(PROTOCOL.read_text())
    source_hashes_valid = sealed == protocol and all(
        common.sha256_file(ROOT / path) == digest
        for path, digest in sealed["sources"].items()
    )
    engineering = result["engineering_gates"]
    engineering.update({
        "effective_interventions_present": bool(non_stop_reranks),
        "all_declared_actions_match_execution": all(
            row.get("executed_action_validation", {}).get("all_equal") is True
            for row in summaries
        ),
        "all_returns_require_robust_median_open_evidence": (
            bool(returns) and all(_median_return_valid(row) for row in returns)
        ),
        "all_return_transactions_ordered": (
            len(traces) == len(summaries)
            and all(_trace_transaction_valid(rows) for rows in traces)
        ),
        "all_masked_argmax_choices_reproducible": (
            bool(reranks) and all(_rerank_event_valid(row) for row in reranks)
        ),
        "temporal_candidate_rows_aligned": (
            all(
                event.get("historical_candidate_row_steps", -1) + 1
                == event.get("temporal_history_steps")
                and event.get("final_candidate_count", 0) > 0
                for event in post_events
            )
            and all(
                event.get("temporal_candidate_row_steps")
                == event.get("temporal_history_steps")
                and set(event.get("restored_control_ids", []))
                == set(event.get("candidate_ids", []))
                for event in probe_events
            )
        ),
        "all_native_trials_budget_feasible": all(
            event.get("step", MAX_LEN) < MAX_LEN - 3
            for event in flat_events
            if event.get("event") == "native_first_trial_created"
        ),
        "return_conditioned_non_stop_rerank_present": bool(non_stop_reranks),
        "returns_restores_and_counters_exact": (
            len(returns) == activity["robust_median_returns"]
            == controller["backtrack_decisions"]
            == controller["successful_returns"]
            == activity["topology_restores"]
            and controller["failed_returns"] == 0
        ),
        "no_rerank_cancellation": activity["remaining_set_rerank_cancellations"] == 0,
        "no_return_schedule_failure": activity["return_schedule_failures"] == 0,
        "no_terminal_pending_transaction": (
            not terminal_pending and controller["terminal_unresolved_excursions"] == 0
        ),
        "v5_17_contract_recorded": all(
            row["safety_funnel"].get("post_gate")
            == "coordinatewise_median_of_three_frozen_Q_heads"
            and "unexhausted checkpoint options" in row["safety_funnel"].get(
                "intervention_contract", ""
            )
            for row in summaries
        ),
        "locked_sources_unchanged": source_hashes_valid,
        "no_unseen_or_test_payload": all(
            row.get("split") == "val_seen"
            and not any(value in FORBIDDEN_SPLITS for value in row.get("argv", []))
            for row in summaries
        ),
    })
    engineering.pop("all_returns_require_unanimous_open_evidence", None)
    engineering.pop("return_conditioned_alternative_present", None)
    engineering.pop("restores_cover_alternative_commits", None)
    directional = result["scientific_gates"]["directional_positive"]
    passed = all(engineering.values()) and directional
    result.update({
        "schema_version": "revealnav-r2r-v5.17-method-screen-result/1",
        "status": "V5_17_METHOD_SCREEN_PASS" if passed else "V5_17_METHOD_SCREEN_FAIL",
        "engineering_gates": engineering,
        "remaining_set_activity": activity,
        "controller_activity": controller,
        "executed_return_events": len(returns),
        "rerank_events": len(reranks),
        "non_stop_rerank_events": len(non_stop_reranks),
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
