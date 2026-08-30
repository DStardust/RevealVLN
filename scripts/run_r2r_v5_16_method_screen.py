#!/usr/bin/env python3
"""Run the frozen V5.16 method screen on the existing 24-episode cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_full_opp_gate_v5_6 as common  # noqa: E402
import run_r2r_v5_10_paired_seen_gate as base  # noqa: E402
import run_r2r_v5_11_paired_seen_gate as v511  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_native_first_deferred_switch_worker_v5_16.py"
METHOD = ROOT / "artifacts/design/R2R_V5_16_METHOD_PROTOCOL.json"
COHORT = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_12_reversible_dev_gate/"
    "R2R_V5_12_REVERSIBLE_DEV_PROTOCOL_V2.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_16_method_screen"
PROTOCOL = OUT / "R2R_V5_16_METHOD_SCREEN_PROTOCOL.json"
RESULT = OUT / "R2R_V5_16_METHOD_SCREEN_RESULT.json"
SEEDS = common.SEEDS


def protocol_value() -> dict:
    method = json.loads(METHOD.read_text())
    cohort = json.loads(COHORT.read_text())
    if not (
        method.get("status") == "FROZEN_BEFORE_V5_16_ONLINE_OUTCOMES"
        and method["implementation"]["sha256"] == common.sha256_file(WORKER)
        and cohort.get("status")
        == "SEALED_V5_12_METHOD_DEVELOPMENT_GATE_ADJUDICATED"
        and cohort.get("treatment_runs") == 72
        and cohort.get("unseen_or_test_allowed") is False
    ):
        raise RuntimeError("V5.16 method or fixed cohort drift")
    return {
        "schema_version": "revealnav-r2r-v5.16-method-screen-protocol/1",
        "status": "SEALED_BEFORE_V5_16_METHOD_SCREEN_OUTCOMES",
        "selection": cohort["selection"],
        "selection_provenance": cohort["selection_provenance"],
        "seeds": list(SEEDS),
        "treatment_runs": 72,
        "baseline": "identical deterministic frozen ETP-R1 trajectory",
        "paired_unit": "episode averaged across three locked REE/Q seeds",
        "uncertainty": "10000 deterministic episode bootstrap replicates",
        "success_gate": (
            "zero engineering failures; >=1 return-conditioned alternative; "
            "mean SPL>0; mean nDTW>0; mean Success>=0"
        ),
        "method": "native-first delayed commitment; unanimous 3-head post-Q",
        "sources": {
            str(RUNNER.relative_to(ROOT)): common.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): common.sha256_file(WORKER),
            str(METHOD.relative_to(ROOT)): common.sha256_file(METHOD),
            str(COHORT.relative_to(ROOT)): common.sha256_file(COHORT),
        },
        "task_metrics_already_opened_for_method_development": True,
        "fresh_confirmation_claim": False,
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def configure() -> None:
    base.WORKER = WORKER
    base.OUT = OUT
    base.PROTOCOL = PROTOCOL
    base.RESULT = RESULT
    base.SEEDS = SEEDS
    base.protocol_value = protocol_value
    base.baseline_summary = v511.baseline_summary
    base.configure_executor()


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.16 screen protocol drift")
    if not PROTOCOL.exists():
        common.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "runs": value["treatment_runs"],
        "episodes": len(value["selection"]),
        "sha256": common.sha256_file(PROTOCOL),
    }))


def verify() -> None:
    base.verify()
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
        "native_first_trials", "unanimous_return_decisions",
        "ensemble_disagreement_vetoes", "ree_closed_return_vetoes",
        "alternative_commits", "alternative_unavailable",
        "return_schedule_failures", "topology_snapshots", "topology_restores",
    )
    activity = {
        key: sum(row["safety_funnel"][key] for row in summaries)
        for key in activity_keys
    }
    executed_returns = [
        event for rows in traces for event in rows
        if event.get("event") == "post_decision"
        and event.get("executed_return") is True
    ]
    engineering = result["engineering_gates"]
    engineering.update({
        "all_declared_actions_match_execution": all(
            row["executed_action_validation"]["all_equal"] for row in summaries
        ),
        "all_returns_require_unanimous_open_evidence": all(
            event.get("unanimous_backtrack") is True
            and event.get("ree_closed_selected_branch") is False
            for event in executed_returns
        ),
        "return_conditioned_alternative_present": activity["alternative_commits"] > 0,
        "restores_cover_alternative_commits": (
            activity["topology_restores"] >= activity["alternative_commits"]
        ),
        "no_return_schedule_failure": activity["return_schedule_failures"] == 0,
        "frozen_worker_unchanged": (
            protocol_value()["sources"][str(WORKER.relative_to(ROOT))]
            == common.sha256_file(WORKER)
        ),
    })
    directional = result["scientific_gates"]["directional_positive"]
    passed = all(engineering.values()) and directional
    result.update({
        "schema_version": "revealnav-r2r-v5.16-method-screen-result/1",
        "status": (
            "V5_16_METHOD_SCREEN_PASS" if passed
            else "V5_16_METHOD_SCREEN_FAIL"
        ),
        "engineering_gates": engineering,
        "native_first_activity": activity,
        "executed_return_events": len(executed_returns),
        "method_screen_pass": passed,
        "task_metrics_already_opened_for_method_development": True,
        "fresh_confirmation_claim": False,
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
        base.executor.execute(gpus, args.command == "resume")
    elif args.command == "verify":
        verify()
    else:
        seal()
        base.executor.execute(gpus, (OUT / "runs").exists())
        verify()


if __name__ == "__main__":
    main()
