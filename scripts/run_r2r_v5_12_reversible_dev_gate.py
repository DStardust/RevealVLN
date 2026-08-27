#!/usr/bin/env python3
"""Paired development gate for the final V5.12 reversible controller."""

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
import run_r2r_v5_11_paired_seen_gate as prior  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_aligned_native_control_opp_worker_v5_12.py"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_12_reversible_dev_gate"
PRIOR_PROTOCOL = OUT / "R2R_V5_12_REVERSIBLE_DEV_PROTOCOL.json"
PROTOCOL = OUT / "R2R_V5_12_REVERSIBLE_DEV_PROTOCOL_V2.json"
RESULT = OUT / "R2R_V5_12_REVERSIBLE_DEV_RESULT.json"
SEEDS = common.SEEDS


def protocol_value() -> dict:
    old_protocol = json.loads(prior.PROTOCOL.read_text())
    old_result = json.loads(prior.RESULT.read_text())
    if not (
        old_protocol.get("status")
        == "SEALED_BEFORE_V5_11_PAIRED_TASK_METRIC_GATE"
        and old_result.get("status")
        == "V5_11_PAIRED_PASS_NEGATIVE_OR_MIXED"
        and old_result.get("unseen_or_test_accessed") is False
    ):
        raise RuntimeError("completed V5.11 development evidence is required")
    selection = old_protocol["selection"]
    prior_protocol = json.loads(PRIOR_PROTOCOL.read_text())
    if not (
        prior_protocol.get("status") == "SEALED_V5_12_METHOD_DEVELOPMENT_GATE"
        and prior_protocol.get("treatment_runs") == 72
    ):
        raise RuntimeError("sealed V5.12 protocol V1 is required")
    return {
        "schema_version": "revealnav-r2r-v5.12-reversible-dev-protocol/2",
        "status": "SEALED_V5_12_METHOD_DEVELOPMENT_GATE_ADJUDICATED",
        "selection": selection,
        "selection_provenance": (
            "same 24 episodes and 17 scenes whose metrics were already opened "
            "by V5.11; this gate is method development, never fresh evidence"
        ),
        "seeds": list(SEEDS),
        "treatment_runs": len(selection) * len(SEEDS),
        "baseline": "identical deterministic frozen ETP-R1 trajectory",
        "paired_unit": "episode averaged across three locked model seeds",
        "uncertainty": "10000 deterministic episode bootstrap replicates",
        "success_gate": "mean SPL>0, nDTW>0, Success>=0",
        "correctness_revision": (
            "equal candidate histories; disagreement only as a one-step "
            "checkpointed excursion; physical and ETP graph restoration before "
            "resuming the retained native action"
        ),
        "v1_adjudication": {
            "prior_protocol_sha256": common.sha256_file(PRIOR_PROTOCOL),
            "observed_failures": 4,
            "single_failure_mode": (
                "trial declared at ETP max_len-1 while the frozen evaluator "
                "forces STOP independently of navigation logits"
            ),
            "repair": "suppress new trial creation only at max_len-1",
            "reuse_rule": (
                "reuse only V1 runs that returned rc=0 under executed-action "
                "identity validation; rerun all four rc=1 jobs"
            ),
            "scientific_metrics_used_for_adjudication": False,
        },
        "sources": {
            str(RUNNER.relative_to(ROOT)): common.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): common.sha256_file(WORKER),
            str(prior.PROTOCOL.relative_to(ROOT)): common.sha256_file(prior.PROTOCOL),
            str(prior.RESULT.relative_to(ROOT)): common.sha256_file(prior.RESULT),
            str(PRIOR_PROTOCOL.relative_to(ROOT)): common.sha256_file(PRIOR_PROTOCOL),
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
    base.baseline_summary = prior.baseline_summary
    base.configure_executor()


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.12 development protocol drift")
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
    result["schema_version"] = "revealnav-r2r-v5.12-reversible-dev-result/2"
    result["status"] = result["status"].replace(
        "V5_10_PAIRED_", "V5_12_REVERSIBLE_DEV_", 1
    )
    result["task_metrics_already_opened_for_method_development"] = True
    result["fresh_confirmation_claim"] = False
    common.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("seal", "run", "resume", "verify", "all")
    )
    parser.add_argument("--gpus", default="0,1")
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
        base.executor.execute(gpus, PROTOCOL.exists() and (OUT / "runs").exists())
        verify()


if __name__ == "__main__":
    main()
