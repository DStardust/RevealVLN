#!/usr/bin/env python3
"""Outcome-blind paired diagnostic for the V5.11 temporal-history fix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_v5_6_fresh_seen_screen as v56  # noqa: E402
import run_r2r_v5_10_native_control_diagnostic as base  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_temporal_native_control_opp_worker_v5_11.py"
V510_PROTOCOL = base.PROTOCOL
V510_RESULT = base.RESULT
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_11_temporal_diagnostic"
PROTOCOL = OUT / "R2R_V5_11_TEMPORAL_PROTOCOL.json"
RESULT = OUT / "R2R_V5_11_TEMPORAL_RESULT.json"


def protocol_value() -> dict:
    v56.validate_lock()
    prior = json.loads(V510_RESULT.read_text())
    if not (
        prior.get("status") == "V5_10_NATIVE_CONTROL_DIAGNOSTIC_PASS"
        and all(prior.get("engineering_gates", {}).values())
        and prior.get("task_metric_payload_read") is False
    ):
        raise RuntimeError("completed V5.10 blind diagnostic is required")
    v510_protocol = json.loads(V510_PROTOCOL.read_text())
    selection = v510_protocol["selection"]
    return {
        "schema_version": "revealnav-r2r-v5.11-temporal-protocol/1",
        "status": "SEALED_BEFORE_V5_11_OUTCOME_BLIND_TEMPORAL_DIAGNOSTIC",
        "correctness_revision": (
            "append every causal navigation prefix exactly once before the "
            "unchanged K=3 native-control decision gate"
        ),
        "selection": selection,
        "runs": len(selection),
        "screen_seed": v510_protocol["screen_seed"],
        "task_metrics_must_not_be_read": True,
        "controller_actions_executed": False,
        "sources": {
            str(RUNNER.relative_to(ROOT)): v56.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): v56.sha256_file(WORKER),
            str(V510_PROTOCOL.relative_to(ROOT)): v56.sha256_file(V510_PROTOCOL),
            str(V510_RESULT.relative_to(ROOT)): v56.sha256_file(V510_RESULT),
        },
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def configure_base() -> None:
    base.WORKER = WORKER
    base.OUT = OUT
    base.PROTOCOL = PROTOCOL
    base.RESULT = RESULT
    base.protocol_value = protocol_value
    base.configure_base()


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.11 diagnostic protocol drift")
    if not PROTOCOL.exists():
        v56.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "runs": value["runs"],
        "sha256": v56.sha256_file(PROTOCOL),
    }))


def verify() -> None:
    base.verify()
    result = json.loads(RESULT.read_text())
    if result.get("status") != "V5_10_NATIVE_CONTROL_DIAGNOSTIC_PASS":
        raise RuntimeError("shared native-control verification failed")
    summaries = [
        json.loads(path.read_text())
        for path in (OUT / "runs").glob("*/RUN_SUMMARY.json")
    ]
    if not all(
        row.get("schema_version") == "revealnav-r2r-full-opp-worker/5.11"
        for row in summaries
    ):
        raise RuntimeError("V5.11 worker schema drift")
    result["schema_version"] = "revealnav-r2r-v5.11-temporal-result/1"
    result["status"] = "V5_11_TEMPORAL_DIAGNOSTIC_PASS"
    result["temporal_prefixes_retained_before_gate"] = sum(
        row["safety_funnel"]["temporal_prefixes_retained_before_gate"]
        for row in summaries
    )
    result["interpretation"] = (
        "V5.11 versus identical frozen ETP traces; candidate coverage and "
        "temporal-history correctness only, with task metrics unread"
    )
    v56.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "run", "resume", "verify"))
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain unique GPU indices")
    if args.command == "seal":
        seal()
    elif args.command in ("run", "resume"):
        base.v57_run.execute(gpus, args.command == "resume")
    else:
        verify()


if __name__ == "__main__":
    main()
