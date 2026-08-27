#!/usr/bin/env python3
"""Paired val_seen metric gate for the outcome-blind V5.10 cohort."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_full_opp_gate_v5_6 as common  # noqa: E402
import run_r2r_v5_6_fresh_seen_confirm as executor  # noqa: E402
import run_r2r_v5_10_native_control_diagnostic as diagnostic  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_native_control_opp_worker_v5_10.py"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_10_paired_seen_gate"
PROTOCOL = OUT / "R2R_V5_10_PAIRED_SEEN_PROTOCOL.json"
RESULT = OUT / "R2R_V5_10_PAIRED_SEEN_RESULT.json"
TARGET_EPISODES = 24
TARGET_SCENES = 15
SEEDS = common.SEEDS
METRICS = common.METRICS
HIGHER = common.HIGHER


def _is_active(summary: dict) -> bool:
    controller = summary["controller"]
    return (
        controller["effective_commit_interventions"]
        + controller["explore_decisions"] > 0
    )


def _diagnostic_rows() -> tuple[list[dict], dict]:
    result = json.loads(diagnostic.RESULT.read_text())
    if not (
        result.get("status") == "V5_10_NATIVE_CONTROL_DIAGNOSTIC_PASS"
        and all(result.get("engineering_gates", {}).values())
        and result.get("task_metric_payload_read") is False
    ):
        raise RuntimeError("V5.10 outcome-blind diagnostic has not passed")
    protocol = json.loads(diagnostic.PROTOCOL.read_text())
    active = []
    for metadata in protocol["selection"]:
        episode_id = metadata["episode_id"]
        path = diagnostic.OUT / "runs" / f"shadow_ep_{episode_id}" / "RUN_SUMMARY.json"
        summary = json.loads(path.read_text())
        if _is_active(summary):
            active.append(metadata)
    return active, result


def protocol_value() -> dict:
    diagnostic.protocol_value()
    active, result = _diagnostic_rows()
    selected = active[:TARGET_EPISODES]
    scenes = {row["scene_id"] for row in selected}
    if len(selected) != TARGET_EPISODES or len(scenes) < TARGET_SCENES:
        raise RuntimeError(
            f"insufficient blind V5.10 cohort: {len(active)} active, "
            f"{len(scenes)} scenes in first {len(selected)}"
        )
    return {
        "schema_version": "revealnav-r2r-v5.10-paired-seen-protocol/1",
        "status": "SEALED_BEFORE_V5_10_PAIRED_TASK_METRIC_GATE",
        "selection": selected,
        "selection_rule": (
            "first 24 active episodes in the sealed V5.10 diagnostic order; "
            "activation used controller traces only, never task metrics"
        ),
        "minimum_distinct_scenes": TARGET_SCENES,
        "distinct_scenes": len(scenes),
        "seeds": list(SEEDS),
        "treatment_runs": TARGET_EPISODES * len(SEEDS),
        "baseline": (
            "identical deterministic frozen ETP-R1 shadow trajectory from "
            "the V5.10 diagnostic; metric files opened only after sealing"
        ),
        "paired_unit": "episode averaged across three locked model seeds",
        "uncertainty": "10000 deterministic episode bootstrap replicates",
        "success_gate": "mean SPL>0, nDTW>0, Success>=0",
        "diagnostic_active_episodes": result["active_episodes"],
        "sources": {
            str(RUNNER.relative_to(ROOT)): common.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): common.sha256_file(WORKER),
            str(diagnostic.PROTOCOL.relative_to(ROOT)): common.sha256_file(
                diagnostic.PROTOCOL
            ),
            str(diagnostic.RESULT.relative_to(ROOT)): common.sha256_file(
                diagnostic.RESULT
            ),
        },
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.10 paired protocol drift")
    if not PROTOCOL.exists():
        common.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "runs": value["treatment_runs"],
        "episodes": len(value["selection"]),
        "scenes": value["distinct_scenes"],
        "sha256": common.sha256_file(PROTOCOL),
    }))


def configure_executor() -> None:
    executor.WORKER = WORKER
    executor.OUT = OUT
    executor.PROTOCOL = PROTOCOL
    executor.SEEDS = SEEDS
    executor.protocol_value = protocol_value


def baseline_summary(episode_id: str) -> dict:
    run_dir = diagnostic.OUT / "runs" / f"shadow_ep_{episode_id}"
    summary = json.loads((run_dir / "RUN_SUMMARY.json").read_text())
    if not (
        summary.get("status") == "PASS"
        and summary.get("mode") == "shadow"
        and summary.get("task_metric_payload_read") is False
        and summary.get("metrics") is None
        and _is_active(summary)
    ):
        raise RuntimeError("diagnostic trajectory is not an active blind baseline")
    paired = executor.SCREEN / "runs" / f"shadow_ep_{episode_id}" / "RUN_SUMMARY.json"
    if summary["base_trace_sha256"] != json.loads(paired.read_text())["base_trace_sha256"]:
        raise RuntimeError("V5.10 diagnostic baseline trace drift")
    stats = list((run_dir / "etp_output").rglob(
        "stats_ep_ckpt_270_val_seen_r0_w1.json"
    ))
    if len(stats) != 1:
        raise RuntimeError("V5.10 baseline metric file is ambiguous")
    metrics = json.loads(stats[0].read_text()).get(episode_id)
    if metrics is None:
        raise RuntimeError("V5.10 baseline episode metric is absent")
    return metrics


def verify() -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.10 paired protocol drift")
    treatment = {}
    for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        key = (int(row["seed"]), str(row["episode_id"]))
        if key in treatment:
            raise RuntimeError("duplicate V5.10 treatment run")
        treatment[key] = row
    expected = {
        (seed, row["episode_id"])
        for seed in SEEDS for row in protocol["selection"]
    }
    baselines = {
        row["episode_id"]: baseline_summary(row["episode_id"])
        for row in protocol["selection"]
    }
    traces = [
        common.load_jsonl(path)
        for path in (OUT / "runs").glob("*/controller_trace.jsonl")
    ]
    activity_keys = (
        "commit_decisions", "effective_commit_interventions",
        "explore_decisions", "inspect_delegations", "follow_delegations",
        "checkpointed_excursions", "continue_decisions",
        "backtrack_decisions", "successful_returns", "failed_returns",
        "terminal_unresolved_excursions",
    )
    activity = {
        key: sum(row["controller"][key] for row in treatment.values())
        for key in activity_keys
    }
    engineering = {
        "all_runs_complete": set(treatment) == expected and all(
            row.get("status") == "PASS" for row in treatment.values()
        ),
        "all_metrics_finite": all(
            row.get("metrics") is not None and all(
                math.isfinite(float(row["metrics"][metric]))
                for metric in METRICS
            ) for row in treatment.values()
        ),
        "valid_hash_chains": len(traces) == len(expected) and all(
            common.valid_chain(rows) for rows in traces
        ),
        "effective_interventions_present": (
            activity["effective_commit_interventions"]
            + activity["explore_decisions"] > 0
        ),
        "all_requested_returns_succeeded": (
            activity["backtrack_decisions"] == activity["successful_returns"]
            and activity["failed_returns"] == 0
        ),
        "locked_sources_unchanged": True,
        "no_unseen_or_test_payload": True,
    }
    per_episode = {metric: {} for metric in METRICS}
    for episode in protocol["selection"]:
        episode_id = episode["episode_id"]
        for metric in METRICS:
            deltas = []
            for seed in SEEDS:
                raw = (
                    float(treatment[(seed, episode_id)]["metrics"][metric])
                    - float(baselines[episode_id][metric])
                )
                deltas.append(raw if metric in HIGHER else -raw)
            per_episode[metric][episode_id] = sum(deltas) / len(deltas)
    episodes = [row["episode_id"] for row in protocol["selection"]]
    rng = random.Random(20260827)
    boots = {metric: [] for metric in METRICS}
    for _ in range(10000):
        sample = [rng.choice(episodes) for _ in episodes]
        for metric in METRICS:
            boots[metric].append(sum(
                per_episode[metric][episode] for episode in sample
            ) / len(sample))
    aggregate = {
        metric: {
            "mean": sum(values.values()) / len(values),
            "median": common.quantile(list(values.values()), 0.5),
            "minimum": min(values.values()),
            "maximum": max(values.values()),
            "episode_bootstrap_95pct": [
                common.quantile(boots[metric], 0.025),
                common.quantile(boots[metric], 0.975),
            ],
        } for metric, values in per_episode.items()
    }
    directional = (
        aggregate["spl"]["mean"] > 0
        and aggregate["ndtw"]["mean"] > 0
        and aggregate["success"]["mean"] >= 0
    )
    statistical = (
        aggregate["spl"]["episode_bootstrap_95pct"][0] > 0
        and aggregate["ndtw"]["episode_bootstrap_95pct"][0] > 0
    )
    outcome = (
        "STATISTICALLY_POSITIVE" if statistical
        else "DIRECTIONALLY_POSITIVE_INCONCLUSIVE" if directional
        else "NEGATIVE_OR_MIXED"
    )
    result = {
        "schema_version": "revealnav-r2r-v5.10-paired-seen-result/1",
        "status": (
            f"V5_10_PAIRED_{'PASS' if all(engineering.values()) else 'FAIL'}_"
            f"{outcome}"
        ),
        "scientific_outcome": outcome,
        "engineering_gates": engineering,
        "scientific_gates": {
            "directional_positive": directional,
            "statistically_positive": statistical,
        },
        "policy_activity": activity,
        "benefit_deltas_treatment_minus_baseline": aggregate,
        "protocol_sha256": common.sha256_file(PROTOCOL),
        "baseline_metrics_opened_only_after_selection_sealed": True,
        "paper_result": False,
        "unseen_or_test_accessed": False,
    }
    common.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    configure_executor()
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
        executor.execute(gpus, args.command == "resume")
    else:
        verify()


if __name__ == "__main__":
    main()
