#!/usr/bin/env python3
"""Paired confirmation of locked V5.6 on fresh active val_seen episodes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_full_opp_gate_v5_6 as common  # noqa: E402


WORKER = ROOT / "scripts/r2r_full_opp_worker_v5_6.py"
LOCK = ROOT / "locks/R2R_FULL_OPP_CONTROLLER_V5_6.json"
SCREEN = ROOT / "artifacts/evaluation/mf2_r2r_v5_6_fresh_seen_screen"
SCREEN_RESULT = SCREEN / "R2R_V5_6_FRESH_SEEN_SCREEN_RESULT.json"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_6_fresh_seen_confirm"
PROTOCOL = OUT / "R2R_V5_6_FRESH_SEEN_CONFIRM_PROTOCOL.json"
RESULT = OUT / "R2R_V5_6_FRESH_SEEN_CONFIRM_RESULT.json"
SEEDS = common.SEEDS
METRICS = common.METRICS
HIGHER = common.HIGHER


def validate_lock() -> dict:
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "LOCKED_FOR_FRESH_VAL_SEEN_CONFIRMATION":
        raise RuntimeError("V5.6 lock status drift")
    for relative, evidence in lock["source_closure"].items():
        path = ROOT / relative
        if (
            path.is_symlink() or not path.is_file()
            or path.stat().st_size != evidence["bytes"]
            or common.sha256_file(path) != evidence["sha256"]
        ):
            raise RuntimeError(f"locked V5.6 source drift: {relative}")
    return lock


def protocol_value() -> dict:
    validate_lock()
    screen = json.loads(SCREEN_RESULT.read_text())
    if not (
        screen.get("status") == "FRESH_SCREEN_PASS_CONFIRMATION_COHORT_READY"
        and all(screen.get("gates", {}).values())
        and screen.get("selection_used_task_metrics") is False
        and screen.get("task_metric_payload_read") is False
        and len(screen.get("selected_confirmation_cohort", [])) == 30
    ):
        raise RuntimeError("fresh confirmation cohort is not ready")
    return {
        "schema_version": "revealnav-r2r-v5.6-fresh-seen-confirm-protocol/1",
        "status": "SEALED_BEFORE_FRESH_PAIRED_CONFIRMATION",
        "selection": screen["selected_confirmation_cohort"],
        "seeds": list(SEEDS), "treatment_runs": 30 * len(SEEDS),
        "baseline": (
            "deterministic frozen ETP-R1 shadow trajectories acquired before "
            "selection; their task metrics were not read until this protocol"
        ),
        "paired_unit": "episode against identical deterministic base policy",
        "uncertainty": "10000 deterministic episode bootstrap replicates",
        "success_gate": "mean SPL>0, nDTW>0, Success>=0",
        "sources": {
            str(LOCK.relative_to(ROOT)): common.sha256_file(LOCK),
            str(SCREEN_RESULT.relative_to(ROOT)): common.sha256_file(SCREEN_RESULT),
        },
        "paper_result": False, "unseen_or_test_allowed": False,
    }


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed fresh confirmation protocol drift")
    if not PROTOCOL.exists():
        common.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "runs": value["treatment_runs"],
        "sha256": common.sha256_file(PROTOCOL),
    }))


def name(seed: int, episode_id: str) -> str:
    return f"revealnav_seed_{seed}_ep_{episode_id}"


def execute(gpus: tuple[int, ...], resume: bool) -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("fresh confirmation protocol must be sealed")
    runs, logs = OUT / "runs", OUT / "logs"
    if runs.exists() and not resume:
        raise RuntimeError("fresh confirmation runs exist; use resume")
    runs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    completed, queue = [], []
    for seed in SEEDS:
        for row in protocol["selection"]:
            job = name(seed, row["episode_id"])
            summary = runs / job / "RUN_SUMMARY.json"
            if resume and summary.is_file() and json.loads(summary.read_text()).get("status") == "PASS":
                completed.append({"job": job, "returncode": 0, "recovered": True})
                continue
            if (runs / job).exists():
                destination = OUT / "interrupted" / f"{job}_{int(time.time())}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(runs / job, destination)
            queue.append((seed, row["episode_id"], job))
    free, active = list(gpus), []
    while queue or active:
        while queue and free:
            seed, episode_id, job = queue.pop(0)
            gpu = free.pop(0)
            stdout = (logs / f"{job}.stdout.log").open("w")
            stderr = (logs / f"{job}.stderr.log").open("w")
            process = subprocess.Popen([
                str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
                "--episode-id", episode_id, "--mode", "revealnav",
                "--seed", str(seed), "--split", "val_seen",
                "--run-dir", str(runs / job),
            ], cwd=ROOT, env={
                **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            }, stdout=stdout, stderr=stderr)
            active.append({
                "job": job, "gpu": gpu, "process": process,
                "streams": (stdout, stderr),
            })
        time.sleep(0.5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({
                "job": item["job"], "gpu": item["gpu"], "returncode": code,
            })
            print(json.dumps(completed[-1]), flush=True)
            free.append(item["gpu"])
            free.sort()
            active.remove(item)
            common.atomic_json(OUT / "RUN_STATUS.json", {
                "status": "RUNNING" if queue or active else "COMPLETE",
                "completed": len(completed), "expected": protocol["treatment_runs"],
                "failures": [row for row in completed if row["returncode"]],
            })
    if any(row["returncode"] for row in completed):
        raise RuntimeError("fresh confirmation worker failure")


def baseline_summary(episode_id: str) -> dict:
    run_dir = SCREEN / "runs" / f"shadow_ep_{episode_id}"
    summary = json.loads((run_dir / "RUN_SUMMARY.json").read_text())
    controller = summary.get("controller") or {}
    if not (
        summary.get("status") == "PASS" and summary.get("mode") == "shadow"
        and summary.get("task_metric_payload_read") is False
        and summary.get("metrics") is None
        and controller.get("checkpointed_excursions") == 0
        and controller.get("continue_decisions") == 0
        and controller.get("backtrack_decisions") == 0
    ):
        raise RuntimeError("screen trajectory is not an unchanged baseline")
    stats = list((run_dir / "etp_output").rglob(
        "stats_ep_ckpt_270_val_seen_r0_w1.json"
    ))
    if len(stats) != 1:
        raise RuntimeError("fresh baseline metric file is ambiguous")
    metrics = json.loads(stats[0].read_text()).get(episode_id)
    if metrics is None:
        raise RuntimeError("fresh baseline episode metric is absent")
    return {"metrics": metrics, "summary": summary, "stats": stats[0]}


def verify() -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("fresh confirmation protocol drift")
    treatment = {}
    for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        treatment[(int(row["seed"]), str(row["episode_id"]))] = row
    expected = {
        (seed, row["episode_id"])
        for seed in SEEDS for row in protocol["selection"]
    }
    baselines = {
        row["episode_id"]: baseline_summary(row["episode_id"])
        for row in protocol["selection"]
    }
    traces = [
        common.load_jsonl(OUT / "runs" / name(seed, episode) / "controller_trace.jsonl")
        for seed, episode in treatment
    ]
    activity_keys = (
        "commit_decisions", "effective_commit_interventions", "explore_decisions",
        "inspect_delegations", "follow_delegations", "checkpointed_excursions",
        "continue_decisions", "backtrack_decisions", "successful_returns",
        "failed_returns", "terminal_unresolved_excursions",
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
                math.isfinite(float(row["metrics"][metric])) for metric in METRICS
            ) for row in treatment.values()
        ),
        "valid_hash_chains": all(common.valid_chain(rows) for rows in traces),
        "effective_interventions_present": (
            activity["effective_commit_interventions"] + activity["explore_decisions"] > 0
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
        base = baselines[episode_id]["metrics"]
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                raw = float(treatment[(seed, episode_id)]["metrics"][metric]) - float(base[metric])
                values.append(raw if metric in HIGHER else -raw)
            per_episode[metric][episode_id] = sum(values) / len(values)
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
            "minimum": min(values.values()), "maximum": max(values.values()),
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
        "schema_version": "revealnav-r2r-v5.6-fresh-seen-confirm-result/1",
        "status": f"FRESH_CONFIRM_{'PASS' if all(engineering.values()) else 'FAIL'}_{outcome}",
        "scientific_outcome": outcome, "engineering_gates": engineering,
        "scientific_gates": {
            "directional_positive": directional,
            "statistically_positive": statistical,
        },
        "policy_activity": activity,
        "benefit_deltas_treatment_minus_baseline": aggregate,
        "protocol_sha256": common.sha256_file(PROTOCOL),
        "baseline_metrics_opened_only_after_selection_sealed": True,
        "paper_result": False, "unseen_or_test_accessed": False,
    }
    common.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
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
        execute(gpus, args.command == "resume")
    else:
        verify()


if __name__ == "__main__":
    main()
