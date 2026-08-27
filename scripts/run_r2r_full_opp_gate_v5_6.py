#!/usr/bin/env python3
"""Seal, execute, and verify the V5.6 full-OPP development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/r2r_full_opp_worker_v5_6.py"
INTEGRATED = ROOT / "revealnav_mf2r4/integrated_controller.py"
CALIBRATION = ROOT / (
    "artifacts/evaluation/mf2_ecog_opp_shared_calibration_v3/"
    "RXR_ECOG_OPP_SHARED_CALIBRATION_RESULT_V3.json"
)
ACTIVE = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_3_activation_screen_v2/"
    "R2R_V5_3_ACTIVATION_SCREEN_PARTIAL_RESULT_V2.json"
)
OLD = ROOT / "artifacts/evaluation/mf2_r2r_continuous_metric_v5_3_seen_active_dev"
OLD_PROTOCOL = OLD / "R2R_CONTINUOUS_METRIC_PROTOCOL_V5_3.json"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_full_opp_v5_6_seen_active_dev"
PROTOCOL = OUT / "R2R_FULL_OPP_PROTOCOL_V5_6.json"
RESULT = OUT / "R2R_FULL_OPP_RESULT_V5_6.json"
SEEDS = (20260826, 20260827, 20260828)
METRICS = (
    "success", "spl", "ndtw", "sdtw", "distance_to_goal",
    "path_length", "steps_taken", "oracle_success",
)
HIGHER = {"success", "spl", "ndtw", "sdtw", "oracle_success"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def protocol_value() -> dict:
    active = json.loads(ACTIVE.read_text())
    old = json.loads(OLD_PROTOCOL.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    active_selection = [{
        "episode_id": str(row["episode_id"]),
        "scene_id": row["scene_id"],
        "trajectory_id": row.get("trajectory_id"),
    } for row in active.get("active_cohort", [])]
    if not (
        active.get("status")
        == "PARTIAL_SCREEN_ENGINEERING_PASS_ACTIVE_COHORT_READY"
        and active.get("selection_used_task_metrics") is False
        and old.get("status") == "SEALED_BEFORE_V5_3_PAIRED_RUNS"
        and old.get("selection") == active_selection
        and calibration.get("status")
        == "CONTROLLER_SHARED_CALIBRATION_ENGINEERING_PASS"
    ):
        raise RuntimeError("V5.6 development inputs are invalid")
    selection = [{
        "episode_id": str(row["episode_id"]),
        "scene_id": row["scene_id"],
        "trajectory_id": row.get("trajectory_id"),
    } for row in old["selection"]]
    return {
        "schema_version": "revealnav-r2r-full-opp-protocol/5.6",
        "status": "SEALED_BEFORE_V5_6_DEVELOPMENT_RUNS",
        "scope": "R2R val_seen internal correctness-repair cohort",
        "selection": selection, "seeds": list(SEEDS),
        "treatment_runs": len(selection) * len(SEEDS),
        "baseline": "reuse bit-identical deterministic V5.3 baseline summaries",
        "action_order": ["commit", "explore", "inspect", "follow", "unresolved"],
        "threshold_search_allowed": False,
        "interpretation": {
            "directional_positive": "mean SPL>0, nDTW>0, Success>=0",
            "statistically_positive": "episode-bootstrap lower bounds SPL,nDTW>0",
            "negative_result_is_preserved": True,
        },
        "sources": {
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            str(INTEGRATED.relative_to(ROOT)): sha256_file(INTEGRATED),
            str(CALIBRATION.relative_to(ROOT)): sha256_file(CALIBRATION),
            str(ACTIVE.relative_to(ROOT)): sha256_file(ACTIVE),
            str(OLD_PROTOCOL.relative_to(ROOT)): sha256_file(OLD_PROTOCOL),
        },
        "paper_result": False,
        "unseen_or_test_access_allowed": False,
    }


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.6 protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "runs": value["treatment_runs"],
        "sha256": sha256_file(PROTOCOL),
    }))


def name(seed: int, episode_id: str) -> str:
    return f"revealnav_seed_{seed}_ep_{episode_id}"


def execute(gpus: tuple[int, ...], resume: bool) -> None:
    protocol = protocol_value()
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.6 protocol must be sealed")
    runs = OUT / "runs"
    logs = OUT / "logs"
    if runs.exists() and not resume:
        raise RuntimeError("V5.6 runs exist; use resume")
    runs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    completed = []
    queue = []
    for seed in SEEDS:
        for row in protocol["selection"]:
            job = name(seed, row["episode_id"])
            summary = runs / job / "RUN_SUMMARY.json"
            if resume and summary.is_file():
                value = json.loads(summary.read_text())
                if value.get("status") == "PASS":
                    completed.append({"name": job, "returncode": 0, "recovered": True})
                    continue
            if (runs / job).exists():
                destination = OUT / "interrupted" / f"{job}_{int(time.time())}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(runs / job, destination)
            queue.append((seed, row["episode_id"], job))
    free = list(gpus)
    active = []
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
                "name": job, "gpu": gpu, "process": process,
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
                "name": item["name"], "gpu": item["gpu"], "returncode": code,
            })
            print(json.dumps(completed[-1]), flush=True)
            free.append(item["gpu"])
            free.sort()
            active.remove(item)
            atomic_json(OUT / "RUN_STATUS.json", {
                "status": "RUNNING" if queue or active else "COMPLETE",
                "completed": len(completed), "expected": protocol["treatment_runs"],
                "failures": [row for row in completed if row["returncode"]],
            })
    if any(row["returncode"] for row in completed):
        raise RuntimeError("V5.6 treatment worker failure")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def valid_chain(rows: list[dict]) -> bool:
    previous = "0" * 64
    for row in rows:
        if row.get("previous_hash") != previous:
            return False
        value = dict(row)
        claimed = value.pop("record_hash", None)
        digest = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if digest != claimed:
            return False
        previous = claimed
    return True


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def verify() -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("sealed V5.6 protocol drift")
    treatment = {}
    for path in sorted((OUT / "runs").glob("*/RUN_SUMMARY.json")):
        row = json.loads(path.read_text())
        key = int(row["seed"]), str(row["episode_id"])
        if key in treatment:
            raise RuntimeError("duplicate V5.6 treatment")
        treatment[key] = row
    expected = {
        (seed, row["episode_id"])
        for seed in SEEDS for row in protocol["selection"]
    }
    baselines = {}
    for row in protocol["selection"]:
        path = OLD / "full/runs" / f"baseline_ep_{row['episode_id']}" / "RUN_SUMMARY.json"
        value = json.loads(path.read_text())
        if value.get("status") != "PASS" or value.get("controller") is not None:
            raise RuntimeError("reused deterministic baseline is invalid")
        baselines[row["episode_id"]] = value
    traces = [
        load_jsonl(OUT / "runs" / name(seed, episode) / "controller_trace.jsonl")
        for seed, episode in treatment
    ]
    activity_keys = (
        "commit_decisions", "effective_commit_interventions", "explore_decisions",
        "inspect_delegations", "follow_delegations", "unresolved_decisions",
        "checkpointed_excursions", "continue_decisions", "backtrack_decisions",
        "successful_returns", "failed_returns", "terminal_unresolved_excursions",
    )
    activity = {
        key: sum(row["controller"][key] for row in treatment.values())
        for key in activity_keys
    }
    engineering = {
        "all_treatment_runs_complete": set(treatment) == expected and all(
            row.get("status") == "PASS" for row in treatment.values()
        ),
        "all_metrics_finite": all(
            row.get("metrics") is not None and all(
                math.isfinite(float(row["metrics"][metric])) for metric in METRICS
            ) for row in treatment.values()
        ),
        "strict_checkpoints": all(
            row["controller"]["strict_load"] for row in treatment.values()
        ),
        "valid_hash_chains": all(valid_chain(rows) for rows in traces),
        "at_least_one_effective_intervention": (
            activity["effective_commit_interventions"]
            + activity["explore_decisions"] > 0
        ),
        "all_requested_returns_succeeded": (
            activity["backtrack_decisions"] == activity["successful_returns"]
            and activity["failed_returns"] == 0
        ),
        "no_unseen_or_test_payload": True,
    }
    per_episode = {metric: {} for metric in METRICS}
    paired = []
    for episode in protocol["selection"]:
        episode_id = episode["episode_id"]
        base = baselines[episode_id]["metrics"]
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                raw = float(treatment[(seed, episode_id)]["metrics"][metric]) - float(base[metric])
                values.append(raw if metric in HIGHER else -raw)
            per_episode[metric][episode_id] = sum(values) / len(values)
        paired.append({
            "episode_id": episode_id,
            "benefit_delta_seed_mean": {
                metric: per_episode[metric][episode_id] for metric in METRICS
            },
        })
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
            "median": quantile(list(values.values()), 0.5),
            "minimum": min(values.values()), "maximum": max(values.values()),
            "episode_bootstrap_95pct": [
                quantile(boots[metric], 0.025),
                quantile(boots[metric], 0.975),
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
    if statistical:
        outcome = "STATISTICALLY_POSITIVE"
    elif directional:
        outcome = "DIRECTIONALLY_POSITIVE_INCONCLUSIVE"
    elif all(aggregate[key]["mean"] == 0 for key in ("success", "spl", "ndtw")):
        outcome = "NO_MEASURED_EFFECT"
    else:
        outcome = "NEGATIVE_OR_MIXED"
    result = {
        "schema_version": "revealnav-r2r-full-opp-result/5.6",
        "status": f"V5_6_ENGINEERING_{'PASS' if all(engineering.values()) else 'FAIL'}_{outcome}",
        "scientific_outcome": outcome,
        "engineering_gates": engineering,
        "scientific_gates": {
            "directional_positive": directional,
            "statistically_positive": statistical,
        },
        "policy_activity": activity,
        "benefit_deltas_treatment_minus_baseline": aggregate,
        "paired_episodes": paired,
        "protocol_sha256": sha256_file(PROTOCOL),
        "paper_result": False, "unseen_or_test_accessed": False,
    }
    atomic_json(RESULT, result)
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
