#!/usr/bin/env python3
"""Seal, execute, and verify paired continuous-controller R2R metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/r2r_continuous_controller_worker_v5_2.py"
SOURCE_PROTOCOL = ROOT / (
    "artifacts/evaluation/mf2_r2r_unseen_fusion_v4_4_2/"
    "R2R_UNSEEN_FUSION_PROTOCOL_V4_4_2.json"
)
SOURCE_RESULT = ROOT / (
    "artifacts/evaluation/mf2_r2r_unseen_fusion_v4_4_2/"
    "R2R_UNSEEN_FUSION_RESULT_V4_4_2.json"
)
ACTION_LOCK = ROOT / "locks/R2R_ACTION_ENABLED_MULTISCENE_V5_1.json"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_continuous_metric_v5_2"
PROTOCOL = OUT / "R2R_CONTINUOUS_METRIC_PROTOCOL_V5_2.json"
RESULT = OUT / "R2R_CONTINUOUS_METRIC_RESULT_V5_2.json"
SEEDS = (20260826, 20260827, 20260828)
METRICS = (
    "success", "spl", "ndtw", "sdtw", "distance_to_goal",
    "path_length", "steps_taken", "oracle_success",
)
HIGHER_IS_BETTER = {"success", "spl", "ndtw", "sdtw", "oracle_success"}


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
    source = json.loads(SOURCE_PROTOCOL.read_text())
    source_result = json.loads(SOURCE_RESULT.read_text())
    lock = json.loads(ACTION_LOCK.read_text())
    if not (
        source.get("status")
        == "SEALED_BEFORE_FRESH_R2R_VAL_UNSEEN_MATCHED_FRONTIER_RUNS"
        and source_result.get("status") == "R2R_UNSEEN_FUSION_CONFIRMATION_PASS"
        and lock.get("status") == "LOCKED_BEFORE_CONTINUOUS_CONTROLLER_METRIC_GATE"
        and tuple(source["seeds"]) == SEEDS
        and len(source["selection"]) == 11
    ):
        raise RuntimeError("continuous metric precondition failed")
    return {
        "schema_version": "revealnav-r2r-continuous-metric-protocol/5.2",
        "status": "SEALED_BEFORE_CONTINUOUS_PAIRED_METRIC_RUNS",
        "scope": "R2R-CE val_unseen engineering cohort; no test payload",
        "selection": source["selection"],
        "selection_reused_without_outcome_filtering": True,
        "seeds": list(SEEDS),
        "runs": {
            "baseline": len(source["selection"]),
            "revealnav": len(source["selection"]) * len(SEEDS),
            "total": len(source["selection"]) * (1 + len(SEEDS)),
        },
        "paired_design": {
            "baseline": "one deterministic frozen ETP-R1 run per episode",
            "treatment": "one conservative RevealNav overlay run per model seed and episode",
            "base_policy_sampling": False,
            "task_seed": 100,
            "unit": "episode paired to the identical frozen ETP-R1 baseline",
            "uncertainty": (
                "10000 deterministic hierarchical bootstrap replicates: "
                "resample episodes, then sample one model seed per episode"
            ),
        },
        "controller_contract": {
            "frozen_frontend": True,
            "commit_and_defer": "ETP-R1 action remains authoritative",
            "checkpointed_excursion": "execute locked fused branch",
            "post_continue": "resume ETP-R1 from reached state",
            "post_backtrack": "execute frozen-control return as the next environment action",
            "no_forced_return": True,
        },
        "metrics": list(METRICS),
        "predeclared_interpretation": {
            "directional_positive": (
                "mean paired SPL > 0, mean paired nDTW > 0, and mean paired "
                "Success >= 0"
            ),
            "statistically_positive": (
                "hierarchical-bootstrap 95% lower bounds for SPL and nDTW > 0"
            ),
            "scientific_failure_is_preserved": True,
            "engineering_success_does_not_imply_navigation_improvement": True,
        },
        "sources": {
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            str(SOURCE_PROTOCOL.relative_to(ROOT)): sha256_file(SOURCE_PROTOCOL),
            str(SOURCE_RESULT.relative_to(ROOT)): sha256_file(SOURCE_RESULT),
            str(ACTION_LOCK.relative_to(ROOT)): sha256_file(ACTION_LOCK),
        },
        "paper_result": False,
        "test_or_test_challenge_allowed": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed continuous metric protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "runs": value["runs"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def job_name(mode: str, episode_id: str, seed: int | None) -> str:
    if mode == "baseline":
        return f"baseline_ep_{episode_id}"
    return f"revealnav_seed_{seed}_ep_{episode_id}"


def jobs(episodes: list[dict]) -> list[tuple[str, int | None, dict]]:
    rows = [("baseline", None, episode) for episode in episodes]
    rows.extend(
        ("revealnav", seed, episode)
        for seed in SEEDS for episode in episodes
    )
    return rows


def launch(
    run_root: Path, mode: str, seed: int | None, episode: dict, gpu: int,
) -> dict:
    name = job_name(mode, episode["episode_id"], seed)
    run_dir = run_root / "runs" / name
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"{name}.stdout.log").open("w")
    stderr = (logs / f"{name}.stderr.log").open("w")
    environment = {
        **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    command = [
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--episode-id", episode["episode_id"], "--mode", mode,
        "--run-dir", str(run_dir),
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
    )
    return {
        "name": name, "mode": mode, "seed": seed,
        "episode_id": episode["episode_id"], "gpu": gpu,
        "process": process, "streams": (stdout, stderr),
    }


def execute(preflight: bool, gpus: tuple[int, ...], resume: bool) -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("continuous metric protocol must be sealed")
    selection = json.loads(PROTOCOL.read_text())["selection"]
    if preflight:
        selection = [row for row in selection if row["episode_id"] == "670"]
        if len(selection) != 1:
            raise RuntimeError("preflight episode absent")
    run_root = OUT / ("preflight" if preflight else "full")
    if run_root.exists() and not resume:
        raise RuntimeError(f"refusing to overwrite {run_root}")
    run_root.mkdir(parents=True, exist_ok=resume)
    completed = []
    completed_keys = set()
    if resume:
        interrupted = run_root / "interrupted"
        for run_dir in sorted((run_root / "runs").glob("*")):
            summary_path = run_dir / "RUN_SUMMARY.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                if summary.get("status") == "PASS":
                    key = (summary["mode"], summary.get("seed"), summary["episode_id"])
                    completed_keys.add(key)
                    completed.append({
                        "name": run_dir.name, "mode": key[0], "seed": key[1],
                        "episode_id": key[2], "gpu": None, "returncode": 0,
                        "recovered_complete": True,
                    })
                    continue
            destination = interrupted / run_dir.name
            interrupted.mkdir(parents=True, exist_ok=True)
            os.replace(run_dir, destination)
    queue = [
        row for row in jobs(selection)
        if (row[0], row[1], row[2]["episode_id"]) not in completed_keys
    ]
    free_gpus = list(gpus)
    active = []
    while queue or active:
        while queue and free_gpus:
            mode, seed, episode = queue.pop(0)
            active.append(launch(
                run_root, mode, seed, episode, free_gpus.pop(0)
            ))
        time.sleep(1)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({
                key: item[key] for key in (
                    "name", "mode", "seed", "episode_id", "gpu"
                )
            } | {"returncode": code})
            active.remove(item)
            free_gpus.append(item["gpu"])
            free_gpus.sort()
            print(json.dumps(completed[-1]), flush=True)
    failures = [row for row in completed if row["returncode"] != 0]
    atomic_json(run_root / "RUN_STATUS.json", {
        "status": "PASS" if not failures else "FAIL",
        "preflight": preflight, "completed": completed, "failures": failures,
    })
    return 0 if not failures else 1


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


def quantile(rows: list[float], probability: float) -> float:
    ordered = sorted(rows)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[float], bootstrap: list[float]) -> dict:
    return {
        "mean": sum(rows) / len(rows),
        "median": quantile(rows, 0.5),
        "minimum": min(rows), "maximum": max(rows),
        "hierarchical_bootstrap_95pct": [
            quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)
        ],
    }


def verify() -> int:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("continuous metric protocol drift")
    observed = {}
    evidence_files = []
    for path in sorted((OUT / "full/runs").glob("*/RUN_SUMMARY.json")):
        row = json.loads(path.read_text())
        key = (row["mode"], row.get("seed"), row["episode_id"])
        if key in observed:
            raise RuntimeError("duplicate continuous metric run")
        observed[key] = row
        evidence_files.append(path)
    expected = {
        (mode, seed, episode["episode_id"])
        for mode, seed, episode in jobs(protocol["selection"])
    }
    reveal_rows = [
        row for key, row in observed.items() if key[0] == "revealnav"
    ]
    controller_traces = {}
    for key, row in observed.items():
        if key[0] != "revealnav":
            continue
        path = OUT / "full/runs" / job_name(key[0], key[2], key[1]) / "controller_trace.jsonl"
        controller_traces[key] = load_jsonl(path)
        evidence_files.append(path)
    finite_metrics = all(
        row.get("metrics") is not None
        and all(
            name in row["metrics"] and math.isfinite(float(row["metrics"][name]))
            for name in METRICS
        )
        for row in observed.values()
    )
    total_excursions = sum(
        row["controller"]["checkpointed_excursions"] for row in reveal_rows
    )
    total_continues = sum(
        row["controller"]["continue_decisions"] for row in reveal_rows
    )
    total_backtracks = sum(
        row["controller"]["backtrack_decisions"] for row in reveal_rows
    )
    total_returns = sum(
        row["controller"]["successful_returns"] for row in reveal_rows
    )
    total_failed_returns = sum(
        row["controller"]["failed_returns"] for row in reveal_rows
    )
    source_metrics = {}
    source_root = SOURCE_PROTOCOL.parent / "full/runs"
    for path in source_root.glob("seed_*_ep_*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        source_metrics.setdefault(row["episode_id"], []).append(row["metrics"])
    baseline_invariant = all(
        len(source_metrics.get(episode["episode_id"], [])) == len(SEEDS)
        and all(
            all(
                observed[("baseline", None, episode["episode_id"])]["metrics"][name]
                == prior[name]
                for name in METRICS
            )
            for prior in source_metrics[episode["episode_id"]]
        )
        for episode in protocol["selection"]
    )
    engineering = {
        "all_44_runs_complete": set(observed) == expected and all(
            row.get("status") == "PASS" for row in observed.values()
        ),
        "all_task_metrics_present_and_finite": finite_metrics,
        "all_controller_checkpoints_strict_loaded": all(
            row.get("controller", {}).get("strict_load") is True
            for row in reveal_rows
        ),
        "all_controller_hash_chains_valid": all(
            valid_chain(rows) for rows in controller_traces.values()
        ),
        "at_least_one_real_checkpointed_excursion": total_excursions > 0,
        "every_excursion_has_one_post_decision": (
            total_excursions == total_continues + total_backtracks
        ),
        "every_requested_return_succeeded": (
            total_backtracks == total_returns and total_failed_returns == 0
        ),
        "baseline_has_no_controller": all(
            row.get("controller") is None for key, row in observed.items()
            if key[0] == "baseline"
        ),
        "baseline_metrics_match_prior_frozen_cohort": baseline_invariant,
        "no_test_or_test_challenge_payload": True,
    }
    deltas = {name: {} for name in METRICS}
    per_run = []
    for episode in protocol["selection"]:
        episode_id = episode["episode_id"]
        baseline = observed[("baseline", None, episode_id)]["metrics"]
        for seed in SEEDS:
            treatment = observed[("revealnav", seed, episode_id)]["metrics"]
            row = {"episode_id": episode_id, "seed": seed, "delta": {}}
            for name in METRICS:
                raw = float(treatment[name]) - float(baseline[name])
                benefit = raw if name in HIGHER_IS_BETTER else -raw
                deltas[name][(episode_id, seed)] = benefit
                row["delta"][name] = benefit
            row["interventions"] = observed[
                ("revealnav", seed, episode_id)
            ]["controller"]
            per_run.append(row)
    rng = random.Random(20260827)
    episodes = [row["episode_id"] for row in protocol["selection"]]
    bootstraps = {name: [] for name in METRICS}
    for _ in range(10000):
        selected = [rng.choice(episodes) for _ in episodes]
        seeds = [rng.choice(SEEDS) for _ in episodes]
        for name in METRICS:
            bootstraps[name].append(sum(
                deltas[name][(episode_id, seed)]
                for episode_id, seed in zip(selected, seeds)
            ) / len(episodes))
    aggregate = {
        name: summarize(list(values.values()), bootstraps[name])
        for name, values in deltas.items()
    }
    per_seed = {
        str(seed): {
            name: sum(
                deltas[name][(episode_id, seed)] for episode_id in episodes
            ) / len(episodes)
            for name in METRICS
        }
        for seed in SEEDS
    }
    directional = (
        aggregate["spl"]["mean"] > 0
        and aggregate["ndtw"]["mean"] > 0
        and aggregate["success"]["mean"] >= 0
    )
    statistical = (
        aggregate["spl"]["hierarchical_bootstrap_95pct"][0] > 0
        and aggregate["ndtw"]["hierarchical_bootstrap_95pct"][0] > 0
    )
    if total_excursions == 0:
        scientific_outcome = "INACTIVE_NO_INTERVENTIONS"
    elif statistical:
        scientific_outcome = "STATISTICALLY_POSITIVE"
    elif directional:
        scientific_outcome = "DIRECTIONALLY_POSITIVE_INCONCLUSIVE"
    elif (
        aggregate["spl"]["mean"] == 0
        and aggregate["ndtw"]["mean"] == 0
        and aggregate["success"]["mean"] == 0
    ):
        scientific_outcome = "NO_MEASURED_EFFECT"
    else:
        scientific_outcome = "NEGATIVE_OR_MIXED"
    evidence = hashlib.sha256()
    for path in sorted(evidence_files):
        evidence.update(str(path.relative_to(ROOT)).encode() + b"\0")
        evidence.update(sha256_file(path).encode() + b"\0")
    engineering_pass = all(engineering.values())
    value = {
        "schema_version": "revealnav-r2r-continuous-metric-result/5.2",
        "status": (
            f"R2R_CONTINUOUS_ENGINEERING_PASS_{scientific_outcome}"
            if engineering_pass else "R2R_CONTINUOUS_ENGINEERING_FAIL"
        ),
        "engineering_gates": engineering,
        "scientific_outcome": scientific_outcome,
        "predeclared_scientific_gates": {
            "directional_positive": directional,
            "statistically_positive": statistical,
        },
        "policy_activity": {
            "checkpointed_excursions": total_excursions,
            "post_continue": total_continues,
            "post_backtrack": total_backtracks,
            "successful_returns": total_returns,
            "failed_returns": total_failed_returns,
        },
        "benefit_deltas_treatment_minus_baseline": aggregate,
        "per_seed_mean_benefit_delta": per_seed,
        "paired_runs": per_run,
        "run_evidence": {
            "files": len(evidence_files),
            "path_sha256_chain": evidence.hexdigest(),
        },
        "protocol_sha256": sha256_file(PROTOCOL),
        "test_or_test_challenge_accessed": False,
        "paper_result": False,
    }
    atomic_json(RESULT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if engineering_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("seal", "preflight", "run", "resume", "verify")
    )
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain distinct device indices")
    if args.mode == "seal":
        return seal()
    if args.mode == "preflight":
        return execute(True, gpus, False)
    if args.mode == "run":
        return execute(False, gpus, False)
    if args.mode == "resume":
        return execute(False, gpus, True)
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
