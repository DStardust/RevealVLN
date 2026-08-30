#!/usr/bin/env python3
"""Seal, run, and verify the fixed RxR V5.22 paired method screen."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/rxr_primary_controller_worker_v5_22.py"
DESIGN = ROOT / "artifacts/design/MF2_RXR_PRIMARY_BENCHMARK_RESTORATION_V5_22.md"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/"
    "val_seen/val_seen_guide.json.gz"
)
OUT = ROOT / "artifacts/evaluation/mf2_rxr_primary_v5_22_seen_dev"
PROTOCOL = OUT / "RXR_PRIMARY_SCREEN_PROTOCOL_V5_22.json"
RESULT = OUT / "RXR_PRIMARY_SCREEN_RESULT_V5_22.json"
PROGRESS = OUT / "RXR_PRIMARY_SCREEN_PROGRESS_V5_22.json"
SEEDS = (20260826, 20260827, 20260828)
SELECTION_SALT = "revealnav-rxr-primary-v5.22-scene-balanced/1"
METRICS = (
    "success", "spl", "ndtw", "sdtw", "distance_to_goal",
    "path_length", "steps_taken", "oracle_success",
)
HIGHER = {"success", "spl", "ndtw", "sdtw", "oracle_success"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def digest(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{value}".encode()).hexdigest()


def scene_id(episode: dict) -> str:
    return Path(episode["scene_id"]).stem


def selection() -> tuple[list[dict], dict]:
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    english = [
        row for row in episodes
        if row["instruction"]["language"] in {"en-US", "en-IN"}
    ]
    ids = [str(row["episode_id"]) for row in english]
    if len(episodes) != 6746 or len(english) != 2255 or len(ids) != len(set(ids)):
        raise RuntimeError("RxR val_seen inventory drift")
    grouped = defaultdict(list)
    for row in english:
        grouped[scene_id(row)].append(row)
    if len(grouped) != 57:
        raise RuntimeError("RxR val_seen English scene inventory drift")
    chosen_scenes = sorted(grouped, key=lambda value: digest(f"scene:{value}"))[:24]
    rows = []
    for scene in chosen_scenes:
        episode = min(
            grouped[scene],
            key=lambda row: digest(f"episode:{scene}:{row['episode_id']}"),
        )
        glb = ROOT / f"third_party/ETP-R1/data/scene_datasets/mp3d/{scene}/{scene}.glb"
        navmesh = glb.with_suffix(".navmesh")
        if (
            not glb.is_file() or glb.is_symlink()
            or not navmesh.is_file() or navmesh.is_symlink()
        ):
            raise RuntimeError(f"RxR scene asset is absent: {scene}")
        rows.append({
            "episode_id": str(episode["episode_id"]),
            "trajectory_id": str(episode["trajectory_id"]),
            "scene_id": scene,
            "language": episode["instruction"]["language"],
            "selection_digest": digest(
                f"episode:{scene}:{episode['episode_id']}"
            ),
        })
    return rows, {
        "all_episodes": len(episodes),
        "english_episodes": len(english),
        "english_scenes": len(grouped),
        "selected_episodes": len(rows),
        "selected_scenes": len({row["scene_id"] for row in rows}),
    }


def protocol_value() -> dict:
    rows, counts = selection()
    return {
        "schema_version": "revealnav-rxr-primary-screen-protocol/5.22",
        "status": "SEALED_BEFORE_RXR_VAL_SEEN_TASK_METRICS",
        "scope": "RxR-CE English guide val_seen method-development screen",
        "selection_salt": SELECTION_SALT,
        "selection": rows,
        "counts": counts,
        "seeds": list(SEEDS),
        "runs": {
            "baseline": len(rows),
            "revealnav": len(rows) * len(SEEDS),
            "total": len(rows) * (1 + len(SEEDS)),
        },
        "controller": {
            "executor": "V5.17 native-first remaining-set rerank",
            "branch_q": "RxR V5.1 source_balanced, seed-paired",
            "baseline": "deterministic official ETP-R1 RxR checkpoint",
            "online_future_frames": 0,
            "online_task_metrics": False,
        },
        "paired_design": {
            "unit": "episode paired to one deterministic ETP-R1 baseline",
            "controller_uncertainty": "three fixed training seeds",
            "selection": "metadata-only one episode per deterministic scene",
            "bootstrap": "10000 episode bootstrap replicates over seed-median deltas",
        },
        "metrics": list(METRICS),
        "scientific_screen": {
            "directional": "mean SPL>0, mean nDTW>0, mean Success>=0",
            "negative_result_preserved": True,
            "no_threshold_or_cohort_repair": True,
        },
        "sources": {
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            str(DESIGN.relative_to(ROOT)): sha256_file(DESIGN),
            str(DATASET.relative_to(ROOT)): sha256_file(DATASET),
        },
        "forbidden": ["RxR val_unseen", "R2R val_unseen", "test", "test_challenge"],
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed RxR V5.22 protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "counts": value["counts"],
        "runs": value["runs"], "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def job_name(mode: str, episode: dict, seed: int | None) -> str:
    if mode == "baseline":
        return f"baseline_ep_{episode['episode_id']}"
    return f"revealnav_seed_{seed}_ep_{episode['episode_id']}"


def all_jobs(rows: list[dict]) -> list[tuple[str, int | None, dict]]:
    jobs = [("baseline", None, episode) for episode in rows]
    jobs.extend(
        ("revealnav", seed, episode)
        for seed in SEEDS for episode in rows
    )
    return jobs


def launch(run_root: Path, mode: str, seed: int | None, episode: dict, gpu: int):
    name = job_name(mode, episode, seed)
    run_dir = run_root / "runs" / name
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"{name}.stdout.log").open("w")
    stderr = (logs / f"{name}.stderr.log").open("w")
    command = [
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--episode-id", episode["episode_id"], "--mode", mode,
        "--split", "val_seen", "--run-dir", str(run_dir),
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
    )
    return {
        "name": name, "mode": mode, "seed": seed,
        "episode_id": episode["episode_id"], "gpu": gpu,
        "process": process, "streams": (stdout, stderr),
    }


def execute(preflight: bool, gpus: tuple[int, ...], resume: bool) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("RxR V5.22 protocol must be sealed before execution")
    rows = json.loads(PROTOCOL.read_text())["selection"]
    jobs = all_jobs(rows[:1])[:2] if preflight else all_jobs(rows)
    run_root = OUT / ("preflight" if preflight else "full")
    if run_root.exists() and not resume:
        raise RuntimeError(f"refusing to overwrite {run_root}")
    run_root.mkdir(parents=True, exist_ok=resume)
    completed = []
    done = set()
    if resume:
        interrupted = run_root / "interrupted"
        for run_dir in sorted((run_root / "runs").glob("*")):
            summary = run_dir / "RUN_SUMMARY.json"
            if summary.is_file() and json.loads(summary.read_text()).get("status") == "PASS":
                value = json.loads(summary.read_text())
                key = (value["mode"], value.get("seed"), value["episode_id"])
                done.add(key)
                completed.append({
                    "name": run_dir.name, "mode": key[0], "seed": key[1],
                    "episode_id": key[2], "gpu": None, "returncode": 0,
                    "recovered_complete": True,
                })
            else:
                interrupted.mkdir(parents=True, exist_ok=True)
                os.replace(run_dir, interrupted / run_dir.name)
    queue = [row for row in jobs if (row[0], row[1], row[2]["episode_id"]) not in done]
    free = sorted(set(gpus))
    if not free:
        raise RuntimeError("at least one GPU is required")
    active = []
    started = time.time()
    atomic_json(PROGRESS, {
        "status": "RUNNING", "preflight": preflight,
        "total": len(jobs), "completed": len(completed),
        "passed": len(completed), "failed": 0,
        "active": [], "queued": len(queue), "elapsed_s": 0.0,
        "last": None,
    })
    while queue or active:
        while queue and free:
            mode, seed, episode = queue.pop(0)
            active.append(launch(run_root, mode, seed, episode, free.pop(0)))
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
            free.append(item["gpu"])
            free.sort()
            atomic_json(PROGRESS, {
                "status": "RUNNING" if queue or active else "EXECUTION_COMPLETE",
                "preflight": preflight,
                "total": len(jobs),
                "completed": len(completed),
                "passed": sum(row["returncode"] == 0 for row in completed),
                "failed": sum(row["returncode"] != 0 for row in completed),
                "active": [{key: row[key] for key in ("name", "gpu")} for row in active],
                "queued": len(queue),
                "elapsed_s": round(time.time() - started, 1),
                "last": completed[-1],
            })
            print(json.dumps(completed[-1], sort_keys=True), flush=True)
    failures = [row for row in completed if row["returncode"] != 0]
    atomic_json(run_root / "RUN_STATUS.json", {
        "status": "PASS" if not failures else "FAIL",
        "preflight": preflight,
        "completed": completed,
        "failures": failures,
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
        observed = value.pop("record_hash", None)
        expected = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if observed != expected:
            return False
        previous = observed
    return True


def interval(values: list[float]) -> list[float]:
    rng = random.Random(20260828)
    draws = []
    for _ in range(10000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return [draws[249], draws[9749]]


def verify(preflight: bool) -> int:
    protocol = json.loads(PROTOCOL.read_text())
    rows = protocol["selection"][:1] if preflight else protocol["selection"]
    run_root = OUT / ("preflight" if preflight else "full")
    expected = all_jobs(rows)[:2] if preflight else all_jobs(rows)
    summaries = {}
    chain_checks = []
    for mode, seed, episode in expected:
        path = run_root / "runs" / job_name(mode, episode, seed) / "RUN_SUMMARY.json"
        if not path.is_file():
            raise RuntimeError(f"missing worker summary: {path}")
        value = json.loads(path.read_text())
        key = (mode, seed, episode["episode_id"])
        summaries[key] = value
        if (
            value.get("status") != "PASS"
            or value.get("dataset") != "RxR-CE-en"
            or value.get("split") != "val_seen"
            or value.get("val_unseen_or_test_accessed") is not False
        ):
            raise RuntimeError(f"worker boundary or status failure: {key}")
        if not isinstance(value.get("metrics"), dict):
            raise RuntimeError(f"worker task metrics absent: {key}")
        if not all(
            metric in value["metrics"]
            and math.isfinite(float(value["metrics"][metric]))
            for metric in METRICS
        ):
            raise RuntimeError(f"worker task metrics invalid: {key}")
        if mode == "revealnav":
            trace = path.parent / "controller_trace.jsonl"
            chain_checks.append(valid_chain(load_jsonl(trace)))
            if (
                value.get("expanded_q_checkpoint", {}).get("seed") != seed
                or value.get("executed_action_validation", {}).get("all_equal") is not True
            ):
                raise RuntimeError(f"controller provenance/action failure: {key}")
    activity = sum(
        value["controller"]["checkpointed_excursions"]
        for key, value in summaries.items() if key[0] == "revealnav"
    )
    deltas = {}
    if not preflight:
        for metric in METRICS:
            per_episode = []
            for episode in rows:
                episode_id = episode["episode_id"]
                baseline = float(summaries[("baseline", None, episode_id)]["metrics"][metric])
                treatment = sorted(
                    float(summaries[("revealnav", seed, episode_id)]["metrics"][metric])
                    for seed in SEEDS
                )[1]
                delta = treatment - baseline
                if metric not in HIGHER:
                    delta = -delta
                per_episode.append(delta)
            deltas[metric] = {
                "mean_benefit": sum(per_episode) / len(per_episode),
                "episode_bootstrap_95pct": interval(per_episode),
                "per_episode": per_episode,
            }
    engineering = {
        "all_expected_runs_complete": len(summaries) == len(expected),
        "all_task_metrics_finite": True,
        "all_controller_hash_chains_valid": all(chain_checks),
        "all_expanded_q_checkpoints_strict_loaded": True,
        "all_declared_actions_match_execution": True,
        "at_least_one_checkpointed_excursion": activity > 0,
        "no_val_unseen_or_test_access": True,
    }
    directional = None if preflight else (
        deltas["spl"]["mean_benefit"] > 0
        and deltas["ndtw"]["mean_benefit"] > 0
        and deltas["success"]["mean_benefit"] >= 0
    )
    value = {
        "schema_version": "revealnav-rxr-primary-screen-result/5.22",
        "status": (
            "RXR_V5_22_PREFLIGHT_PASS" if preflight and all(engineering.values())
            else "RXR_V5_22_METHOD_SCREEN_PASS" if all(engineering.values()) and directional
            else "RXR_V5_22_ENGINEERING_PASS_SCIENTIFIC_FAIL" if all(engineering.values())
            else "RXR_V5_22_ENGINEERING_FAIL"
        ),
        "preflight": preflight,
        "protocol_sha256": sha256_file(PROTOCOL),
        "runs": len(summaries),
        "controller_activity": {"checkpointed_excursions": activity},
        "engineering_gates": engineering,
        "benefit_deltas_treatment_seed_median_minus_baseline": deltas,
        "scientific_directional_gate": directional,
        "R2R_artifacts_retained": True,
        "val_unseen_or_test_accessed": False,
        "paper_result": False,
    }
    output = OUT / (
        "RXR_PRIMARY_PREFLIGHT_RESULT_V5_22.json" if preflight else RESULT.name
    )
    atomic_json(output, value)
    print(json.dumps({
        "status": value["status"], "runs": value["runs"],
        "controller_activity": value["controller_activity"],
        "engineering_gates": engineering,
        "scientific_directional_gate": directional,
        "mean_benefit": {
            key: row["mean_benefit"] for key, row in deltas.items()
        },
    }, indent=2, sort_keys=True))
    return 0 if all(engineering.values()) else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--preflight", action="store_true")
    execute_parser.add_argument("--resume", action="store_true")
    execute_parser.add_argument("--gpus", default="0")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.command == "seal":
        return seal()
    if args.command == "execute":
        gpus = tuple(int(value) for value in args.gpus.split(",") if value)
        return execute(args.preflight, gpus, args.resume)
    return verify(args.preflight)


if __name__ == "__main__":
    raise SystemExit(main())
