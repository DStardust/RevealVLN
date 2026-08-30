#!/usr/bin/env python3
"""Run and score the sealed scene-held-out R2R-train confirmation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
JOINT_PROTOCOL = ROOT / (
    "artifacts/training/mf3zk_joint_v1/MF3ZK_JOINT_PROTOCOL.json"
)
TRAINING_RESULT = ROOT / (
    "artifacts/training/mf3zk_joint_v1/gates/MF3ZK_JOINT_TRAINING_RESULT.json"
)
WORKER = ROOT / "scripts/r2r_mf3zk_train_confirmation_worker.py"
OUT = ROOT / "artifacts/training/mf3zk_joint_v1/confirmation"
PROTOCOL = OUT / "MF3ZK_TRAIN_CONFIRMATION_PROTOCOL.json"
PROGRESS = OUT / "MF3ZK_TRAIN_CONFIRMATION_PROGRESS.json"
RESULT = OUT / "MF3ZK_TRAIN_CONFIRMATION_RESULT.json"
MODES = ("baseline", "mf3zg", "mf3zk")
METRICS = ("success", "spl", "ndtw", "sdtw")
UTILITY_WEIGHTS = {"ndtw": 0.50, "sdtw": 0.25, "spl": 0.25}


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


def _load_routes() -> list[dict]:
    protocol = json.loads(JOINT_PROTOCOL.read_text())
    if protocol.get("status") != "SEALED_BEFORE_MF3ZK_JOINT_TRAINING":
        raise RuntimeError("joint protocol is not sealed")
    result = json.loads(TRAINING_RESULT.read_text())
    if result.get("status") != "PASS" or result.get("unseen_or_test_read") is not False:
        raise RuntimeError("MF3ZK train gates are not ready for confirmation")
    routes = protocol.get("r2r_train", {}).get("confirmation_routes")
    fit = set(protocol["r2r_train"]["fit_scenes"])
    if not isinstance(routes, list) or len(routes) != 52:
        raise RuntimeError("confirmation route inventory drift")
    if set(row["scene_id"] for row in routes) & fit:
        raise RuntimeError("confirmation route scene overlaps fit scene")
    ids = [str(row["episode_id"]) for row in routes]
    if len(ids) != len(set(ids)):
        raise RuntimeError("confirmation episode IDs are not unique")
    return routes


def prepare() -> int:
    routes = _load_routes()
    value = {
        "schema_version": "revealnav-mf3zk-r2r-train-confirmation-protocol/1",
        "status": "SEALED_BEFORE_MF3ZK_TRAIN_CONFIRMATION",
        "split": "train",
        "cohort": "R2R train scene-held-out confirmation",
        "modes": list(MODES),
        "routes": routes,
        "counts": {
            "episodes": len(routes),
            "scenes": len({row["scene_id"] for row in routes}),
            "runs": len(routes) * len(MODES),
        },
        "selection_is_independent_of_metrics": True,
        "threshold_or_model_tuning_on_confirmation": False,
        "public_split_access": {
            "r2r_val_seen": False,
            "r2r_val_unseen": False,
            "r2r_test": False,
            "rxr_val_seen": False,
            "rxr_val_unseen": False,
        },
        "sources": {
            "joint_protocol": {
                "path": str(JOINT_PROTOCOL.relative_to(ROOT)),
                "bytes": JOINT_PROTOCOL.stat().st_size,
                "sha256": sha256_file(JOINT_PROTOCOL),
            },
            "training_result": {
                "path": str(TRAINING_RESULT.relative_to(ROOT)),
                "bytes": TRAINING_RESULT.stat().st_size,
                "sha256": sha256_file(TRAINING_RESULT),
            },
            "worker": {
                "path": str(WORKER.relative_to(ROOT)),
                "bytes": WORKER.stat().st_size,
                "sha256": sha256_file(WORKER),
            },
        },
        "primary_comparison": "MF3ZK minus frozen MF3ZG on paired episodes",
        "non_conclusion": "This is a train-only confirmation, not a public benchmark result.",
    }
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != value:
            raise RuntimeError("confirmation protocol drift")
    else:
        atomic_json(PROTOCOL, value)
    print(json.dumps({"status": value["status"], "counts": value["counts"]}, indent=2, sort_keys=True))
    return 0


def _valid_summary(path: Path, episode_id: str, mode: str) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    metrics = value.get("metrics")
    valid = (
        value.get("status") == "PASS"
        and str(value.get("episode_id")) == str(episode_id)
        and value.get("split") == "train"
        and value.get("mode") == mode
        and value.get("confirmation_only") is True
        and value.get("public_unseen_accessed") is False
        and value.get("unseen_or_test_read") is False
        and isinstance(metrics, dict)
        and all(math.isfinite(float(metrics[key])) for key in METRICS)
    )
    if mode != "baseline":
        valid = valid and value.get("executed_action_validation", {}).get("all_equal") is True
    else:
        valid = valid and value.get("controller") is None
    return bool(valid)


def run(gpus: tuple[int, ...], workers_per_gpu: int, resume: bool) -> int:
    if any(gpu not in (0, 1) for gpu in gpus):
        raise ValueError("confirmation is restricted to free GPUs 0 and 1")
    if workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "SEALED_BEFORE_MF3ZK_TRAIN_CONFIRMATION":
        raise RuntimeError("confirmation protocol is not sealed")
    if protocol["sources"]["worker"]["sha256"] != sha256_file(WORKER):
        raise RuntimeError("confirmation worker changed after protocol seal")
    routes = protocol["routes"]
    RUNS = OUT / "runs"
    LOGS = OUT / "logs"
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    queue = []
    completed = []
    for route in routes:
        episode = str(route["episode_id"])
        for mode in MODES:
            name = f"{mode}_ep_{episode}"
            directory = RUNS / name
            if resume and _valid_summary(directory / "RUN_SUMMARY.json", episode, mode):
                completed.append({"job": name, "returncode": 0, "recovered": True})
                continue
            if directory.exists():
                stale = OUT / "interrupted" / f"{name}_{int(time.time())}"
                stale.parent.mkdir(parents=True, exist_ok=True)
                os.replace(directory, stale)
            queue.append((route, mode, name, directory))
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    active = []
    started = time.time()
    last_write = 0.0
    total = len(routes) * len(MODES)
    while queue or active:
        while queue and slots:
            route, mode, name, directory = queue.pop(0)
            gpu = slots.pop(0)
            stdout = (LOGS / f"{name}.stdout").open("w")
            stderr = (LOGS / f"{name}.stderr").open("w")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(ROOT),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            process = subprocess.Popen([
                str(PYTHON), str(WORKER), "--episode-id", str(route["episode_id"]),
                "--mode", mode, "--run-dir", str(directory),
            ], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            active.append({
                "process": process, "gpu": gpu, "job": name,
                "episode_id": str(route["episode_id"]), "mode": mode,
                "streams": (stdout, stderr),
            })
        now = time.time()
        if now - last_write >= 5:
            elapsed = now - started
            rate = len(completed) / elapsed if elapsed > 0 else 0.0
            atomic_json(PROGRESS, {
                "status": "RUNNING", "total": total,
                "completed": len(completed),
                "failed": sum(x.get("returncode") != 0 for x in completed),
                "queued": len(queue),
                "active": [
                    {key: item[key] for key in ("job", "episode_id", "mode", "gpu")}
                    for item in active
                ],
                "elapsed_s": round(elapsed, 1),
                "eta_s": None if rate == 0 else round((total - len(completed)) / rate, 1),
                "monitor_command": f"{PYTHON} scripts/run_mf3zk_train_confirmation.py monitor",
            })
            last_write = now
        time.sleep(0.5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({
                "job": item["job"], "episode_id": item["episode_id"],
                "mode": item["mode"], "gpu": item["gpu"], "returncode": code,
            })
            slots.append(item["gpu"])
            slots.sort()
            active.remove(item)
    failures = [item for item in completed if item.get("returncode") != 0]
    atomic_json(PROGRESS, {
        "status": "COMPLETE" if not failures else "FAIL",
        "total": total, "completed": len(completed), "failed": len(failures),
        "failures": failures, "queued": 0, "active": [],
        "elapsed_s": round(time.time() - started, 1), "eta_s": 0,
    })
    return 0 if not failures else 2


def _utility(metrics: dict) -> float:
    return sum(UTILITY_WEIGHTS[key] * float(metrics[key]) for key in UTILITY_WEIGHTS)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def _bootstrap(rows: list[dict], metrics: tuple[str, ...], seed: int = 20260830, replicates: int = 10000) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_id"])].append(row)
    scenes = sorted(grouped)
    if not scenes:
        raise RuntimeError("empty confirmation rows")
    rng = np.random.default_rng(seed)
    samples = {metric: np.zeros(replicates, dtype=np.float64) for metric in metrics}
    for index in range(replicates):
        drawn = rng.choice(scenes, size=len(scenes), replace=True)
        selected = [row for scene in drawn for row in grouped[str(scene)]]
        for metric in metrics:
            samples[metric][index] = sum(float(row[metric]) for row in selected) / len(selected)
    return {
        metric: {
            "mean": float(sum(float(row[metric]) for row in rows) / len(rows)),
            "scene_bootstrap_95pct": [
                _percentile(samples[metric], 0.025),
                _percentile(samples[metric], 0.975),
            ],
        }
        for metric in metrics
    }


def aggregate() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    routes = protocol["routes"]
    runs = OUT / "runs"
    pairs = []
    for route in routes:
        episode = str(route["episode_id"])
        summaries = {}
        for mode in MODES:
            path = runs / f"{mode}_ep_{episode}" / "RUN_SUMMARY.json"
            if not _valid_summary(path, episode, mode):
                raise RuntimeError(f"invalid confirmation summary: {path}")
            summaries[mode] = json.loads(path.read_text())
        baseline = summaries["baseline"]["metrics"]
        frozen = summaries["mf3zg"]["metrics"]
        learned = summaries["mf3zk"]["metrics"]
        row = {"episode_id": episode, "scene_id": route["scene_id"]}
        for metric in METRICS:
            row[f"mf3zg_minus_baseline_{metric}"] = float(frozen[metric]) - float(baseline[metric])
            row[f"mf3zk_minus_baseline_{metric}"] = float(learned[metric]) - float(baseline[metric])
            row[f"mf3zk_minus_mf3zg_{metric}"] = float(learned[metric]) - float(frozen[metric])
        row["mf3zg_minus_baseline_utility"] = _utility(frozen) - _utility(baseline)
        row["mf3zk_minus_baseline_utility"] = _utility(learned) - _utility(baseline)
        row["mf3zk_minus_mf3zg_utility"] = _utility(learned) - _utility(frozen)
        row["mf3zg_actions_changed"] = int((summaries["mf3zg"].get("controller") or {}).get("actions_changed", 0))
        row["mf3zk_actions_changed"] = int((summaries["mf3zk"].get("controller") or {}).get("actions_changed", 0))
        pairs.append(row)
    comparison_metrics = (*METRICS, "utility")
    def projected(prefix: str) -> list[dict]:
        result = []
        for row in pairs:
            item = {"scene_id": row["scene_id"]}
            item.update({
                metric: row[f"{prefix}_{metric}"]
                for metric in comparison_metrics
            })
            result.append(item)
        return result
    aggregate = {
        "mf3zg_minus_baseline": _bootstrap(projected("mf3zg_minus_baseline"), comparison_metrics, 20260830),
        "mf3zk_minus_baseline": _bootstrap(projected("mf3zk_minus_baseline"), comparison_metrics, 20260831),
        "mf3zk_minus_mf3zg": _bootstrap(projected("mf3zk_minus_mf3zg"), comparison_metrics, 20260832),
    }
    new = aggregate["mf3zk_minus_mf3zg"]
    gates = {
        "all_pairs_complete": len(pairs) == len(routes),
        "utility_point_positive": new["utility"]["mean"] > 0.0,
        "utility_scene_bootstrap_lower_95_positive": new["utility"]["scene_bootstrap_95pct"][0] > 0.0,
        "success_point_nonnegative": new["success"]["mean"] >= 0.0,
        "spl_point_nonnegative": new["spl"]["mean"] >= 0.0,
        "ndtw_point_nonnegative": new["ndtw"]["mean"] >= 0.0,
        "new_actions_present": sum(row["mf3zk_actions_changed"] for row in pairs) > 0,
    }
    value = {
        "schema_version": "revealnav-mf3zk-r2r-train-confirmation-result/1",
        "status": "TRAIN_CONFIRMATION_PASS" if all(gates.values()) else "TRAIN_CONFIRMATION_FAIL",
        "task_metric_run_authorized": False,
        "public_unseen_authorized": False,
        "split": "train",
        "episodes": len(pairs), "scenes": len({row["scene_id"] for row in pairs}),
        "aggregate": aggregate, "gates": gates, "per_episode": pairs,
        "source_protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "bytes": PROTOCOL.stat().st_size,
            "sha256": sha256_file(PROTOCOL),
        },
        "threshold_or_model_tuning_on_confirmation": False,
        "unseen_or_test_read": False,
        "non_conclusion": "Confirmation evidence does not authorize or represent public benchmark metrics.",
    }
    if RESULT.exists():
        raise RuntimeError("refusing to overwrite confirmation result")
    atomic_json(RESULT, value)
    print(json.dumps({"status": value["status"], "gates": gates}, indent=2, sort_keys=True))
    return 0 if value["status"] == "TRAIN_CONFIRMATION_PASS" else 2


def monitor() -> int:
    if not PROGRESS.is_file():
        print(json.dumps({"status": "NOT_STARTED", "progress": str(PROGRESS.relative_to(ROOT))}))
        return 1
    print(PROGRESS.read_text())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    child = sub.add_parser("run")
    child.add_argument("--gpus", default="0,1")
    child.add_argument("--workers-per-gpu", type=int, default=4)
    child.add_argument("--resume", action="store_true")
    sub.add_parser("aggregate")
    sub.add_parser("monitor")
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "run":
        return run(tuple(int(value) for value in args.gpus.split(",") if value), args.workers_per_gpu, args.resume)
    if args.command == "aggregate":
        return aggregate()
    return monitor()


if __name__ == "__main__":
    raise SystemExit(main())
