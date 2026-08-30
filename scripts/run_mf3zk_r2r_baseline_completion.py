#!/usr/bin/env python3
"""Complete exact-episode native baselines for the sealed MF3ZK cohort.

The earlier R2R train campaign selected one representative per trajectory
with a different deterministic salt.  It therefore cannot serve as a paired
baseline for every MF3ZK treatment episode.  This small, train-only campaign
fills only the missing episode IDs; it never reruns treatment episodes and
never opens a public split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
SELECTION = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_collection/"
    "MF3ZK_R2R_COLLECTION_SELECTION.json"
)
OLD_RUNS = ROOT / "artifacts/phase1/r2r_train_net_advantage/full/runs"
OUT = ROOT / "artifacts/training/mf3zk_joint_v1/r2r_baseline_completion"
RUNS = OUT / "runs"
LOGS = OUT / "logs"
LEGACY_PROTOCOL = OUT / "MF3ZK_R2R_BASELINE_COMPLETION_PROTOCOL.json"
PROTOCOL = OUT / "MF3ZK_R2R_BASELINE_COMPLETION_PROTOCOL_V2.json"
PROGRESS = OUT / "MF3ZK_R2R_BASELINE_COMPLETION_PROGRESS.json"
WORKER = ROOT / "scripts/r2r_mf3zk_train_baseline_worker.py"
JOINT_PROTOCOL = ROOT / (
    "artifacts/training/mf3zk_joint_v1/MF3ZK_JOINT_PROTOCOL.json"
)


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


def _exact_stats(directory: Path) -> list[Path]:
    if not directory.is_dir() or directory.is_symlink():
        return []
    return list(directory.rglob("stats_ep_ckpt_270_train_r0_w1.json"))


def _old_baseline_exists(episode_id: str) -> bool:
    return len(_exact_stats(OLD_RUNS / f"ep_{episode_id}")) == 1


def _load_routes() -> list[dict]:
    value = json.loads(SELECTION.read_text())
    if not (
        value.get("status") == "SEALED_BEFORE_R2R_MF3ZK_COLLECTION"
        and value.get("split") == "train"
        and value.get("task_metrics_used_for_selection") is False
        and value.get("unseen_or_test_read") is False
    ):
        raise RuntimeError("MF3ZK collection selection is not sealed")
    routes = value.get("routes")
    if not isinstance(routes, list) or len(routes) != 1200:
        raise RuntimeError("MF3ZK collection route inventory drift")
    return routes


def prepare() -> int:
    routes = _load_routes()
    missing = [row for row in routes if not _old_baseline_exists(str(row["episode_id"]))]
    value = {
        "schema_version": "revealnav-mf3zk-r2r-baseline-completion-protocol/2",
        "status": "SEALED_BEFORE_MF3ZK_BASELINE_COMPLETION_V2",
        "split": "train",
        "controller_revision": "native_etp_r1_baseline",
        "paired_with": "MF3ZK treatment episode ID exactly",
        "selection_rule": "all MF3ZK treatment routes lacking an exact-episode preexisting baseline",
        "routes": missing,
        "counts": {
            "treatment_routes": len(routes),
            "preexisting_exact_baselines": len(routes) - len(missing),
            "completion_routes": len(missing),
            "scenes": len({row["scene_id"] for row in missing}),
        },
        "sources": {
            "treatment_selection": {
                "path": str(SELECTION.relative_to(ROOT)),
                "bytes": SELECTION.stat().st_size,
                "sha256": sha256_file(SELECTION),
            },
            "joint_protocol": {
                "path": str(JOINT_PROTOCOL.relative_to(ROOT)),
                "bytes": JOINT_PROTOCOL.stat().st_size,
                "sha256": sha256_file(JOINT_PROTOCOL),
            },
            "worker": {
                "path": str(WORKER.relative_to(ROOT)),
                "bytes": WORKER.stat().st_size,
                "sha256": sha256_file(WORKER),
            },
        },
        "task_metrics_used_for_selection": False,
        "public_split_access": {
            "r2r_val_seen": False,
            "r2r_val_unseen": False,
            "r2r_test": False,
        },
        "non_conclusion": "This protocol supplies paired train-only baselines; it is not a benchmark result.",
        "repair_reason": (
            "V2 uses a train-only worker; the prior V1 attempt was parser-incompatible "
            "and produced no valid baseline summaries."
        ),
    }
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != value:
            raise RuntimeError("baseline completion protocol drift")
    else:
        atomic_json(PROTOCOL, value)
    print(json.dumps({"status": value["status"], "counts": value["counts"]}, indent=2, sort_keys=True))
    return 0


def _valid_summary(path: Path, episode_id: str) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    metrics = value.get("metrics")
    return bool(
        value.get("status") == "PASS"
        and str(value.get("episode_id")) == str(episode_id)
        and value.get("split") == "train"
        and value.get("mode") == "baseline"
        and value.get("public_unseen_accessed") is False
        and value.get("controller") is None
        and isinstance(metrics, dict)
        and all(math.isfinite(float(metrics[key])) for key in ("success", "spl", "ndtw", "sdtw"))
        and len(_exact_stats(path.parent)) == 1
    )


def run(gpus: tuple[int, ...], workers_per_gpu: int, resume: bool) -> int:
    if any(gpu not in (0, 1) for gpu in gpus):
        raise ValueError("baseline completion is restricted to free GPUs 0 and 1")
    if workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "SEALED_BEFORE_MF3ZK_BASELINE_COMPLETION_V2":
        raise RuntimeError("baseline completion protocol is not sealed")
    if protocol["sources"]["worker"]["sha256"] != sha256_file(WORKER):
        raise RuntimeError("baseline worker changed after protocol seal")
    routes = protocol["routes"]
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    queue = []
    completed = []
    for row in routes:
        episode = str(row["episode_id"])
        directory = RUNS / f"ep_{episode}"
        if resume and _valid_summary(directory / "RUN_SUMMARY.json", episode):
            completed.append({"episode_id": episode, "returncode": 0, "recovered": True})
            continue
        if directory.exists():
            stale = OUT / "interrupted" / f"ep_{episode}_{int(time.time())}"
            stale.parent.mkdir(parents=True, exist_ok=True)
            os.replace(directory, stale)
        queue.append(row)
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    active = []
    started = time.time()
    last_write = 0.0
    while queue or active:
        while queue and slots:
            row = queue.pop(0)
            gpu = slots.pop(0)
            episode = str(row["episode_id"])
            directory = RUNS / f"ep_{episode}"
            stdout = (LOGS / f"ep_{episode}.stdout").open("w")
            stderr = (LOGS / f"ep_{episode}.stderr").open("w")
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
                str(PYTHON), str(WORKER), "--episode-id", episode,
                "--run-dir", str(directory),
            ], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            active.append({
                "process": process, "gpu": gpu, "episode_id": episode,
                "streams": (stdout, stderr),
            })
        now = time.time()
        if now - last_write >= 5:
            elapsed = now - started
            rate = len(completed) / elapsed if elapsed > 0 else 0.0
            atomic_json(PROGRESS, {
                "status": "RUNNING", "total": len(routes),
                "completed": len(completed),
                "failed": sum(x.get("returncode") != 0 for x in completed),
                "queued": len(queue),
                "active": [{"episode_id": x["episode_id"], "gpu": x["gpu"]} for x in active],
                "elapsed_s": round(elapsed, 1),
                "eta_s": None if rate == 0 else round((len(routes) - len(completed)) / rate, 1),
                "monitor_command": f"{PYTHON} scripts/run_mf3zk_r2r_baseline_completion.py monitor",
            })
            last_write = now
        time.sleep(0.5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({"episode_id": item["episode_id"], "gpu": item["gpu"], "returncode": code})
            slots.append(item["gpu"])
            slots.sort()
            active.remove(item)
    failures = [x for x in completed if x.get("returncode") != 0]
    atomic_json(PROGRESS, {
        "status": "COMPLETE" if not failures else "FAIL",
        "total": len(routes), "completed": len(completed),
        "failed": len(failures), "failures": failures,
        "queued": 0, "active": [], "elapsed_s": round(time.time() - started, 1),
        "eta_s": 0,
    })
    return 0 if not failures else 2


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
    sub.add_parser("monitor")
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "run":
        return run(tuple(int(value) for value in args.gpus.split(",") if value), args.workers_per_gpu, args.resume)
    return monitor()


if __name__ == "__main__":
    raise SystemExit(main())
