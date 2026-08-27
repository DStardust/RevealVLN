#!/usr/bin/env python3
"""Resumable R2R-train feature, counterfactual-label, and training pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
WORKER = ROOT / "scripts/r2r_train_net_advantage_worker.py"
LABELER = ROOT / "scripts/build_r2r_train_net_advantage_labels.py"
TRAINER = ROOT / "scripts/train_r2r_sparse_net_advantage.py"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz"
)
BASE = ROOT / "artifacts/phase1/r2r_train_net_advantage"
PILOT_TARGET = 96


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


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


def scene_id(path: str) -> str:
    return Path(path).parts[-2]


def canonical_routes() -> list[dict]:
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    grouped = defaultdict(list)
    for row in episodes:
        grouped[str(row["trajectory_id"])].append(row)
    routes = []
    for trajectory, rows in grouped.items():
        selected = min(rows, key=lambda row: stable_hash({
            "episode_id": str(row["episode_id"]), "trajectory_id": trajectory,
        }))
        routes.append({
            "episode_id": str(selected["episode_id"]),
            "trajectory_id": trajectory,
            "scene_id": scene_id(selected["scene_id"]),
            "reference_points": len(selected["reference_path"]),
        })
    return sorted(routes, key=lambda row: stable_hash(row))


def pilot_routes(routes: list[dict]) -> list[dict]:
    by_scene = defaultdict(list)
    for row in routes:
        if row["reference_points"] >= 5:
            by_scene[row["scene_id"]].append(row)
    for rows in by_scene.values():
        rows.sort(key=lambda row: stable_hash({"pilot": row}))
    queue = deque(sorted(by_scene, key=lambda scene: stable_hash({"scene": scene})))
    selected = []
    while queue and len(selected) < PILOT_TARGET:
        scene = queue.popleft()
        selected.append(by_scene[scene].pop())
        if by_scene[scene]:
            queue.append(scene)
    if len(selected) != PILOT_TARGET:
        raise RuntimeError("insufficient scene-balanced pilot routes")
    return selected


def layout(cohort: str) -> dict[str, Path]:
    root = BASE / cohort
    return {
        "root": root,
        "runs": root / "runs",
        "selection": root / "R2R_TRAIN_NET_ADVANTAGE_SELECTION.json",
        "progress": root / "R2R_TRAIN_NET_ADVANTAGE_PROGRESS.json",
        "labels": root / "labels",
        "training": root / "training",
    }


def prepare(cohort: str) -> dict:
    paths = layout(cohort)
    paths["root"].mkdir(parents=True, exist_ok=True)
    routes = canonical_routes()
    selected = pilot_routes(routes) if cohort == "pilot" else routes
    value = {
        "schema_version": "revealnav-r2r-train-net-advantage-selection/1",
        "status": "SEALED_R2R_TRAIN_NET_ADVANTAGE_SELECTION",
        "cohort": cohort,
        "split": "train",
        "selection_rule": (
            "one deterministic instruction per trajectory; scene-balanced routes "
            "with at least five reference points" if cohort == "pilot" else
            "one deterministic instruction per all 3603 R2R train trajectories"
        ),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATASET),
        "available_episodes": 10819,
        "available_trajectories": len(routes),
        "selected_episodes": len(selected),
        "selected_scenes": len({row["scene_id"] for row in selected}),
        "selection": selected,
        "task_metrics_used_for_selection": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    if paths["selection"].exists() and json.loads(paths["selection"].read_text()) != value:
        raise RuntimeError("sealed train selection drift")
    if not paths["selection"].exists():
        atomic_json(paths["selection"], value)
    return value


def valid_summary(path: Path, expected: dict) -> tuple[bool, int]:
    if not path.is_file():
        return False, 0
    value = json.loads(path.read_text())
    valid = (
        value.get("status") == "PASS"
        and value.get("split") == "train"
        and value.get("episode_id") == expected["episode_id"]
        and value.get("trajectory_id") == expected["trajectory_id"]
        and value.get("scene_id") == expected["scene_id"]
        and value.get("task_metric_payload_read") is False
        and value.get("ground_truth_payload_read") is False
        and value.get("native_action_overridden") is False
        and value.get("unseen_or_test_read") is False
    )
    return valid, int(value.get("feature_event_count", 0)) if valid else 0


def move_interrupted(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    destination = run_dir.parents[1] / "interrupted" / (
        run_dir.name + f"_{int(time.time())}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(run_dir, destination)


def run_one(row: dict, gpu: int, run_dir: Path) -> dict:
    move_interrupted(run_dir)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    command = [
        str(PYTHON), str(WORKER), "--episode-id", row["episode_id"],
        "--run-dir", str(run_dir),
    ]
    started = time.monotonic()
    process = subprocess.run(
        command, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log = run_dir / "worker.log"
    if run_dir.is_dir():
        log.write_text(process.stdout)
    valid, events = valid_summary(run_dir / "RUN_SUMMARY.json", row)
    return {
        "episode_id": row["episode_id"], "gpu": gpu,
        "rc": process.returncode, "valid": valid, "events": events,
        "wall_time_s": round(time.monotonic() - started, 3),
        "error_tail": "\n".join(process.stdout.splitlines()[-12:]) if not valid else None,
    }


def progress_value(cohort: str, selected: list[dict], active: dict, failures: list[dict]) -> dict:
    paths = layout(cohort)
    completed = 0
    events = 0
    zero_event = 0
    for row in selected:
        valid, count = valid_summary(paths["runs"] / f"ep_{row['episode_id']}" / "RUN_SUMMARY.json", row)
        if valid:
            completed += 1
            events += count
            zero_event += int(count == 0)
    return {
        "schema_version": "revealnav-r2r-train-net-advantage-progress/1",
        "status": "COMPLETE" if completed == len(selected) and not failures else "RUNNING",
        "cohort": cohort,
        "selected": len(selected), "completed": completed,
        "remaining": len(selected) - completed,
        "feature_events": events, "zero_event_episodes": zero_event,
        "active": active, "failures": failures,
        "updated_unix": time.time(),
    }


def collect(cohort: str, gpus: tuple[int, ...]) -> dict:
    selection = prepare(cohort)
    selected = selection["selection"]
    paths = layout(cohort)
    paths["runs"].mkdir(parents=True, exist_ok=True)
    pending = []
    for row in selected:
        valid, _ = valid_summary(paths["runs"] / f"ep_{row['episode_id']}" / "RUN_SUMMARY.json", row)
        if not valid:
            pending.append(row)
    active = {}
    failures = []
    atomic_json(paths["progress"], progress_value(cohort, selected, active, failures))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        running = {}
        iterator = iter(pending)
        for slot, gpu in enumerate(gpus):
            try:
                row = next(iterator)
            except StopIteration:
                break
            run_dir = paths["runs"] / f"ep_{row['episode_id']}"
            future = executor.submit(run_one, row, gpu, run_dir)
            running[future] = (slot, gpu, row)
            active[str(slot)] = {"gpu": gpu, "episode_id": row["episode_id"]}
        while running:
            done, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                slot, gpu, row = running.pop(future)
                result = future.result()
                if not result["valid"]:
                    failures.append(result)
                active.pop(str(slot), None)
                try:
                    next_row = next(iterator)
                except StopIteration:
                    pass
                else:
                    run_dir = paths["runs"] / f"ep_{next_row['episode_id']}"
                    new = executor.submit(run_one, next_row, gpu, run_dir)
                    running[new] = (slot, gpu, next_row)
                    active[str(slot)] = {
                        "gpu": gpu, "episode_id": next_row["episode_id"]
                    }
            atomic_json(paths["progress"], progress_value(cohort, selected, active, failures))
    value = progress_value(cohort, selected, active, failures)
    atomic_json(paths["progress"], value)
    if failures or value["completed"] != value["selected"]:
        raise RuntimeError(f"feature collection incomplete: {len(failures)} failures")
    return value


def call(command: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT)})
    process = subprocess.run(
        command, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(process.stdout)
    if process.returncode:
        raise RuntimeError(f"stage failed rc={process.returncode}: {process.stdout[-2000:]}")


def label(cohort: str) -> None:
    paths = layout(cohort)
    call([
        str(PYTHON), str(LABELER), "--runs", str(paths["runs"]),
        "--output-dir", str(paths["labels"]),
    ], paths["labels"] / "labeler.log")


def train(cohort: str) -> None:
    paths = layout(cohort)
    manifest = paths["labels"] / "R2R_TRAIN_NET_ADVANTAGE_MANIFEST.json"
    call([
        str(PYTHON), str(TRAINER), "--manifest", str(manifest),
        "--output-dir", str(paths["training"]), "--device", "cuda:0",
    ], paths["training"] / "training.log")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "collect", "label", "train", "all"))
    parser.add_argument("--cohort", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise SystemExit("--gpus must contain non-negative GPU slot indices")
    if args.command in ("prepare", "all"):
        value = prepare(args.cohort)
        print(json.dumps({
            "stage": "prepare", "episodes": value["selected_episodes"],
            "scenes": value["selected_scenes"],
        }))
    if args.command in ("collect", "all"):
        value = collect(args.cohort, gpus)
        print(json.dumps({"stage": "collect", **value}, sort_keys=True))
    if args.command in ("label", "all"):
        label(args.cohort)
        print(json.dumps({"stage": "label", "status": "PASS"}))
    if args.command in ("train", "all"):
        train(args.cohort)
        print(json.dumps({"stage": "train", "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
