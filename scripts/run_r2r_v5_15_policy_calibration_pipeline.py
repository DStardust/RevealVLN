#!/usr/bin/env python3
"""Resumable R2R-train V5.6 proposal collection and causal labeling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
WORKER = ROOT / "scripts/r2r_train_v5_15_policy_proposal_worker.py"
BUILDER = ROOT / "scripts/build_r2r_train_net_advantage_labels.py"
CALIBRATOR = ROOT / "scripts/calibrate_r2r_v5_15_policy_threshold.py"
SOURCE_SELECTION = ROOT / (
    "artifacts/phase1/r2r_train_net_advantage/full/"
    "R2R_TRAIN_NET_ADVANTAGE_SELECTION.json"
)
OUT = ROOT / "artifacts/phase1/r2r_train_policy_calibration_v5_15"
RUNS = OUT / "runs"
LABELS = OUT / "labels"
CALIBRATION = OUT / "calibration"
SELECTION = OUT / "R2R_TRAIN_V5_15_POLICY_SELECTION.json"
PROGRESS = OUT / "R2R_TRAIN_V5_15_POLICY_PROGRESS.json"
ATTEMPTS = OUT / "R2R_TRAIN_V5_15_POLICY_ATTEMPTS.json"
PID = OUT / "R2R_TRAIN_V5_15_POLICY.pid"
LOG = OUT / "R2R_TRAIN_V5_15_POLICY.log"
SEEDS = (20260826, 20260827, 20260828)
DEFAULT_GPUS = "0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,3,4,5,6,7"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def process_alive(value: int | None) -> bool:
    if value is None:
        return False
    try:
        state = (Path("/proc") / str(value) / "stat").read_text().split()[2]
        return state != "Z"
    except (OSError, IndexError):
        return False


def read_pid() -> int | None:
    try:
        return int(PID.read_text())
    except (OSError, ValueError):
        return None


def prepare() -> dict:
    source = load(SOURCE_SELECTION)
    if (
        source.get("status") != "SEALED_R2R_TRAIN_NET_ADVANTAGE_SELECTION"
        or source.get("cohort") != "full"
        or source.get("split") != "train"
        or source.get("unseen_or_test_read") is not False
        or len(source.get("selection", [])) != 3603
    ):
        raise RuntimeError("full R2R-train source selection is invalid")
    value = {
        "schema_version": "revealnav-r2r-v5.15-policy-selection/1",
        "status": "SEALED_R2R_TRAIN_V5_15_POLICY_SELECTION",
        "split": "train",
        "source_selection": str(SOURCE_SELECTION.relative_to(ROOT)),
        "source_selection_sha256": sha256_file(SOURCE_SELECTION),
        "selection_rule": (
            "all 3603 deterministic R2R-train trajectories crossed with the "
            "three already-locked V5.6 controller seeds"
        ),
        "episodes": source["selection"],
        "unique_episodes": len(source["selection"]),
        "controller_seeds": list(SEEDS),
        "selected_runs": len(source["selection"]) * len(SEEDS),
        "policy_mode": "shadow; native ETP-R1 action is never overridden",
        "task_metrics_used_for_selection": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    if SELECTION.is_file() and load(SELECTION) != value:
        raise RuntimeError("sealed V5.15 policy selection drift")
    if not SELECTION.is_file():
        atomic_json(SELECTION, value)
    return value


def rows(selection: dict) -> list[dict]:
    return [
        {**episode, "controller_seed": seed}
        for seed in SEEDS for episode in selection["episodes"]
    ]


def run_dir(row: dict) -> Path:
    return RUNS / f"seed_{row['controller_seed']}" / f"ep_{row['episode_id']}"


def valid_summary(row: dict) -> tuple[bool, dict]:
    value = load(run_dir(row) / "RUN_SUMMARY.json")
    valid = (
        value.get("status") == "PASS"
        and value.get("schema_version")
        == "revealnav-r2r-v5.15-policy-proposal-worker/1"
        and value.get("split") == "train"
        and value.get("episode_id") == row["episode_id"]
        and value.get("trajectory_id") == row["trajectory_id"]
        and value.get("scene_id") == row["scene_id"]
        and value.get("controller_seed") == row["controller_seed"]
        and value.get("source_policy") == "V5.6 shadow proposals"
        and value.get("task_metric_payload_read") is False
        and value.get("ground_truth_payload_read") is False
        and value.get("native_action_overridden") is False
        and value.get("unseen_or_test_read") is False
    )
    return valid, value if valid else {}


def move_interrupted(path: Path, attempt: int) -> None:
    if not path.exists():
        return
    destination = (
        OUT / "interrupted" / path.parent.name
        / f"{path.name}_attempt_{attempt}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError("interrupted-run destination collision")
    os.replace(path, destination)


def totals(selection_rows: list[dict]) -> tuple[int, int, int, dict[int, int]]:
    completed = events = missing = 0
    by_seed = {seed: 0 for seed in SEEDS}
    for row in selection_rows:
        valid, summary = valid_summary(row)
        if valid:
            completed += 1
            events += int(summary.get("feature_event_count", 0))
            missing += int(summary.get("missing_causal_inputs", 0))
            by_seed[row["controller_seed"]] += int(
                summary.get("feature_event_count", 0)
            )
    return completed, events, missing, by_seed


def write_progress(
    status: str, selected: int, completed: int, events: int, missing: int,
    by_seed: dict[int, int], active: list[dict], exhausted: list[dict],
    stage: str = "collection",
) -> None:
    atomic_json(PROGRESS, {
        "schema_version": "revealnav-r2r-v5.15-policy-progress/1",
        "status": status, "stage": stage, "selected": selected,
        "completed": completed, "remaining": selected - completed,
        "proposal_events": events, "missing_causal_inputs": missing,
        "proposal_events_by_seed": {str(k): v for k, v in by_seed.items()},
        "active": active, "exhausted_failures": exhausted,
        "updated_unix": time.time(), "unseen_or_test_read": False,
    })


def collect(selection_rows: list[dict], gpus: tuple[int, ...]) -> None:
    attempts = load(ATTEMPTS)
    completed, events, missing, by_seed = totals(selection_rows)
    queue = [row for row in selection_rows if not valid_summary(row)[0]]
    free = list(enumerate(gpus))
    active: list[dict] = []
    exhausted: list[dict] = []
    write_progress(
        "RUNNING", len(selection_rows), completed, events, missing,
        by_seed, [], exhausted,
    )
    while queue or active:
        while queue and free:
            slot, gpu = free.pop(0)
            row = queue.pop(0)
            key = f"seed_{row['controller_seed']}_ep_{row['episode_id']}"
            history = attempts.setdefault(key, [])
            attempt = len(history) + 1
            path = run_dir(row)
            move_interrupted(path, attempt)
            path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path = OUT / "logs" / f"{key}.stdout.log"
            stderr_path = OUT / "logs" / f"{key}.stderr.log"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout = stdout_path.open("a")
            stderr = stderr_path.open("a")
            command = [
                str(PYTHON), str(WORKER), "--episode-id", row["episode_id"],
                "--seed", str(row["controller_seed"]), "--run-dir", str(path),
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, env={
                    **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                }, stdout=stdout, stderr=stderr,
            )
            history.append({
                "attempt": attempt, "gpu": gpu, "status": "RUNNING",
                "started_unix": time.time(),
            })
            active.append({
                "slot": slot, "gpu": gpu, "row": row, "key": key,
                "process": process, "streams": (stdout, stderr),
            })
            atomic_json(ATTEMPTS, attempts)
        time.sleep(0.5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            key = item["key"]
            row = item["row"]
            valid, summary = valid_summary(row)
            passed = code == 0 and valid
            attempts[key][-1].update({
                "status": "PASS" if passed else "FAIL",
                "returncode": code, "finished_unix": time.time(),
            })
            if passed:
                completed += 1
                count = int(summary.get("feature_event_count", 0))
                events += count
                missing += int(summary.get("missing_causal_inputs", 0))
                by_seed[row["controller_seed"]] += count
            elif len(attempts[key]) < 5:
                queue.append(row)
            else:
                exhausted.append({
                    "episode_id": row["episode_id"],
                    "controller_seed": row["controller_seed"],
                    "returncode": code, "attempts": len(attempts[key]),
                })
            free.append((item["slot"], item["gpu"]))
            free.sort()
            active.remove(item)
            atomic_json(ATTEMPTS, attempts)
            write_progress(
                "RUNNING" if queue or active else "COLLECTION_COMPLETE",
                len(selection_rows), completed, events, missing, by_seed,
                [{
                    "episode_id": row_["row"]["episode_id"],
                    "controller_seed": row_["row"]["controller_seed"],
                    "gpu": row_["gpu"],
                } for row_ in active], exhausted,
            )
    if exhausted or completed != len(selection_rows):
        raise RuntimeError("V5.15 policy collection exhausted one or more jobs")


def build_labels() -> dict:
    selection_rows = rows(load(SELECTION))
    completed, events, missing, by_seed = totals(selection_rows)
    write_progress(
        "RUNNING", len(selection_rows), completed, events, missing, by_seed,
        [], [], stage="causal_label_build",
    )
    process = subprocess.run([
        str(PYTHON), str(BUILDER), "--runs", str(RUNS),
        "--output-dir", str(LABELS), "--summary-pattern",
        "seed_*/ep_*/RUN_SUMMARY.json",
    ], cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"}, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with LOG.open("a") as stream:
        stream.write(process.stdout)
    if process.returncode:
        raise RuntimeError("V5.15 causal label build failed")
    manifest = load(LABELS / "R2R_TRAIN_NET_ADVANTAGE_MANIFEST.json")
    if (
        manifest.get("status")
        != "R2R_TRAIN_POLICY_INDUCED_NET_ADVANTAGE_DATASET_READY"
        or manifest.get("unseen_or_test_read") is not False
    ):
        raise RuntimeError("V5.15 policy-induced manifest gate failed")
    write_progress(
        "DATASET_READY", len(selection_rows), completed, events, missing,
        by_seed, [], [], stage="policy_calibration",
    )
    return manifest


def calibrate(manifest: dict) -> dict:
    selection_rows = rows(load(SELECTION))
    completed, events, missing, by_seed = totals(selection_rows)
    write_progress(
        "RUNNING", len(selection_rows), completed, events, missing, by_seed,
        [], [], stage="policy_calibration",
    )
    manifest_path = LABELS / "R2R_TRAIN_NET_ADVANTAGE_MANIFEST.json"
    process = subprocess.run([
        str(PYTHON), str(CALIBRATOR), "--manifest", str(manifest_path),
        "--output-dir", str(CALIBRATION), "--device", "cuda:0",
    ], cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"}, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with LOG.open("a") as stream:
        stream.write(process.stdout)
    result = load(CALIBRATION / "R2R_V5_15_POLICY_CALIBRATION_RESULT.json")
    if result.get("status") not in (
        "R2R_V5_15_POLICY_CALIBRATION_PASS",
        "R2R_V5_15_POLICY_CALIBRATION_FAIL",
    ):
        raise RuntimeError("V5.15 policy calibration wrote no valid result")
    if (
        result.get("unseen_or_test_read") is not False
        or result.get("task_metric_payload_read") is not False
        or result.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("V5.15 policy calibration boundary drift")
    passed = result["status"].endswith("_PASS")
    write_progress(
        "CALIBRATION_PASS" if passed else "CALIBRATION_FAIL",
        len(selection_rows), completed, events, missing, by_seed, [], [],
        stage="complete",
    )
    return result


def run(gpus: tuple[int, ...]) -> int:
    selection = prepare()
    selection_rows = rows(selection)
    collect(selection_rows, gpus)
    manifest = build_labels()
    calibration = calibrate(manifest)
    print(json.dumps({
        "status": manifest["status"], "completed_runs": manifest["completed_runs"],
        "policy_rows": manifest["training_rows"],
        "calibration_status": calibration["status"],
    }, sort_keys=True))
    return 0 if calibration["status"].endswith("_PASS") else 2


def launch(gpus: str) -> int:
    current = read_pid()
    if process_alive(current):
        raise RuntimeError(f"V5.15 pipeline already running as PID {current}")
    OUT.mkdir(parents=True, exist_ok=True)
    stream = LOG.open("a")
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__).resolve()), "run", "--gpus", gpus],
        cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
    )
    stream.close()
    PID.write_text(f"{process.pid}\n")
    return process.pid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "launch", "monitor"))
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or any(value < 0 for value in gpus):
        raise SystemExit("--gpus must contain non-negative GPU indices")
    if args.command == "prepare":
        value = prepare()
        print(json.dumps({
            "status": value["status"], "selected_runs": value["selected_runs"],
        }, sort_keys=True))
        return 0
    if args.command == "run":
        return run(gpus)
    if args.command == "launch":
        print(json.dumps({
            "status": "LAUNCHED", "pid": launch(args.gpus),
            "monitor": (
                "scripts/run_r2r_v5_15_policy_calibration_pipeline.py monitor"
            ),
        }, sort_keys=True))
        return 0
    current = read_pid()
    print(json.dumps({
        "pid": current, "alive": process_alive(current),
        "progress": load(PROGRESS), "log": str(LOG.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
