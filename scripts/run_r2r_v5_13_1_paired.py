#!/usr/bin/env python3
"""Prepare, run, resume, and aggregate the sealed V5.13.1 R2R matrix."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_r2r_v5_13_paired import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_TRAINING_RESULT,
    evaluate,
    sha256_file,
    validate_training_result,
    write_tables,
)
WORKER = ROOT / "scripts/r2r_v5_13_group_worker.py"
BASE = ROOT / "artifacts/evaluation/mf2_r2r_v5_13_1_net_advantage"
PYTHON = ROOT / ".envs/etpr1/bin/python"
NET_GROUPS = frozenset((
    "net_advantage_only", "v5_6_net_advantage",
    "v5_6_net_advantage_no_return",
))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def paths(split: str) -> dict[str, Path]:
    root = BASE / split
    return {
        "root": root,
        "selection": root / "R2R_V5_13_1_SELECTION.json",
        "runs": root / "runs",
        "logs": root / "logs",
        "attempts": root / "JOB_ATTEMPTS.json",
        "status": root / "RUN_STATUS.json",
        "result": root / "R2R_V5_13_1_PAIRED_RESULT.json",
        "tables": root / "tables",
        "pid": root / "ORCHESTRATOR.pid",
        "orchestrator_log": root / "ORCHESTRATOR.log",
    }


def dataset_path(split: str) -> Path:
    return ROOT / (
        "third_party/ETP-R1/data/datasets/"
        f"R2R_VLNCE_v1-3_preprocessed_xlmr/{split}/{split}.json.gz"
    )


def load_protocol(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "SEALED_V5_13_1_BEFORE_FULL_TRAINING_AND_UNSEEN_EVALUATION":
        raise RuntimeError("V5.13.1 protocol is not sealed")
    for relative, expected in value["sources"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"sealed source drift: {relative}")
    return value


def prepare(split: str, protocol_path: Path) -> dict:
    protocol = load_protocol(protocol_path)
    dataset = dataset_path(split)
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    selection = [{
        "episode_id": str(row["episode_id"]),
        "trajectory_id": str(row.get("trajectory_id", "")),
        "scene_id": Path(row["scene_id"]).parts[-2],
    } for row in episodes]
    if len({row["episode_id"] for row in selection}) != len(selection):
        raise RuntimeError("evaluation split contains duplicate episode ids")
    value = {
        "schema_version": "revealnav-r2r-v5.13.1-selection/1",
        "status": "SEALED_COMPLETE_R2R_VALIDATION_SPLIT",
        "split": split,
        "selection_rule": "all episodes in the authorized validation split",
        "episodes": len(selection),
        "scenes": len({row["scene_id"] for row in selection}),
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": sha256_file(dataset),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_schema": protocol["schema_version"],
        "selection": selection,
        "metric_used_for_selection": False,
        "test_or_challenge_read": False,
    }
    target = paths(split)["selection"]
    if target.exists() and json.loads(target.read_text()) != value:
        raise RuntimeError("sealed complete-split selection drift")
    if not target.exists():
        atomic_json(target, value)
    return value


def checkpoints(training: dict) -> dict[int, dict]:
    return {int(row["seed"]): row["checkpoint"] for row in training["results"]}


def job_matrix(protocol: dict, selection: list[dict]) -> list[dict]:
    jobs = []
    for group in protocol["groups"]:
        for seed in group["seeds"]:
            for episode in selection:
                jobs.append({
                    "group": group["id"], "seed": int(seed),
                    "episode_id": episode["episode_id"],
                })
    return jobs


def job_name(row: dict) -> str:
    return f"seed_{row['seed']}_ep_{row['episode_id']}"


def valid_summary(path: Path, row: dict, checkpoint: dict | None) -> bool:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    valid = (
        value.get("status") == "PASS"
        and value.get("group") == row["group"]
        and value.get("seed") == row["seed"]
        and str(value.get("episode_id")) == row["episode_id"]
        and value.get("metrics") is not None
    )
    if checkpoint is not None:
        valid = valid and (
            value.get("net_advantage_checkpoint", {}).get("sha256")
            == checkpoint["sha256"]
        )
    return valid


def execute(
    split: str, gpus: tuple[int, ...], resume: bool,
    protocol_path: Path, training_result_path: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    training = validate_training_result(training_result_path)
    locked_checkpoints = checkpoints(training)
    layout = paths(split)
    selection = json.loads(layout["selection"].read_text())
    if (
        selection.get("status") != "SEALED_COMPLETE_R2R_VALIDATION_SPLIT"
        or selection.get("split") != split
        or selection.get("protocol") != str(protocol_path.relative_to(ROOT))
        or selection.get("test_or_challenge_read") is not False
    ):
        raise RuntimeError("evaluation selection split drift")
    selected_dataset = (ROOT / selection["dataset"]).resolve()
    if (
        selected_dataset != dataset_path(split).resolve()
        or selected_dataset.is_symlink() or not selected_dataset.is_file()
        or sha256_file(selected_dataset) != selection["dataset_sha256"]
        or selection.get("episodes") != len(selection.get("selection", []))
        or selection.get("metric_used_for_selection") is not False
    ):
        raise RuntimeError("sealed evaluation selection provenance drift")
    jobs = job_matrix(protocol, selection["selection"])
    if layout["runs"].exists() and not resume:
        raise RuntimeError("evaluation runs exist; use resume")
    layout["runs"].mkdir(parents=True, exist_ok=True)
    layout["logs"].mkdir(parents=True, exist_ok=True)
    attempts = (
        json.loads(layout["attempts"].read_text())
        if layout["attempts"].is_file() else {}
    )
    completed = []
    queue = []
    for row in jobs:
        checkpoint = locked_checkpoints.get(row["seed"]) if row["group"] in NET_GROUPS else None
        run_dir = layout["runs"] / row["group"] / job_name(row)
        if resume and valid_summary(run_dir / "RUN_SUMMARY.json", row, checkpoint):
            completed.append({**row, "returncode": 0, "recovered": True})
            continue
        if run_dir.exists():
            name = f"{row['group']}:{job_name(row)}"
            interrupted = (
                layout["root"] / "interrupted" / row["group"]
                / f"{job_name(row)}_attempt_{len(attempts.get(name, []))}"
            )
            interrupted.parent.mkdir(parents=True, exist_ok=True)
            os.replace(run_dir, interrupted)
        queue.append(row)

    free = list(enumerate(gpus))
    active = []
    atomic_json(layout["status"], {
        "status": "RUNNING", "split": split, "completed": len(completed),
        "expected": len(jobs), "remaining": len(queue), "slots": len(gpus),
        "active": [], "failures": [],
    })
    while queue or active:
        while queue and free:
            slot, gpu = free.pop(0)
            row = queue.pop(0)
            name = f"{row['group']}:{job_name(row)}"
            attempt_rows = attempts.setdefault(name, [])
            attempt_rows.append({
                "attempt": len(attempt_rows) + 1, "gpu": gpu,
                "started_unix": time.time(), "status": "RUNNING",
            })
            checkpoint = locked_checkpoints.get(row["seed"]) if row["group"] in NET_GROUPS else None
            run_dir = layout["runs"] / row["group"] / job_name(row)
            command = [
                str(PYTHON), str(WORKER), "--group", row["group"],
                "--episode-id", row["episode_id"], "--seed", str(row["seed"]),
                "--split", split, "--run-dir", str(run_dir),
            ]
            if checkpoint is not None:
                command.extend([
                    "--net-advantage-checkpoint", str(ROOT / checkpoint["path"]),
                    "--net-advantage-sha256", checkpoint["sha256"],
                ])
            stdout = (layout["logs"] / f"{name.replace(':', '__')}.stdout.log").open("a")
            stderr = (layout["logs"] / f"{name.replace(':', '__')}.stderr.log").open("a")
            process = subprocess.Popen(
                command, cwd=ROOT, env={
                    **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                }, stdout=stdout, stderr=stderr,
            )
            active.append({
                "slot": slot, "gpu": gpu, "row": row, "name": name,
                "process": process, "streams": (stdout, stderr),
            })
            atomic_json(layout["attempts"], attempts)
        time.sleep(0.5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            attempts[item["name"]][-1].update({
                "status": "PASS" if code == 0 else "FAIL",
                "returncode": code, "finished_unix": time.time(),
            })
            completed.append({**item["row"], "returncode": code})
            free.append((item["slot"], item["gpu"]))
            free.sort()
            active.remove(item)
            atomic_json(layout["attempts"], attempts)
            atomic_json(layout["status"], {
                "status": "RUNNING" if queue or active else "COMPLETE",
                "split": split, "completed": len(completed),
                "expected": len(jobs), "remaining": len(queue) + len(active),
                "slots": len(gpus),
                "active": [{
                    "group": row["row"]["group"], "seed": row["row"]["seed"],
                    "episode_id": row["row"]["episode_id"], "gpu": row["gpu"],
                } for row in active],
                "failures": [row for row in completed if row["returncode"]],
            })
            print(json.dumps(completed[-1], sort_keys=True), flush=True)
    atomic_json(layout["status"], {
        "status": "COMPLETE", "split": split, "completed": len(completed),
        "expected": len(jobs), "remaining": 0, "slots": len(gpus),
        "active": [],
        "failures": [row for row in completed if row["returncode"]],
    })
    if any(row["returncode"] for row in completed):
        raise RuntimeError("one or more V5.13.1 workers failed; inspect and resume")


def aggregate(
    split: str, protocol_path: Path, training_result_path: Path,
) -> dict:
    layout = paths(split)
    value = evaluate(
        protocol_path, training_result_path, layout["runs"], 10000
    )
    atomic_json(layout["result"], value)
    write_tables(value, layout["tables"])
    return value


def pid_alive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        os.kill(int(path.read_text()), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def launch(
    split: str, gpus: str, protocol_path: Path, training_result_path: Path,
) -> int:
    validate_training_result(training_result_path)
    prepare(split, protocol_path)
    layout = paths(split)
    if pid_alive(layout["pid"]):
        raise RuntimeError("V5.13.1 orchestrator is already running")
    command = [
        str(PYTHON), str(Path(__file__).resolve()),
        "all",
        "--split", split, "--gpus", gpus,
        "--protocol", str(protocol_path),
        "--training-result", str(training_result_path),
    ]
    layout["root"].mkdir(parents=True, exist_ok=True)
    log = layout["orchestrator_log"].open("a")
    process = subprocess.Popen(
        command, cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    layout["pid"].write_text(f"{process.pid}\n")
    return process.pid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=(
            "prepare", "run", "resume", "verify", "launch", "all"
        )
    )
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--training-result", type=Path, default=DEFAULT_TRAINING_RESULT
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    training_result_path = args.training_result.resolve()
    if any(
        ROOT not in path.parents for path in (protocol_path, training_result_path)
    ):
        raise SystemExit("protocol and training result must remain inside the project")
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or any(value < 0 for value in gpus):
        raise SystemExit("--gpus must contain non-negative GPU indices")
    if args.command == "prepare":
        if args.split == "val_unseen":
            validate_training_result(training_result_path)
        value = prepare(args.split, protocol_path)
        print(json.dumps({
            "status": value["status"], "episodes": value["episodes"],
            "scenes": value["scenes"], "split": args.split,
        }, sort_keys=True))
    elif args.command in ("run", "resume"):
        execute(
            args.split, gpus, args.command == "resume",
            protocol_path, training_result_path,
        )
    elif args.command == "verify":
        value = aggregate(args.split, protocol_path, training_result_path)
        print(json.dumps({
            "status": value["status"], "main_gates": value["main_gates"],
        }, sort_keys=True))
        return 0 if value["status"] == "PASS" else 2
    elif args.command == "launch":
        pid = launch(
            args.split, args.gpus, protocol_path, training_result_path
        )
        print(json.dumps({
            "status": "LAUNCHED", "pid": pid,
            "monitor": "scripts/monitor_r2r_v5_13_1_paired.py",
        }, sort_keys=True))
    else:
        validate_training_result(training_result_path)
        prepare(args.split, protocol_path)
        execute(
            args.split, gpus, paths(args.split)["runs"].exists(),
            protocol_path, training_result_path,
        )
        value = aggregate(args.split, protocol_path, training_result_path)
        print(json.dumps({
            "status": value["status"], "main_gates": value["main_gates"],
        }, sort_keys=True))
        return 0 if value["status"] == "PASS" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
