#!/usr/bin/env python3
"""Run the V5.15 policy-calibrated R2R val-seen paired evaluation."""

from __future__ import annotations

import argparse
import hashlib
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
    load_group,
    paired_comparison,
)


PYTHON = ROOT / ".envs/etpr1/bin/python"
WORKER = ROOT / "scripts/r2r_v5_13_group_worker.py"
CALIBRATION = ROOT / (
    "artifacts/phase1/r2r_train_policy_calibration_v5_15/calibration/"
    "R2R_V5_15_POLICY_CALIBRATION_RESULT.json"
)
CALIBRATION_PROTOCOL = ROOT / (
    "artifacts/design/R2R_V5_15_POLICY_CALIBRATION_PROTOCOL.json"
)
V514_ROOT = ROOT / "artifacts/evaluation/mf2_r2r_v5_14_net_advantage/val_seen"
V514_RESULT = V514_ROOT / "R2R_V5_13_1_PAIRED_RESULT.json"
BASE = ROOT / "artifacts/evaluation/mf2_r2r_v5_15_policy_calibrated"
PROTOCOL = BASE / "R2R_V5_15_PAIRED_PROTOCOL.json"
ROOT_OUT = BASE / "val_seen"
RUNS = ROOT_OUT / "runs"
LOGS = ROOT_OUT / "logs"
ATTEMPTS = ROOT_OUT / "JOB_ATTEMPTS.json"
STATUS = ROOT_OUT / "RUN_STATUS.json"
RESULT = ROOT_OUT / "R2R_V5_15_PAIRED_RESULT.json"
SELECTION = ROOT_OUT / "R2R_V5_15_SELECTION.json"
PID = ROOT_OUT / "ORCHESTRATOR.pid"
ORCHESTRATOR_LOG = ROOT_OUT / "ORCHESTRATOR.log"
TREATMENTS = (
    "net_advantage_only",
    "v5_6_net_advantage",
    "v5_6_net_advantage_no_return",
)
SEEDS = (20260826, 20260827, 20260828)
METRICS = (
    "success", "oracle_success", "spl", "ndtw", "sdtw",
    "distance_to_goal", "path_length", "steps_taken", "collisions",
)
DEFAULT_GPUS = "0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7"


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


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def read_pid() -> int | None:
    try:
        return int(PID.read_text())
    except (OSError, ValueError):
        return None


def validate_calibration() -> tuple[dict, dict]:
    value = load(CALIBRATION)
    protocol = load(CALIBRATION_PROTOCOL)
    if (
        protocol.get("status")
        != "SEALED_V5_15_BEFORE_POLICY_INDUCED_COLLECTION"
        or any(
            not (ROOT / relative).is_file()
            or sha256_file(ROOT / relative) != expected
            for relative, expected in protocol.get("sources", {}).items()
        )
        or value.get("status") != "R2R_V5_15_POLICY_CALIBRATION_PASS"
        or value.get("unseen_or_test_read") is not False
        or value.get("task_metric_payload_read") is not False
        or not value.get("gates")
        or not all(value["gates"].values())
    ):
        raise RuntimeError("V5.15 train-only policy calibration gate did not pass")
    checkpoint = value.get("checkpoint", {})
    checkpoint_path = (ROOT / checkpoint.get("path", "")).resolve()
    if (
        checkpoint.get("member_seeds") != list(SEEDS)
        or ROOT not in checkpoint_path.parents
        or checkpoint_path.is_symlink()
        or not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size != checkpoint.get("bytes")
        or sha256_file(checkpoint_path) != checkpoint.get("sha256")
    ):
        raise RuntimeError("V5.15 calibrated checkpoint provenance drift")
    frozen = Path(value["frozen_ensemble_result"])
    frozen = frozen if frozen.is_absolute() else ROOT / frozen
    if (
        ROOT not in frozen.resolve().parents
        or not frozen.is_file()
        or sha256_file(frozen) != value["frozen_ensemble_result_sha256"]
    ):
        raise RuntimeError("V5.15 frozen ensemble provenance drift")
    return value, checkpoint


def seal_protocol(calibration: dict, checkpoint: dict) -> dict:
    value = {
        "schema_version": "revealnav-r2r-v5.15-paired-protocol/1",
        "status": "SEALED_V5_15_AFTER_TRAIN_ONLY_GATE_BEFORE_VAL_SEEN_RERUN",
        "method_revision": (
            "V5.14 ensemble weights and V5.6 proposal policy unchanged; only "
            "the online threshold is recalibrated on R2R-train policy-induced rows"
        ),
        "split": "complete R2R val_seen; development evidence only",
        "paper_result": False,
        "val_unseen_or_test_opened": False,
        "calibration_result": str(CALIBRATION.relative_to(ROOT)),
        "calibration_result_sha256": sha256_file(CALIBRATION),
        "calibration_protocol": str(CALIBRATION_PROTOCOL.relative_to(ROOT)),
        "calibration_protocol_sha256": sha256_file(CALIBRATION_PROTOCOL),
        "checkpoint": checkpoint,
        "controls": {
            "reuse": ["etp_r1", "v5_6"],
            "reason": "their code, checkpoints, actions, and metrics are unchanged",
            "source_result": str(V514_RESULT.relative_to(ROOT)),
            "source_result_sha256": sha256_file(V514_RESULT),
        },
        "rerun_groups": list(TREATMENTS),
        "seeds": list(SEEDS),
        "metrics": list(METRICS),
        "bootstrap_replicates": 10000,
        "primary_gate": {
            "directional": "main minus ETP-R1: SPL>0, nDTW>0, Success>=0",
            "statistical": "paired-bootstrap lower bounds for SPL and nDTW >0",
            "incremental": "main mean SPL exceeds V5.6",
        },
        "sources": {
            str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(
                Path(__file__).resolve()
            ),
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            "scripts/evaluate_r2r_v5_13_paired.py": sha256_file(
                ROOT / "scripts/evaluate_r2r_v5_13_paired.py"
            ),
            "scripts/r2r_v5_6_net_advantage_controller.py": sha256_file(
                ROOT / "scripts/r2r_v5_6_net_advantage_controller.py"
            ),
            "scripts/revealnav_net_advantage.py": sha256_file(
                ROOT / "scripts/revealnav_net_advantage.py"
            ),
        },
    }
    if PROTOCOL.is_file() and load(PROTOCOL) != value:
        raise RuntimeError("sealed V5.15 paired protocol drift")
    if not PROTOCOL.is_file():
        atomic_json(PROTOCOL, value)
    return value


def prepare() -> list[dict]:
    calibration, checkpoint = validate_calibration()
    protocol = seal_protocol(calibration, checkpoint)
    source = load(V514_ROOT / "R2R_V5_13_1_SELECTION.json")
    dataset = (ROOT / source.get("dataset", "")).resolve()
    if (
        source.get("status") != "SEALED_COMPLETE_R2R_VALIDATION_SPLIT"
        or source.get("split") != "val_seen"
        or source.get("episodes") != 778
        or source.get("test_or_challenge_read") is not False
        or ROOT not in dataset.parents
        or dataset.is_symlink()
        or not dataset.is_file()
        or sha256_file(dataset) != source.get("dataset_sha256")
        or sha256_file(V514_RESULT) != protocol["controls"]["source_result_sha256"]
    ):
        raise RuntimeError("V5.14 complete val-seen control provenance drift")
    selection = {
        "schema_version": "revealnav-r2r-v5.15-selection/1",
        "status": "SEALED_COMPLETE_R2R_VAL_SEEN_REUSE",
        "episodes": source["episodes"],
        "scenes": source["scenes"],
        "selection": source["selection"],
        "source_selection": str(
            (V514_ROOT / "R2R_V5_13_1_SELECTION.json").relative_to(ROOT)
        ),
        "source_selection_sha256": sha256_file(
            V514_ROOT / "R2R_V5_13_1_SELECTION.json"
        ),
        "dataset": source["dataset"],
        "dataset_sha256": source["dataset_sha256"],
        "metric_used_for_selection": False,
        "val_unseen_or_test_read": False,
    }
    if SELECTION.is_file() and load(SELECTION) != selection:
        raise RuntimeError("sealed V5.15 val-seen selection drift")
    if not SELECTION.is_file():
        atomic_json(SELECTION, selection)
    return source["selection"]


def jobs(episodes: list[dict]) -> list[dict]:
    return [
        {"group": group, "seed": seed, "episode_id": row["episode_id"]}
        for group in TREATMENTS for seed in SEEDS for row in episodes
    ]


def run_dir(row: dict) -> Path:
    return RUNS / row["group"] / f"seed_{row['seed']}_ep_{row['episode_id']}"


def valid_summary(row: dict, checkpoint: dict) -> bool:
    value = load(run_dir(row) / "RUN_SUMMARY.json")
    return (
        value.get("status") == "PASS"
        and value.get("group") == row["group"]
        and value.get("seed") == row["seed"]
        and str(value.get("episode_id")) == row["episode_id"]
        and value.get("split") == "val_seen"
        and value.get("metrics") is not None
        and value.get("net_advantage_checkpoint", {}).get("sha256")
        == checkpoint["sha256"]
    )


def write_status(
    completed: int, total: int, active: list[dict], failures: list[dict],
    slots: int,
) -> None:
    atomic_json(STATUS, {
        "schema_version": "revealnav-r2r-v5.15-run-status/1",
        "status": "COMPLETE" if completed == total and not active else "RUNNING",
        "completed": completed,
        "expected": total,
        "remaining": total - completed,
        "slots": slots,
        "active": [
            {**item["row"], "gpu": item["gpu"]} for item in active
        ],
        "exhausted_failures": failures,
        "updated_unix": time.time(),
        "val_unseen_or_test_read": False,
    })


def execute(gpus: tuple[int, ...]) -> None:
    episodes = prepare()
    _, checkpoint = validate_calibration()
    matrix = jobs(episodes)
    completed = sum(valid_summary(row, checkpoint) for row in matrix)
    queue = [row for row in matrix if not valid_summary(row, checkpoint)]
    attempts = load(ATTEMPTS)
    free = list(enumerate(gpus))
    active: list[dict] = []
    exhausted: list[dict] = []
    write_status(completed, len(matrix), active, exhausted, len(gpus))
    while queue or active:
        while queue and free:
            slot, gpu = free.pop(0)
            row = queue.pop(0)
            key = f"{row['group']}:seed_{row['seed']}_ep_{row['episode_id']}"
            history = attempts.setdefault(key, [])
            path = run_dir(row)
            if path.exists():
                destination = (
                    ROOT_OUT / "interrupted" / row["group"]
                    / f"seed_{row['seed']}_ep_{row['episode_id']}_attempt_{len(history) + 1}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise RuntimeError("interrupted-run destination collision")
                os.replace(path, destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            LOGS.mkdir(parents=True, exist_ok=True)
            name = key.replace(":", "__")
            stdout = (LOGS / f"{name}.stdout.log").open("a")
            stderr = (LOGS / f"{name}.stderr.log").open("a")
            command = [
                str(PYTHON), str(WORKER), "--group", row["group"],
                "--episode-id", row["episode_id"], "--seed", str(row["seed"]),
                "--split", "val_seen", "--run-dir", str(path),
                "--net-advantage-checkpoint", str(ROOT / checkpoint["path"]),
                "--net-advantage-sha256", checkpoint["sha256"],
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, env={
                    **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                }, stdout=stdout, stderr=stderr,
            )
            history.append({
                "attempt": len(history) + 1, "gpu": gpu,
                "status": "RUNNING", "started_unix": time.time(),
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
            row = item["row"]
            passed = code == 0 and valid_summary(row, checkpoint)
            attempts[item["key"]][-1].update({
                "status": "PASS" if passed else "FAIL",
                "returncode": code, "finished_unix": time.time(),
            })
            if passed:
                completed += 1
            elif len(attempts[item["key"]]) < 5:
                queue.append(row)
            else:
                exhausted.append({
                    **row, "attempts": len(attempts[item["key"]]),
                    "returncode": code,
                })
            free.append((item["slot"], item["gpu"]))
            free.sort()
            active.remove(item)
            atomic_json(ATTEMPTS, attempts)
            write_status(completed, len(matrix), active, exhausted, len(gpus))
    if exhausted or completed != len(matrix):
        raise RuntimeError("V5.15 paired evaluation exhausted one or more jobs")


def mean_metrics(rows: dict) -> dict:
    return {
        metric: sum(float(row["metrics"][metric]) for row in rows.values())
        / len(rows)
        for metric in METRICS
    }


def controller_totals(rows: dict) -> dict:
    keys = (
        "net_advantage_decisions", "net_advantage_approvals",
        "net_advantage_vetoes", "checkpointed_excursions",
        "backtrack_decisions", "successful_returns", "failed_returns",
        "no_return_suppressions",
    )
    return {
        key: sum(int((row.get("controller") or {}).get(key, 0)) for row in rows.values())
        for key in keys
    }


def aggregate() -> dict:
    prepare()
    controls = {
        group: load_group(V514_ROOT / "runs" / group, list(METRICS), group)
        for group in ("etp_r1", "v5_6")
    }
    treatments = {
        group: load_group(RUNS / group, list(METRICS), group)
        for group in TREATMENTS
    }
    groups = {**controls, **treatments}
    episodes = [{key[0] for key in rows} for rows in groups.values()]
    if any(value != episodes[0] for value in episodes[1:]) or len(episodes[0]) != 778:
        raise RuntimeError("V5.15 paired episode coverage differs")
    comparisons = {
        "v5_6_net_advantage_minus_etp_r1": paired_comparison(
            groups["v5_6_net_advantage"], groups["etp_r1"], list(METRICS), 10000
        ),
        "v5_6_net_advantage_minus_v5_6": paired_comparison(
            groups["v5_6_net_advantage"], groups["v5_6"], list(METRICS), 10000
        ),
        "net_advantage_only_minus_etp_r1": paired_comparison(
            groups["net_advantage_only"], groups["etp_r1"], list(METRICS), 10000
        ),
        "v5_6_net_advantage_minus_v5_6_net_advantage_no_return": paired_comparison(
            groups["v5_6_net_advantage"],
            groups["v5_6_net_advantage_no_return"], list(METRICS), 10000,
        ),
    }
    primary = comparisons["v5_6_net_advantage_minus_etp_r1"][
        "benefit_treatment_minus_baseline"
    ]
    incremental = comparisons["v5_6_net_advantage_minus_v5_6"][
        "benefit_treatment_minus_baseline"
    ]
    gates = {
        "all_groups_complete_and_paired": True,
        "primary_directionally_positive": (
            primary["spl"]["mean"] > 0
            and primary["ndtw"]["mean"] > 0
            and primary["success"]["mean"] >= 0
        ),
        "primary_statistically_positive": (
            primary["spl"]["episode_bootstrap_95pct"][0] > 0
            and primary["ndtw"]["episode_bootstrap_95pct"][0] > 0
        ),
        "net_advantage_improves_v5_6_mean_spl": incremental["spl"]["mean"] > 0,
    }
    value = {
        "schema_version": "revealnav-r2r-v5.15-paired-result/1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "split": "val_seen",
        "paper_result": False,
        "main_gates": gates,
        "group_metrics": {group: mean_metrics(rows) for group, rows in groups.items()},
        "controller_totals": {
            group: controller_totals(rows) for group, rows in groups.items()
        },
        "comparisons": comparisons,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "calibration_result": str(CALIBRATION.relative_to(ROOT)),
        "calibration_result_sha256": sha256_file(CALIBRATION),
        "reused_control_groups": ["etp_r1", "v5_6"],
        "reused_control_result": str(V514_RESULT.relative_to(ROOT)),
        "reused_control_result_sha256": sha256_file(V514_RESULT),
        "val_unseen_or_test_read": False,
    }
    atomic_json(RESULT, value)
    return value


def run(gpus: tuple[int, ...]) -> int:
    execute(gpus)
    value = aggregate()
    print(json.dumps({
        "status": value["status"], "main_gates": value["main_gates"],
        "group_metrics": value["group_metrics"],
    }, sort_keys=True))
    return 0 if value["status"] == "PASS" else 2


def launch(gpus: str) -> int:
    current = read_pid()
    if process_alive(current):
        raise RuntimeError(f"V5.15 paired evaluation already runs as PID {current}")
    validate_calibration()
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    stream = ORCHESTRATOR_LOG.open("a")
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__).resolve()), "run", "--gpus", gpus],
        cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
    )
    stream.close()
    PID.write_text(f"{process.pid}\n")
    return process.pid


def monitor() -> dict:
    status = load(STATUS)
    result = load(RESULT)
    attempts = load(ATTEMPTS)
    completed = int(status.get("completed", 0))
    expected = int(status.get("expected", 0))
    passed_attempts = [
        history[-1] for history in attempts.values()
        if history and history[-1].get("status") == "PASS"
    ]
    durations = [
        row["finished_unix"] - row["started_unix"] for row in passed_attempts
        if row.get("finished_unix") is not None
    ]
    slots = max(1, int(status.get("slots", 1)))
    eta = (
        (expected - completed) * sum(durations) / len(durations) / slots
        if durations and expected else None
    )
    return {
        "orchestrator_pid": read_pid(),
        "orchestrator_alive": process_alive(read_pid()),
        "status": status.get("status", "WAITING"),
        "completed": completed,
        "expected": expected or None,
        "progress_percent": round(100 * completed / expected, 2) if expected else 0,
        "active": len(status.get("active", [])),
        "exhausted_failures": status.get("exhausted_failures", []),
        "eta_minutes": round(eta / 60, 1) if eta is not None else None,
        "result_status": result.get("status"),
        "main_gates": result.get("main_gates"),
        "log": str(ORCHESTRATOR_LOG.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "launch", "monitor", "verify"))
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or any(value < 0 for value in gpus):
        raise SystemExit("--gpus must contain non-negative GPU indices")
    if args.command == "prepare":
        rows = prepare()
        print(json.dumps({"status": "PREPARED", "episodes": len(rows)}))
        return 0
    if args.command == "run":
        return run(gpus)
    if args.command == "launch":
        print(json.dumps({"status": "LAUNCHED", "pid": launch(args.gpus)}))
        return 0
    if args.command == "verify":
        value = aggregate()
        print(json.dumps({"status": value["status"], "main_gates": value["main_gates"]}))
        return 0 if value["status"] == "PASS" else 2
    print(json.dumps(monitor(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
