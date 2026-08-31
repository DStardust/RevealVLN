#!/usr/bin/env python3
"""Verify, run, and monitor the execution-equivalent fast CAR evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/training/mf3zm_car_v1"
SOURCE_PROTOCOL = OUT / "MF3ZM_CAR_PROTOCOL.json"
EXECUTION_PROTOCOL = OUT / "MF3ZM_CAR_FAST_EXECUTION_PROTOCOL.json"
EQUIVALENCE = OUT / "MF3ZM_CAR_FAST_EQUIVALENCE.json"
PARENT_PROGRESS = OUT / "MF3ZM_CAR_FAST_PARENT.json"
JOB_DIR = OUT / "fast_jobs"
RESULT = OUT / "MF3ZM_CAR_TRAIN_DEVELOPMENT_RESULT.json"
GATE = OUT / "gates/MF3ZM_CAR_MODEL.pt"
TRAINER = ROOT / "scripts/train_mf3zm_car.py"
FAST_IMPLEMENTATION = ROOT / "revealnav_mf3/car_fast.py"

THREADS_PER_JOB = 12
MAX_FITS = 66
TRAINING_STEPS = 800
SEEDS_PER_FIT = 3
JOBS = {
    "car_mainline": {},
    "car_no_scene_constraint": {"scene_constraint": False},
    "car_soft_risk": {"risk_mode": "soft"},
    "car_28d": {"representation": "engineered_28d"},
    "car_policy_only": {"representation": "policy_only"},
    "car_no_risk": {"risk_mode": "none"},
    "rxr_only_car": {"arm": "RxR"},
    "r2r_only_car": {"arm": "R2R"},
    "dsr_v1_expanded_data": None,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_equivalence() -> int:
    import torch
    from revealnav_mf3.car import CAR_POLICY_FEATURE_NAMES
    from revealnav_mf3.car_fast import fit_car_ensemble_fast
    from revealnav_mf3.car_selection import fit_car_ensemble

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(7)
    rows = 24
    inputs = {"policy_only": rng.normal(
        size=(rows, len(CAR_POLICY_FEATURE_NAMES))
    ).astype(np.float32)}
    target = rng.normal(scale=0.2, size=rows)
    scenes = np.asarray([f"scene_{index % 6}" for index in range(rows)])
    datasets = np.asarray([
        "RxR" if index % 2 else "R2R" for index in range(rows)
    ])
    episodes = np.asarray([f"episode_{index}" for index in range(rows)])
    kwargs = {
        "weight_decay": 0.001,
        "seeds": (11, 12, 13),
        "learning_rate": 0.005,
        "dual_learning_rate": 0.05,
        "training_steps": 5,
        "dual_cap": 100.0,
        "representation": "policy_only",
        "risk_mode": "hard",
        "scene_constraint": True,
        "use_cuda": False,
    }
    reference = fit_car_ensemble(
        inputs, target, scenes, datasets, episodes, **kwargs
    )
    accelerated = fit_car_ensemble_fast(
        inputs, target, scenes, datasets, episodes, **kwargs
    )
    max_difference = max(
        float((value - accelerated[0][model_index].state_dict()[name])
              .abs().max())
        for model_index, model in enumerate(reference[0])
        for name, value in model.state_dict().items()
    )
    hard_equal = (
        [item["hard_selected_by_domain"] for item in reference[2]]
        == [item["hard_selected_by_domain"] for item in accelerated[2]]
    )
    zero_equal = (
        [(item["zero_risk_steps"], item["zero_scene_steps"])
         for item in reference[2]]
        == [(item["zero_risk_steps"], item["zero_scene_steps"])
            for item in accelerated[2]]
    )
    value = {
        "schema_version": "revealnav-mf3zm-car-fast-equivalence/1",
        "status": "PASS" if (
            reference[1] == accelerated[1]
            and max_difference <= 1e-7 and hard_equal and zero_equal
        ) else "FAIL",
        "algorithm_change": False,
        "synthetic_rows": rows,
        "training_steps": kwargs["training_steps"],
        "initialization_hashes_equal": reference[1] == accelerated[1],
        "maximum_state_absolute_difference": max_difference,
        "hard_authorization_counts_equal": hard_equal,
        "zero_selection_diagnostics_equal": zero_equal,
    }
    atomic_json(EQUIVALENCE, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "PASS" else 1


def build_execution_protocol() -> dict:
    equivalence = json.loads(EQUIVALENCE.read_text())
    if equivalence.get("status") != "PASS":
        raise RuntimeError("fast CAR equivalence is not PASS")
    return {
        "schema_version": "revealnav-mf3zm-car-fast-execution/1",
        "status": "SEALED_BEFORE_FAST_TRAINING",
        "scientific_revision": "mf3zm_car_v1",
        "algorithm_change": False,
        "scientific_protocol": inventory(SOURCE_PROTOCOL),
        "fast_implementation": inventory(FAST_IMPLEMENTATION),
        "runner": inventory(Path(__file__).resolve()),
        "equivalence": inventory(EQUIVALENCE),
        "execution_changes": [
            "constant fold/domain/scene tensors are precomputed",
            "leave-one-scene constraints are evaluated as a tensor batch",
            "independent pre-registered arms run in separate CPU processes",
            "CPU numerical backend replaces CUDA for the small intervention head",
        ],
        "threads_per_job": THREADS_PER_JOB,
        "jobs": list(JOBS),
        "frozen_training": {
            "training_steps": TRAINING_STEPS,
            "seeds_per_fit": SEEDS_PER_FIT,
            "maximum_fits_per_job": MAX_FITS,
            "decision_rule": "switch_logit > 0",
        },
        "public_split_access": False,
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
    }


def seal_execution() -> int:
    if EXECUTION_PROTOCOL.exists():
        raise RuntimeError("fast execution protocol already exists")
    value = build_execution_protocol()
    atomic_json(EXECUTION_PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "jobs": len(value["jobs"]),
        "threads_per_job": value["threads_per_job"],
        "sha256": sha256_file(EXECUTION_PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def verify_execution_protocol() -> dict:
    if not EXECUTION_PROTOCOL.is_file() or EXECUTION_PROTOCOL.is_symlink():
        raise RuntimeError("fast execution protocol is unavailable")
    value = json.loads(EXECUTION_PROTOCOL.read_text())
    if (
        value.get("status") != "SEALED_BEFORE_FAST_TRAINING"
        or value.get("algorithm_change") is not False
        or value.get("public_split_access") is not False
        or value.get("jobs") != list(JOBS)
        or int(value.get("threads_per_job", -1)) != THREADS_PER_JOB
    ):
        raise RuntimeError("fast execution protocol semantics drift")
    for key in ("scientific_protocol", "fast_implementation", "runner",
                "equivalence"):
        item = value[key]
        path = ROOT / item["path"]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != int(item["bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"fast execution source drift: {item['path']}")
    return value


class JobProgress:
    def __init__(self, job: str) -> None:
        self.job = job
        self.path = JOB_DIR / f"{job}.progress.json"
        self.started = time.time()
        self.completed_fits = 0
        self.current: dict = {}
        self.status = "RUNNING"
        self.message = "loading sealed data"
        self.write()

    def write(self) -> None:
        atomic_json(self.path, {
            "status": self.status,
            "job": self.job,
            "pid": os.getpid(),
            "started_unix": self.started,
            "updated_unix": time.time(),
            "completed_fits": self.completed_fits,
            "maximum_fits": MAX_FITS,
            "current": self.current,
            "message": self.message,
        })

    def begin_fit(self, kind: str, rows: int, weight_decay: float) -> None:
        self.current = {
            "kind": kind,
            "fit": self.completed_fits + 1,
            "rows": int(rows),
            "weight_decay": float(weight_decay),
            "seed_index": 0,
            "seed_count": SEEDS_PER_FIT,
            "step": 0,
            "steps": TRAINING_STEPS,
        }
        self.message = "fit running"
        self.write()

    def update_step(
        self, seed_index: int, seed: int, step: int, steps: int,
    ) -> None:
        self.current.update({
            "seed_index": int(seed_index),
            "seed": int(seed),
            "step": int(step),
            "steps": int(steps),
        })
        self.write()

    def end_fit(self) -> None:
        self.completed_fits += 1
        self.current = {}
        self.message = "fit complete"
        self.write()

    def finish(self) -> None:
        self.completed_fits = MAX_FITS
        self.current = {}
        self.status = "COMPLETE"
        self.message = "job complete"
        self.write()

    def error(self, exc: BaseException) -> None:
        self.status = "ERROR"
        self.message = f"{type(exc).__name__}: {exc}"
        self.current["traceback"] = traceback.format_exc(limit=10)
        self.write()


def worker(job: str) -> int:
    if job not in JOBS:
        raise ValueError(f"unknown fast CAR job: {job}")
    verify_execution_protocol()
    import torch
    import revealnav_mf3.car_selection as car_selection
    import revealnav_mf3.dsr_selection as dsr_selection
    from revealnav_mf3.car_fast import fit_car_ensemble_fast

    torch.set_num_threads(THREADS_PER_JOB)
    torch.set_num_interop_threads(1)
    progress = JobProgress(job)
    output = JOB_DIR / f"{job}.json"
    model_output = JOB_DIR / f"{job}.models.pt"
    if output.exists() or model_output.exists():
        raise RuntimeError(f"fast CAR job output already exists: {job}")
    trainer = load_module(TRAINER, f"mf3zm_car_fast_worker_{job}")
    try:
        protocol, rows, arrays = trainer._load_data()
        original_dsr_fit = dsr_selection._fit_ensemble

        def accelerated_fit(*args, **kwargs):
            target = args[1] if len(args) > 1 else kwargs["target"]
            progress.begin_fit(
                "car", len(target), float(kwargs["weight_decay"])
            )
            kwargs["use_cuda"] = False
            result = fit_car_ensemble_fast(
                *args, **kwargs, progress_callback=progress.update_step
            )
            progress.end_fit()
            return result

        def observed_dsr_fit(*args, **kwargs):
            target = args[1] if len(args) > 1 else kwargs["target"]
            progress.begin_fit(
                "dsr", len(target), float(kwargs["weight_decay"])
            )
            result = original_dsr_fit(*args, **kwargs)
            progress.end_fit()
            return result

        car_selection.fit_car_ensemble = accelerated_fit
        dsr_selection._fit_ensemble = observed_dsr_fit
        if job == "dsr_v1_expanded_data":
            value = trainer._fit_dsr_control(protocol, rows, arrays)
            models = []
        else:
            value, models = trainer._fit_arm(
                protocol, rows, arrays, **JOBS[job]
            )
        if models:
            part = model_output.with_name(model_output.name + ".part")
            torch.save({
                "state_dicts": [
                    {name: tensor.detach().cpu()
                     for name, tensor in model.state_dict().items()}
                    for model in models
                ]
            }, part)
            os.replace(part, model_output)
        atomic_json(output, {
            "status": "COMPLETE",
            "job": job,
            "value": trainer._jsonable(value),
            "models": inventory(model_output) if model_output.exists() else None,
        })
        progress.finish()
        return 0
    except BaseException as exc:
        progress.error(exc)
        atomic_json(output, {
            "status": "ERROR",
            "job": job,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        })
        return 1


def run_parent() -> int:
    verify_execution_protocol()
    if RESULT.exists() or GATE.exists():
        raise RuntimeError("refusing to overwrite CAR result or checkpoint")
    if JOB_DIR.exists() and any(JOB_DIR.iterdir()):
        raise RuntimeError("fast CAR job directory is not empty")
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    atomic_json(PARENT_PROGRESS, {
        "status": "RUNNING", "pid": os.getpid(),
        "started_unix": started, "jobs": list(JOBS),
    })
    processes = {}
    for job in JOBS:
        log_path = JOB_DIR / f"{job}.log"
        stream = log_path.open("w")
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(THREADS_PER_JOB),
            "MKL_NUM_THREADS": str(THREADS_PER_JOB),
            "OPENBLAS_NUM_THREADS": str(THREADS_PER_JOB),
            "PYTHONNOUSERSITE": "1",
        })
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker",
             "--job", job],
            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
        )
        stream.close()
        processes[job] = process
    exit_codes = {job: process.wait() for job, process in processes.items()}

    trainer = load_module(TRAINER, "mf3zm_car_fast_aggregate")
    protocol = trainer.verify_protocol()
    rows = trainer._canonical_rows()
    controls: dict[str, dict] = {}
    errors: dict[str, dict] = {}
    main: dict | None = None
    main_models = None
    for job in JOBS:
        output = JOB_DIR / f"{job}.json"
        value = json.loads(output.read_text()) if output.is_file() else {
            "status": "ERROR", "error": "worker produced no output"
        }
        if exit_codes[job] != 0 or value.get("status") != "COMPLETE":
            errors[job] = {
                "status": "CONTROL_ERROR" if job != "car_mainline"
                else "MAINLINE_ERROR",
                "error_type": value.get("error_type", "WorkerError"),
                "error": value.get("error", f"worker exit {exit_codes[job]}"),
                "traceback": value.get("traceback"),
            }
            continue
        if job == "car_mainline":
            main = value["value"]
            main_models = value.get("models")
        else:
            controls[job] = value["value"]

    if main is None:
        main = {
            "status": "NESTED_CAR_FAIL",
            "failure_reasons": ["mainline_execution_error"],
        }
    main_pass = main.get("status") == "NESTED_CAR_PASS"
    gate = None
    if main_pass:
        if not main_models:
            raise RuntimeError("passing mainline omitted final models")
        payload = __import__("torch").load(
            ROOT / main_models["path"], map_location="cpu", weights_only=True
        )
        from revealnav_mf3.car import build_model
        models = []
        for state_dict in payload["state_dicts"]:
            model = build_model("semantic")
            model.load_state_dict(state_dict, strict=True)
            models.append(model)
        gate = trainer._save_gate(models, protocol)

    result = {
        "schema_version": "revealnav-mf3zm-car-result/1",
        "status": "TRAIN_DEVELOPMENT_PASS" if main_pass
        else "TRAIN_DEVELOPMENT_FAIL",
        "revision": "mf3zm_car_v1",
        "source_protocol": trainer.inventory(SOURCE_PROTOCOL),
        "source_audit": trainer.inventory(trainer.AUDIT),
        "execution_protocol": inventory(EXECUTION_PROTOCOL),
        "execution_mode": "batched_constraints_parallel_cpu",
        "rows": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "domains": dict(Counter(row["dataset"] for row in rows)),
        "mainline": main,
        "controls_run_independently_of_mainline": True,
        "controls": controls,
        "control_errors": errors,
        "model": gate,
        "checkpoint_created": gate is not None,
        "old_confirmation_reused": False,
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
        "public_split_access": {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        },
    }
    trainer.atomic_json(RESULT, trainer._jsonable(result))
    atomic_json(PARENT_PROGRESS, {
        "status": "COMPLETE", "pid": os.getpid(),
        "started_unix": started, "finished_unix": time.time(),
        "jobs": list(JOBS), "worker_exit_codes": exit_codes,
        "scientific_status": result["status"],
        "checkpoint_created": result["checkpoint_created"],
    })
    return 0 if main_pass and not errors else 2


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def monitor() -> int:
    parent = json.loads(PARENT_PROGRESS.read_text()) if PARENT_PROGRESS.is_file() else {}
    now = time.time()
    total = 0.0
    lines = []
    for job in JOBS:
        path = JOB_DIR / f"{job}.progress.json"
        if not path.is_file():
            lines.append(f"{job:29} waiting")
            continue
        value = json.loads(path.read_text())
        completed = float(value.get("completed_fits", 0))
        current = value.get("current") or {}
        if current:
            seed_index = float(current.get("seed_index", 0))
            step = float(current.get("step", 0))
            steps = max(1.0, float(current.get("steps", TRAINING_STEPS)))
            completed += (seed_index + step / steps) / SEEDS_PER_FIT
        if value.get("status") == "COMPLETE":
            completed = float(MAX_FITS)
        total += min(float(MAX_FITS), completed)
        percent = 100.0 * min(float(MAX_FITS), completed) / MAX_FITS
        detail = ""
        if current:
            detail = (
                f" fit {current.get('fit')}/{MAX_FITS}"
                f" seed {int(current.get('seed_index', 0)) + 1}/3"
                f" step {current.get('step', 0)}/{current.get('steps', TRAINING_STEPS)}"
            )
        lines.append(
            f"{job:29} {percent:6.2f}% {value.get('status')}{detail}"
        )
    overall = 100.0 * total / (len(JOBS) * MAX_FITS)
    width = 40
    filled = min(width, int(width * overall / 100.0))
    elapsed = now - float(parent.get("started_unix", now))
    print(
        f"MF3ZM-CAR [{'#' * filled}{'-' * (width - filled)}] "
        f"{overall:6.2f}% {parent.get('status', 'NOT_STARTED')} "
        f"elapsed {duration(elapsed)}"
    )
    print("\n".join(lines))
    if RESULT.is_file():
        result = json.loads(RESULT.read_text())
        print(
            f"result={result.get('status')} "
            f"checkpoint={result.get('checkpoint_created')} "
            f"public_unseen={result.get('public_unseen_authorized')}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "verify", "seal", "run", "worker", "monitor",
    ))
    parser.add_argument("--job", choices=tuple(JOBS))
    args = parser.parse_args()
    if args.command == "verify":
        return verify_equivalence()
    if args.command == "seal":
        return seal_execution()
    if args.command == "run":
        return run_parent()
    if args.command == "worker":
        if args.job is None:
            parser.error("worker requires --job")
        return worker(args.job)
    return monitor()


if __name__ == "__main__":
    raise SystemExit(main())
