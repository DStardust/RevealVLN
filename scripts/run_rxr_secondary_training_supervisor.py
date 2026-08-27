#!/usr/bin/env python3
"""Persistently wait for secondary features, then run the fixed data ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
PHASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SECONDARY = PHASE / "secondary_expansion_v1"
FEATURE_GATE = SECONDARY / "multibranch/RXR_SECONDARY_FEATURE_GATE.json"
PIPELINE_STATUS = SECONDARY / "RXR_SECONDARY_AUTOMATIC_PIPELINE_STATUS.json"
OUT = ROOT / "artifacts/evaluation/mf2_secondary_augmentation_v1"
STATUS = OUT / "RXR_SECONDARY_TRAINING_STATUS.json"
CONSOLE = OUT / "RXR_SECONDARY_TRAINING_SUPERVISOR.log"
PROTOCOL = OUT / "RXR_SECONDARY_AUGMENTATION_PROTOCOL_V1.json"
COMPARISON = OUT / "RXR_SECONDARY_AUGMENTATION_COMPARISON_V1.json"
SEEDS = (20260826, 20260827, 20260828)
GPUS = (0, 1, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def load(path: Path) -> dict | None:
    if not path.is_file() or path.is_symlink():
        return None
    return json.loads(path.read_text())


def write_protocol() -> None:
    value = {
        "schema_version": "revealnav-mf2-secondary-augmentation-protocol/1",
        "status": "FIXED_BEFORE_AUGMENTED_TRAINING",
        "conditions": [
            "primary_only",
            "primary_plus_automatic_secondary",
        ],
        "primary_control": (
            "reuse the already completed h128 balanced-tuning checkpoints "
            "under the identical seeds and optimization protocol"
        ),
        "augmented_condition": (
            "add automatic secondary pseudo-labels to train only; retain the "
            "unchanged human-audited primary development split"
        ),
        "models": ["balanced_full_ree", "balanced_history_direct_uad"],
        "seeds": list(SEEDS),
        "hidden_dim": 128,
        "maximum_epochs": 20,
        "early_stopping_patience": 4,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "batch_size": 8,
        "checkpoint_selection": [
            "higher development U/A/D macro-F1",
            "lower development false-ready rate",
            "lower native development loss",
        ],
        "positive_signal_criteria": {
            "full_macro_f1_mean_improves": True,
            "full_macro_f1_improves_at_least_two_seeds": True,
            "full_false_ready_mean_degradation_max": 0.02,
            "full_vs_history_macro_f1_gap_narrows": True,
            "augmentation_benefits_full_more_than_history": True,
        },
        "topology_only_training_included": False,
        "secondary_evaluation_use_allowed": False,
        "gold_access_allowed": False,
        "additional_hyperparameter_search_allowed": False,
        "paper_result": False,
    }
    if PROTOCOL.exists():
        current = json.loads(PROTOCOL.read_text())
        if current != value:
            raise RuntimeError("existing augmentation protocol drift")
    else:
        atomic_json(PROTOCOL, value)


def run_logged(name: str, command: list[str], environment: dict | None = None) -> int:
    log_path = OUT / f"{name}.log"
    with log_path.open("a") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def completed_seed(seed: int) -> bool:
    value = load(OUT / f"seed_{seed}/result.json")
    return bool(
        value
        and value.get("status") == "AUGMENTED_DEVELOPMENT_RUN_COMPLETE"
        and value.get("seed") == seed
        and value.get("hidden_dim") == 128
    )


def supervise() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "revealnav-mf2-secondary-training-status/1",
        "status": "WAITING_FOR_SECONDARY_FEATURE_GATE",
        "pid": os.getpid(),
        "started": time.time(),
        "training_authorized": False,
        "topology_only_training_included": False,
        "stages": [],
    }
    atomic_json(STATUS, state)
    while True:
        gate = load(FEATURE_GATE)
        pipeline = load(PIPELINE_STATUS)
        if pipeline and pipeline.get("status") == "FAIL":
            state["status"] = "FAIL_SECONDARY_PIPELINE"
            state["pipeline_status"] = pipeline
            atomic_json(STATUS, state)
            return 1
        if (
            gate
            and gate.get("status") == "FEATURE_GATE_PASS_AUTOMATIC_TRAIN_READY"
            and gate.get("training_authorized") is True
        ):
            state["feature_gate_sha256"] = sha256_file(FEATURE_GATE)
            break
        state["last_wait_update"] = time.time()
        state["upstream_stage"] = (
            pipeline.get("current_stage") if pipeline else None
        )
        atomic_json(STATUS, state)
        time.sleep(15)

    state["status"] = "BUILDING_AUGMENTED_MANIFEST"
    atomic_json(STATUS, state)
    write_protocol()
    code = run_logged("build_augmented_manifest", [
        sys.executable, "scripts/build_rxr_secondary_augmented_manifest.py"
    ])
    state["stages"].append({
        "name": "build_augmented_manifest", "returncode": code,
        "status": "PASS" if code == 0 else "FAIL",
    })
    if code:
        state["status"] = "FAIL_AUGMENTED_MANIFEST"
        atomic_json(STATUS, state)
        return code

    state["status"] = "TRAINING_THREE_SEEDS"
    state["training_authorized"] = True
    state["protocol_sha256"] = sha256_file(PROTOCOL)
    state["runs"] = {}
    processes = {}
    logs = {}
    for seed, gpu in zip(SEEDS, GPUS):
        if completed_seed(seed):
            state["runs"][str(seed)] = {
                "status": "PASS_REUSED_COMPLETE", "physical_gpu": gpu,
            }
            continue
        run_dir = OUT / f"seed_{seed}"
        if run_dir.exists():
            raise RuntimeError(f"incomplete run directory requires review: {run_dir}")
        log_path = OUT / f"seed_{seed}.log"
        logs[seed] = log_path.open("a")
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        processes[seed] = subprocess.Popen(
            [
                sys.executable,
                "scripts/run_rxr_secondary_augmentation_seed.py",
                "--seed", str(seed), "--device", "cuda",
            ],
            cwd=ROOT,
            env=environment,
            stdout=logs[seed],
            stderr=subprocess.STDOUT,
        )
        state["runs"][str(seed)] = {
            "status": "RUNNING", "physical_gpu": gpu,
            "pid": processes[seed].pid,
        }
    atomic_json(STATUS, state)
    while processes:
        for seed, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            logs.pop(seed).close()
            processes.pop(seed)
            state["runs"][str(seed)].update({
                "status": "PASS" if code == 0 else "FAIL",
                "returncode": code,
                "completed": time.time(),
            })
        state["last_training_update"] = time.time()
        atomic_json(STATUS, state)
        if processes:
            time.sleep(5)
    if not all(completed_seed(seed) for seed in SEEDS):
        state["status"] = "FAIL_TRAINING_RUN"
        atomic_json(STATUS, state)
        return 1

    state["status"] = "AGGREGATING_DEVELOPMENT_ABLATION"
    atomic_json(STATUS, state)
    aggregate_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    code = run_logged("aggregate", [
        sys.executable, "scripts/aggregate_rxr_secondary_augmentation.py"
    ], aggregate_env)
    state["stages"].append({
        "name": "aggregate", "returncode": code,
        "status": "PASS" if code == 0 else "FAIL",
    })
    if code or not COMPARISON.is_file():
        state["status"] = "FAIL_AGGREGATION"
        atomic_json(STATUS, state)
        return code or 1

    state["status"] = "RUNNING_REGRESSION"
    atomic_json(STATUS, state)
    regression = []
    for offset, test in enumerate((
        "tests/test_toporeveal.py", "tests/test_revealnav_mf2.py"
    )):
        code = run_logged(
            f"regression_{offset}", [sys.executable, test, "-q"]
        )
        regression.append({"test": test, "returncode": code})
    state["regression"] = regression
    if any(row["returncode"] for row in regression):
        state["status"] = "FAIL_REGRESSION"
        atomic_json(STATUS, state)
        return 1

    comparison = json.loads(COMPARISON.read_text())
    state.update({
        "status": "AUGMENTED_MODEL_DEVELOPMENT_ABLATION_COMPLETE",
        "completed": time.time(),
        "comparison": {
            "path": str(COMPARISON.relative_to(ROOT)),
            "sha256": sha256_file(COMPARISON),
            "result_status": comparison["status"],
        },
        "gold_payload_read": False,
        "paper_result": False,
    })
    atomic_json(STATUS, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        OUT.mkdir(parents=True, exist_ok=True)
        with CONSOLE.open("a") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({
            "status": "DETACHED",
            "pid": process.pid,
            "status_path": str(STATUS.relative_to(ROOT)),
            "log_path": str(CONSOLE.relative_to(ROOT)),
        }, indent=2))
        return 0
    return supervise()


if __name__ == "__main__":
    raise SystemExit(main())
