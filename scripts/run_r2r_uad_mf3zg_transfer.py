#!/usr/bin/env python3
"""Seal, run, and verify zero-tuning MF3ZG transfer to R2R."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
WORKER = ROOT / "scripts/r2r_uad_mf3zg_worker.py"
BASE_WORKER = ROOT / "scripts/rxr_uad_controller_worker_mf3.py"
DATA_ROOT = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed_xlmr"
)
R2R_CHECKPOINT = ROOT / (
    "third_party/ETP-R1/data/logs/checkpoints/"
    "release_r2r_grpo/store/ckpt.iter270.pth"
)
JOINT_PRETRAINED = ROOT / (
    "third_party/ETP-R1/pretrained/r2r_rxr_ce/"
    "mlm.sap_habitat_depth/store2/model_step_367500.pt"
)
MF3ZG_GATE = ROOT / (
    "artifacts/training/mf3zg_hierarchical_core_preserving_gate_v1/"
    "MF3ZG_SHADOW_GATE.json"
)
MF3ZG_FREEZE = ROOT / (
    "artifacts/evaluation/mf3zg_core_preserving_hierarchy_freeze_v1/"
    "MF3ZG_VAL_SEEN_FREEZE.json"
)
OUT = ROOT / "artifacts/evaluation/mf3zg_zero_tuning_r2r_transfer_v1"
SEEN_PROTOCOL = OUT / "MF3ZG_R2R_VAL_SEEN_PROTOCOL.json"
SEEN_PROGRESS = OUT / "MF3ZG_R2R_VAL_SEEN_PROGRESS.json"
SEEN_RESULT = OUT / "MF3ZG_R2R_VAL_SEEN_RESULT.json"
UNSEEN_PROTOCOL = OUT / "MF3ZG_R2R_VAL_UNSEEN_PROTOCOL.json"
UNSEEN_PROGRESS = OUT / "MF3ZG_R2R_VAL_UNSEEN_PROGRESS.json"
UNSEEN_RESULT = OUT / "MF3ZG_R2R_VAL_UNSEEN_RESULT.json"
METRICS = ("success", "spl", "ndtw", "sdtw")
UTILITY_WEIGHTS = {"ndtw": 0.50, "sdtw": 0.25, "spl": 0.25}
SEEN_SALT = "revealnav-mf3zg-r2r-val-seen-one-per-scene/1"
UNSEEN_SALT = "revealnav-mf3zg-r2r-val-unseen-full-split/1"
EXPECTED_BIG_ASSETS = {
    R2R_CHECKPOINT: {
        "bytes": 1874802927,
        "sha256": "8f90cebba7eefb9648054aa74e8c8664f23e643073ef033b99edfbe85c54f61c",
    },
    JOINT_PRETRAINED: {
        "bytes": 2264172675,
        "sha256": "203fe62cc22c63261a5c5b6a3638bc52fd3b08a7f09dd31d8539bf2beab6c3cf",
    },
}
SOURCE_FILES = (
    WORKER,
    BASE_WORKER,
    MF3ZG_GATE,
    MF3ZG_FREEZE,
    ROOT / "artifacts/training/mf3ze_action_aligned_return_gate_v1/MF3ZE_GATE_MODELS.npz",
    ROOT / "artifacts/training/mf3zf_action_aligned_return_gate_v1/MF3ZF_GATE_MODELS.npz",
    *tuple(
        ROOT / (
            "artifacts/training/mf3v_horizon_ranker_v1/fold_final/"
            f"seed_{seed}/horizon_ranker_mf3v.pt"
        )
        for seed in (20260826, 20260827, 20260828)
    ),
)


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


def _dataset_path(split: str) -> Path:
    return DATA_ROOT / split / f"{split}.json.gz"


def _load_episodes(split: str) -> list[dict]:
    path = _dataset_path(split)
    if path.is_symlink() or not path.is_file() or ROOT not in path.resolve().parents:
        raise RuntimeError(f"unsafe R2R dataset path: {path}")
    with gzip.open(path, "rt") as stream:
        value = json.load(stream)
    rows = value.get("episodes")
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid R2R {split} payload")
    ids = [str(row.get("episode_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate R2R {split} episode id")
    return rows


def _scene(row: dict) -> str:
    value = Path(str(row.get("scene_id", ""))).stem
    if len(value) != 11:
        raise RuntimeError("invalid MP3D scene id in R2R payload")
    return value


def _digest(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def select_one_per_scene(rows: list[dict], salt: str = SEEN_SALT) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_scene(row)].append(row)
    selected = []
    for scene in sorted(grouped):
        row = min(
            grouped[scene],
            key=lambda item: _digest(salt, f"{scene}:{item['episode_id']}"),
        )
        selected.append({
            "scene_id": scene,
            "episode_id": str(row["episode_id"]),
            "selection_digest": _digest(
                salt, f"{scene}:{row['episode_id']}"
            ),
        })
    return selected


def _full_split(rows: list[dict]) -> list[dict]:
    return [
        {
            "scene_id": _scene(row),
            "episode_id": str(row["episode_id"]),
            "selection_digest": _digest(
                UNSEEN_SALT, f"{_scene(row)}:{row['episode_id']}"
            ),
        }
        for row in sorted(rows, key=lambda item: int(item["episode_id"]))
    ]


def _evidence(path: Path, *, hash_file: bool = True) -> dict:
    if path.is_symlink() or not path.is_file() or ROOT not in path.resolve().parents:
        raise RuntimeError(f"unsafe or absent source: {path}")
    value = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
    }
    if hash_file:
        value["sha256"] = sha256_file(path)
    return value


def _source_closure() -> list[dict]:
    return [_evidence(path) for path in SOURCE_FILES]


def _big_asset_closure() -> list[dict]:
    result = []
    for path, expected in EXPECTED_BIG_ASSETS.items():
        evidence = _evidence(path)
        if evidence["bytes"] != expected["bytes"] or evidence["sha256"] != expected["sha256"]:
            raise RuntimeError(f"accepted R2R asset drift: {path}")
        result.append(evidence)
    return result


def _validate_method_freeze() -> None:
    gate = json.loads(MF3ZG_GATE.read_text())
    freeze = json.loads(MF3ZG_FREEZE.read_text())
    if not (
        gate.get("status") == "SHADOW_GATE_PASS"
        and gate.get("task_metric_run_authorized") is True
        and gate.get("controls", {}).get("unseen_or_test_read") is False
        and freeze.get("status") == "MF3ZG_VAL_SEEN_FROZEN"
        and freeze.get("success_summary", {}).get("all_task_metric_gates_pass") is True
    ):
        raise RuntimeError("MF3ZG is not frozen for zero-tuning transfer")


def _validate_closure(protocol: dict) -> None:
    for evidence in protocol["source_closure"]:
        path = ROOT / evidence["path"]
        if (
            path.is_symlink() or not path.is_file()
            or path.stat().st_size != evidence["bytes"]
            or sha256_file(path) != evidence["sha256"]
        ):
            raise RuntimeError(f"R2R transfer source drift: {path}")
    for evidence in protocol["big_asset_closure"]:
        path = ROOT / evidence["path"]
        expected = EXPECTED_BIG_ASSETS.get(path)
        if (
            expected is None or path.is_symlink() or not path.is_file()
            or path.stat().st_size != evidence["bytes"]
            or evidence["bytes"] != expected["bytes"]
            or evidence["sha256"] != expected["sha256"]
        ):
            raise RuntimeError(f"R2R transfer big-asset drift: {path}")


def _seen_protocol_value() -> dict:
    _validate_method_freeze()
    dataset = _dataset_path("val_seen")
    rows = _load_episodes("val_seen")
    selected = select_one_per_scene(rows)
    if len(rows) != 778 or len(selected) != 53:
        raise RuntimeError("R2R val_seen inventory drift")
    return {
        "schema_version": "revealnav-mf3zg-r2r-val-seen-transfer-protocol/1",
        "status": "SEALED_BEFORE_R2R_VAL_SEEN_METRICS",
        "controller_revision": "mf3zg",
        "parameters_frozen_from": "RxR train-only gates and RxR val_seen freeze",
        "threshold_or_model_tuning_on_r2r": False,
        "paired_unit": "one deterministic episode per R2R val_seen scene",
        "selection_salt": SEEN_SALT,
        "selection": selected,
        "inventory": {"episodes": len(rows), "scenes": len(selected)},
        "runs": {"baseline": len(selected), "mf3zg": len(selected), "total": 2 * len(selected)},
        "primary_utility": "0.50*nDTW + 0.25*SDTW + 0.25*SPL",
        "success_gate": (
            "paired mean utility > 0; SR, SPL, and nDTW deltas non-negative; "
            "at least one executed MF3ZG action differs from baseline"
        ),
        "uncertainty": "10000 deterministic scene bootstrap replicates",
        "dataset": _evidence(dataset),
        "source_closure": _source_closure(),
        "big_asset_closure": _big_asset_closure(),
        "r2r_val_unseen_read_or_authorized": False,
        "test_or_test_challenge_accessed": False,
    }


def seal_seen() -> int:
    value = _seen_protocol_value()
    if SEEN_PROTOCOL.exists() and json.loads(SEEN_PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed R2R val_seen transfer protocol drift")
    if not SEEN_PROTOCOL.exists():
        atomic_json(SEEN_PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "episodes": len(value["selection"]),
        "runs": value["runs"]["total"],
        "protocol": str(SEEN_PROTOCOL.relative_to(ROOT)),
    }))
    return 0


def _unseen_protocol_value() -> dict:
    seen = json.loads(SEEN_RESULT.read_text())
    if not (
        seen.get("status") == "R2R_VAL_SEEN_TRANSFER_GATE_PASS"
        and all(seen.get("gates", {}).values())
    ):
        raise RuntimeError("R2R val_seen transfer gate has not passed")
    rows = _load_episodes("val_unseen")
    selected = _full_split(rows)
    scenes = {_scene(row) for row in rows}
    if len(rows) != 1839 or len(scenes) != 11:
        raise RuntimeError("R2R val_unseen inventory drift")
    return {
        "schema_version": "revealnav-mf3zg-r2r-val-unseen-full-protocol/1",
        "status": "SEALED_BEFORE_R2R_VAL_UNSEEN_METRICS",
        "controller_revision": "mf3zg",
        "parameters_frozen_from": "RxR only; R2R val_seen was a zero-tuning transfer gate",
        "threshold_or_model_tuning_on_r2r": False,
        "selection": selected,
        "selection_salt": UNSEEN_SALT,
        "paired_unit": "every episode in the public R2R val_unseen split",
        "inventory": {"episodes": len(rows), "scenes": len(scenes)},
        "runs": {"baseline": len(rows), "mf3zg": len(rows), "total": 2 * len(rows)},
        "primary_utility": "0.50*nDTW + 0.25*SDTW + 0.25*SPL",
        "success_gate": (
            "paired utility scene-bootstrap lower 95% bound > 0; SR and SPL "
            "deltas non-negative; at least one is positive; executed changes present"
        ),
        "uncertainty": "10000 deterministic MP3D-scene cluster bootstrap replicates",
        "dataset": _evidence(_dataset_path("val_unseen")),
        "source_closure": json.loads(SEEN_PROTOCOL.read_text())["source_closure"],
        "big_asset_closure": json.loads(SEEN_PROTOCOL.read_text())["big_asset_closure"],
        "seen_result": _evidence(SEEN_RESULT),
        "prior_r2r_val_unseen_use_by_obsolete_non_mf3zg_branches": True,
        "mf3zg_or_its_thresholds_selected_on_r2r_val_unseen": False,
        "test_or_test_challenge_accessed": False,
    }


def seal_unseen() -> int:
    value = _unseen_protocol_value()
    if UNSEEN_PROTOCOL.exists() and json.loads(UNSEEN_PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed R2R val_unseen protocol drift")
    if not UNSEEN_PROTOCOL.exists():
        atomic_json(UNSEEN_PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "episodes": len(value["selection"]),
        "runs": value["runs"]["total"],
        "protocol": str(UNSEEN_PROTOCOL.relative_to(ROOT)),
    }))
    return 0


def _run_name(mode: str, episode_id: str) -> str:
    return f"{mode}_ep_{episode_id}"


def _execute(
    split: str, preflight: bool, gpus: tuple[int, ...],
    workers_per_gpu: int, resume: bool,
) -> int:
    if any(gpu not in (0, 1) for gpu in gpus):
        raise ValueError("this campaign is restricted to free GPUs 0 and 1")
    if workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    protocol_path = SEEN_PROTOCOL if split == "val_seen" else UNSEEN_PROTOCOL
    progress_path = SEEN_PROGRESS if split == "val_seen" else UNSEEN_PROGRESS
    protocol = json.loads(protocol_path.read_text())
    _validate_closure(protocol)
    rows = protocol["selection"][:1] if preflight else protocol["selection"]
    planned = [
        (mode, row) for row in rows for mode in ("baseline", "ensemble")
    ]
    root = OUT / split / ("preflight" if preflight else "full")
    root.mkdir(parents=True, exist_ok=True)
    completed: list[dict] = []
    queue = []
    for mode, row in planned:
        name = _run_name(mode, row["episode_id"])
        directory = root / "runs" / name
        summary = directory / "RUN_SUMMARY.json"
        if resume and summary.is_file():
            value = json.loads(summary.read_text())
            if value.get("status") == "PASS":
                completed.append({"job": name, "returncode": 0, "recovered": True})
                continue
        if directory.exists():
            destination = root / "interrupted" / f"{name}_{int(time.time())}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(directory, destination)
        queue.append((mode, row, name, directory))
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    active: list[dict] = []
    started = time.time()
    while queue or active:
        while queue and slots:
            mode, row, name, directory = queue.pop(0)
            gpu = slots.pop(0)
            logs = root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            stdout = (logs / f"{name}.stdout").open("w")
            stderr = (logs / f"{name}.stderr").open("w")
            command = [
                str(PYTHON), str(WORKER), "--episode-id", row["episode_id"],
                "--mode", mode, "--split", split, "--run-dir", str(directory),
            ]
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            if split == "val_unseen":
                command.extend(("--authorization-json", str(protocol_path)))
                environment["REVEALNAV_R2R_UNSEEN_PROTOCOL_SHA256"] = sha256_file(
                    protocol_path
                )
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
            )
            active.append({
                "process": process, "gpu": gpu, "mode": mode,
                "episode_id": row["episode_id"], "job": name,
                "streams": (stdout, stderr),
            })
        elapsed = time.time() - started
        finished_now = len(completed)
        rate = finished_now / elapsed if elapsed > 0 else 0.0
        remaining = len(planned) - finished_now
        atomic_json(progress_path, {
            "status": "RUNNING",
            "split": split,
            "preflight": preflight,
            "total": len(planned),
            "completed": finished_now,
            "failed": sum(row["returncode"] != 0 for row in completed),
            "queued": len(queue),
            "active": [
                {key: row[key] for key in ("gpu", "mode", "episode_id")}
                for row in active
            ],
            "elapsed_s": round(elapsed, 1),
            "eta_s": None if rate == 0 else round(remaining / rate, 1),
            "monitor_command": (
                f"watch -n 5 'python -m json.tool {progress_path}'"
            ),
        })
        time.sleep(1)
        for row in list(active):
            code = row["process"].poll()
            if code is None:
                continue
            for stream in row["streams"]:
                stream.close()
            completed.append({
                "job": row["job"], "episode_id": row["episode_id"],
                "mode": row["mode"], "gpu": row["gpu"], "returncode": code,
            })
            slots.append(row["gpu"])
            slots.sort()
            active.remove(row)
    failures = [row for row in completed if row["returncode"] != 0]
    atomic_json(progress_path, {
        "status": "COMPLETE" if not failures else "FAIL",
        "split": split,
        "preflight": preflight,
        "total": len(planned),
        "completed": len(completed),
        "failed": len(failures),
        "failures": failures,
        "queued": 0,
        "active": [],
        "elapsed_s": round(time.time() - started, 1),
        "eta_s": 0,
    })
    return 0 if not failures else 2


def _utility(metrics: dict) -> float:
    return sum(UTILITY_WEIGHTS[key] * float(metrics[key]) for key in UTILITY_WEIGHTS)


def _load_summary(path: Path, *, split: str, mode: str, episode_id: str) -> dict:
    value = json.loads(path.read_text())
    boundary = (
        value.get("status") == "PASS"
        and value.get("split") == split
        and value.get("mode") == mode
        and value.get("episode_id") == episode_id
        and value.get("revision") == "mf3zg"
        and value.get("threshold_or_model_tuning_on_r2r") is False
        and isinstance(value.get("metrics"), dict)
        and all(math.isfinite(float(value["metrics"][key])) for key in METRICS)
    )
    if split == "val_seen":
        boundary = boundary and value.get("public_unseen_accessed") is False
    else:
        boundary = boundary and value.get("public_unseen_authorized") is True
    if mode == "ensemble":
        boundary = boundary and value.get("executed_action_validation", {}).get("all_equal") is True
    if not boundary:
        raise RuntimeError(f"R2R MF3ZG summary boundary failure: {path}")
    return value


def _percentile(values: list[float], q: float) -> float:
    return sorted(values)[round((len(values) - 1) * q)]


def scene_cluster_bootstrap(
    rows: list[dict], metrics: tuple[str, ...], seed: int = 20260830,
    replicates: int = 10000,
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    scenes = sorted(grouped)
    rng = random.Random(seed)
    samples = {metric: [] for metric in metrics}
    for _ in range(replicates):
        drawn = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        sample = [row for scene in drawn for row in grouped[scene]]
        for metric in metrics:
            samples[metric].append(
                sum(float(row[metric]) for row in sample) / len(sample)
            )
    return {
        metric: {
            "mean": sum(float(row[metric]) for row in rows) / len(rows),
            "scene_bootstrap_95pct": [
                _percentile(samples[metric], 0.025),
                _percentile(samples[metric], 0.975),
            ],
        }
        for metric in metrics
    }


def _verify(split: str, preflight: bool) -> int:
    protocol_path = SEEN_PROTOCOL if split == "val_seen" else UNSEEN_PROTOCOL
    result_path = SEEN_RESULT if split == "val_seen" else UNSEEN_RESULT
    protocol = json.loads(protocol_path.read_text())
    _validate_closure(protocol)
    selected = protocol["selection"][:1] if preflight else protocol["selection"]
    root = OUT / split / ("preflight" if preflight else "full")
    pairs = []
    total_changed = 0
    total_authorized = 0
    for row in selected:
        summaries = {}
        for mode in ("baseline", "ensemble"):
            path = root / "runs" / _run_name(mode, row["episode_id"]) / "RUN_SUMMARY.json"
            summaries[mode] = _load_summary(
                path, split=split, mode=mode, episode_id=row["episode_id"]
            )
        baseline = summaries["baseline"]["metrics"]
        treatment = summaries["ensemble"]["metrics"]
        delta = {
            metric: float(treatment[metric]) - float(baseline[metric])
            for metric in METRICS
        }
        delta["utility"] = _utility(treatment) - _utility(baseline)
        controller = summaries["ensemble"]["controller"]
        total_changed += int(controller["actions_changed"])
        total_authorized += int(controller["authorized"])
        pairs.append({
            "scene_id": row["scene_id"],
            "episode_id": row["episode_id"],
            **delta,
            "actions_changed": int(controller["actions_changed"]),
            "authorized": int(controller["authorized"]),
        })
    if preflight:
        path = OUT / split / "MF3ZG_R2R_PREFLIGHT.json"
        atomic_json(path, {
            "status": "PREFLIGHT_PASS",
            "split": split,
            "runs": len(selected) * 2,
            "controller_loaded": True,
            "declared_actions_match_execution": True,
            "task_metric_claim": False,
        })
        return 0
    aggregate = scene_cluster_bootstrap(
        pairs, (*METRICS, "utility"),
        seed=20260830 if split == "val_seen" else 20260831,
    )
    if split == "val_seen":
        gates = {
            "utility_point_positive": aggregate["utility"]["mean"] > 0,
            "success_point_nonnegative": aggregate["success"]["mean"] >= 0,
            "spl_point_nonnegative": aggregate["spl"]["mean"] >= 0,
            "ndtw_point_nonnegative": aggregate["ndtw"]["mean"] >= 0,
            "effective_actions_present": total_changed > 0,
        }
        passed_status = "R2R_VAL_SEEN_TRANSFER_GATE_PASS"
        failed_status = "R2R_VAL_SEEN_TRANSFER_GATE_FAIL"
    else:
        gates = {
            "utility_lower_95_positive": (
                aggregate["utility"]["scene_bootstrap_95pct"][0] > 0
            ),
            "success_point_nonnegative": aggregate["success"]["mean"] >= 0,
            "spl_point_nonnegative": aggregate["spl"]["mean"] >= 0,
            "success_or_spl_strictly_positive": (
                aggregate["success"]["mean"] > 0 or aggregate["spl"]["mean"] > 0
            ),
            "effective_actions_present": total_changed > 0,
        }
        passed_status = "R2R_VAL_UNSEEN_ADVANTAGE_PASS"
        failed_status = "R2R_VAL_UNSEEN_ADVANTAGE_FAIL"
    passed = all(gates.values())
    atomic_json(result_path, {
        "schema_version": f"revealnav-mf3zg-r2r-{split}-result/1",
        "status": passed_status if passed else failed_status,
        "controller_revision": "mf3zg",
        "episodes": len(pairs),
        "scenes": len({row["scene_id"] for row in pairs}),
        "aggregate_mf3zg_minus_etpr1": aggregate,
        "controller_activity": {
            "authorized": total_authorized,
            "actions_changed": total_changed,
        },
        "gates": gates,
        "per_episode": pairs,
        "threshold_or_model_tuning_on_r2r": False,
        "r2r_val_unseen_accessed": split == "val_unseen",
        "test_or_test_challenge_accessed": False,
    })
    return 0 if passed else 2


def monitor(split: str) -> int:
    path = SEEN_PROGRESS if split == "val_seen" else UNSEEN_PROGRESS
    if not path.is_file():
        print(json.dumps({"status": "NOT_STARTED", "progress": str(path.relative_to(ROOT))}))
        return 1
    print(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal-seen")
    sub.add_parser("seal-unseen")
    for command in ("run-seen", "run-unseen"):
        child = sub.add_parser(command)
        child.add_argument("--preflight", action="store_true")
        child.add_argument("--gpus", default="0,1")
        child.add_argument("--workers-per-gpu", type=int, default=1)
        child.add_argument("--resume", action="store_true")
    for command in ("verify-seen", "verify-unseen"):
        child = sub.add_parser(command)
        child.add_argument("--preflight", action="store_true")
    child = sub.add_parser("monitor")
    child.add_argument("--split", choices=("val_seen", "val_unseen"), default="val_seen")
    args = parser.parse_args()
    if args.command == "seal-seen":
        return seal_seen()
    if args.command == "seal-unseen":
        return seal_unseen()
    if args.command.startswith("run-"):
        split = "val_seen" if args.command == "run-seen" else "val_unseen"
        return _execute(
            split, args.preflight,
            tuple(int(value) for value in args.gpus.split(",") if value),
            args.workers_per_gpu, args.resume,
        )
    if args.command.startswith("verify-"):
        split = "val_seen" if args.command == "verify-seen" else "val_unseen"
        return _verify(split, args.preflight)
    return monitor(args.split)


if __name__ == "__main__":
    raise SystemExit(main())
