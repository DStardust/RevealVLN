#!/usr/bin/env python3
"""Run and assemble the sealed R2R-train MF3ZK switch collection.

Collection is outcome-blind: workers may execute only the frozen MF3ZF
proposal, and task metrics are read solely by ``assemble`` after all workers
finish.  The controller is never run on a public evaluation split here.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
PROTOCOL = ROOT / "artifacts/training/mf3zk_joint_v1/MF3ZK_JOINT_PROTOCOL.json"
OUT = ROOT / "artifacts/training/mf3zk_joint_v1/r2r_collection"
SELECTION = OUT / "MF3ZK_R2R_COLLECTION_SELECTION.json"
PROGRESS = OUT / "MF3ZK_R2R_COLLECTION_PROGRESS.json"
MANIFEST = OUT / "MF3ZK_R2R_DIRECT_SWITCH_MANIFEST.json"
WORKER = ROOT / "scripts/r2r_mf3zk_train_collection_worker.py"
BASE_RUNS = ROOT / "artifacts/phase1/r2r_train_net_advantage/full/runs"
BASELINE_COMPLETION_RUNS = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_baseline_completion/runs"
)
BASELINE_COMPLETION_PROTOCOL = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_baseline_completion/"
    "MF3ZK_R2R_BASELINE_COMPLETION_PROTOCOL_V2.json"
)
COLLECTION_GATE = ROOT / (
    "artifacts/training/mf3zf_expanded_collection_v1/"
    "MF3ZF_COLLECTION_GATE.json"
)
CORE_THRESHOLD = 2.1383049488067627
LOWER_THRESHOLD = 1.6816482543945312
UPPER_THRESHOLD = 2.6732332706451416
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


def _load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    if not (
        value.get("status") == "SEALED_BEFORE_MF3ZK_JOINT_TRAINING"
        and value.get("revision") == "mf3zk"
        and value.get("r2r_train", {}).get("collection_routes")
        and value.get("public_split_access", {}).get("r2r_val_unseen") is False
    ):
        raise RuntimeError("MF3ZK joint protocol is not sealed for collection")
    return value


def select() -> int:
    protocol = _load_protocol()
    routes = protocol["r2r_train"]["collection_routes"]
    value = {
        "schema_version": "revealnav-mf3zk-r2r-collection-selection/1",
        "status": "SEALED_BEFORE_R2R_MF3ZK_COLLECTION",
        "split": "train",
        "revision": "mf3zk",
        "proposal_revision": "mf3zf",
        "selection_rule": "the first 1200 deterministic trajectory representatives from fit scenes in the sealed joint protocol",
        "routes": routes,
        "counts": {
            "routes": len(routes),
            "scenes": len({row["scene_id"] for row in routes}),
        },
        "sources": {
            "joint_protocol": {
                "path": str(PROTOCOL.relative_to(ROOT)),
                "bytes": PROTOCOL.stat().st_size,
                "sha256": sha256_file(PROTOCOL),
            },
            "worker": {
                "path": str(WORKER.relative_to(ROOT)),
                "bytes": WORKER.stat().st_size,
                "sha256": sha256_file(WORKER),
            },
            "collection_gate": {
                "path": str(COLLECTION_GATE.relative_to(ROOT)),
                "bytes": COLLECTION_GATE.stat().st_size,
                "sha256": sha256_file(COLLECTION_GATE),
            },
        },
        "task_metrics_used_for_selection": False,
        "unseen_or_test_read": False,
    }
    if SELECTION.exists() and json.loads(SELECTION.read_text()) != value:
        raise RuntimeError("R2R MF3ZK collection selection drift")
    if not SELECTION.exists():
        atomic_json(SELECTION, value)
    print(json.dumps({
        "status": value["status"], "routes": len(routes),
        "scenes": value["counts"]["scenes"],
    }, indent=2, sort_keys=True))
    return 0


def _summary_valid(path: Path, episode_id: str) -> tuple[bool, dict | None]:
    if not path.is_file() or path.is_symlink():
        return False, None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return False, None
    valid = (
        value.get("status") == "PASS"
        and value.get("episode_id") == str(episode_id)
        and value.get("split") == "train"
        and value.get("revision") == "mf3zk"
        and value.get("proposal_revision") == "mf3zf"
        and value.get("collection_only") is True
        and value.get("task_metric_payload_read") is False
        and value.get("future_frames_used") == 0
        and value.get("unseen_or_test_read") is False
        and value.get("executed_action_validation", {}).get("all_equal") is True
        and int(value.get("changed_actions", -1)) in (0, 1)
    )
    return valid, value if valid else None


def run(gpus: tuple[int, ...], workers_per_gpu: int, resume: bool) -> int:
    if any(gpu not in (0, 1) for gpu in gpus):
        raise ValueError("MF3ZK collection is restricted to free GPUs 0 and 1")
    if workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    selection = json.loads(SELECTION.read_text())
    if selection["sources"]["worker"]["sha256"] != sha256_file(WORKER):
        raise RuntimeError("collection worker changed after selection seal")
    routes = selection["routes"]
    runs = OUT / "runs"
    logs = OUT / "logs"
    runs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    queue = []
    recovered = []
    for row in routes:
        episode = str(row["episode_id"])
        run_dir = runs / f"ep_{episode}"
        valid, _ = _summary_valid(run_dir / "RUN_SUMMARY.json", episode)
        if resume and valid:
            recovered.append({"episode_id": episode, "returncode": 0, "recovered": True})
            continue
        if run_dir.exists():
            # A previous partial run is retained for provenance under a
            # timestamped sibling; no existing evidence is overwritten.
            stale = OUT / "interrupted" / f"ep_{episode}_{int(time.time())}"
            stale.parent.mkdir(parents=True, exist_ok=True)
            os.replace(run_dir, stale)
        queue.append(row)
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    active = []
    completed = list(recovered)
    started = time.time()
    last_write = 0.0
    while queue or active:
        while queue and slots:
            row = queue.pop(0)
            gpu = slots.pop(0)
            episode = str(row["episode_id"])
            run_dir = runs / f"ep_{episode}"
            stdout = (logs / f"ep_{episode}.stdout").open("w")
            stderr = (logs / f"ep_{episode}.stderr").open("w")
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
                "--run-dir", str(run_dir),
            ], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            active.append({
                "process": process, "gpu": gpu, "episode_id": episode,
                "streams": (stdout, stderr),
            })
        now = time.time()
        if now - last_write >= 5.0:
            elapsed = now - started
            done = len(completed)
            rate = done / elapsed if elapsed > 0 else 0.0
            atomic_json(PROGRESS, {
                "status": "RUNNING", "total": len(routes),
                "completed": done,
                "valid_completed": sum(
                    _summary_valid(runs / f"ep_{x['episode_id']}" / "RUN_SUMMARY.json", x["episode_id"])[0]
                    for x in routes if (runs / f"ep_{x['episode_id']}" / "RUN_SUMMARY.json").is_file()
                ),
                "failed": sum(x.get("returncode") != 0 for x in completed),
                "queued": len(queue),
                "active": [
                    {"episode_id": x["episode_id"], "gpu": x["gpu"]}
                    for x in active
                ],
                "elapsed_s": round(elapsed, 1),
                "eta_s": None if rate == 0 else round((len(routes) - done) / rate, 1),
                "monitor_command": f"{PYTHON} scripts/run_mf3zk_r2r_collection.py monitor",
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
                "episode_id": item["episode_id"], "gpu": item["gpu"],
                "returncode": code,
            })
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


def _read_metrics(run_dir: Path, episode_id: str) -> tuple[Path, dict]:
    matches = list(run_dir.rglob("stats_ep_ckpt_270_train_r0_w1.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one train metric file for {episode_id}")
    value = json.loads(matches[0].read_text())
    metrics = value.get(str(episode_id))
    if not isinstance(metrics, dict):
        raise RuntimeError(f"missing train metrics for {episode_id}")
    for key in ("success", "spl", "ndtw", "sdtw"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"non-finite metric {key} for {episode_id}")
    return matches[0], {key: float(metrics[key]) for key in ("success", "spl", "ndtw", "sdtw")}


def _baseline_source(episode_id: str) -> tuple[Path, str] | None:
    """Resolve an exact-episode native baseline without cross-episode pairing."""
    candidates = (
        (BASE_RUNS / f"ep_{episode_id}", "preexisting_train_baseline"),
        (BASELINE_COMPLETION_RUNS / f"ep_{episode_id}", "mf3zk_completed_train_baseline"),
    )
    for directory, source_type in candidates:
        if not directory.is_dir() or directory.is_symlink():
            continue
        stats = list(directory.rglob("stats_ep_ckpt_270_train_r0_w1.json"))
        if len(stats) == 1:
            if source_type == "mf3zk_completed_train_baseline":
                summary_path = directory / "RUN_SUMMARY.json"
                if not summary_path.is_file() or summary_path.is_symlink():
                    continue
                summary = json.loads(summary_path.read_text())
                if not (
                    summary.get("status") == "PASS"
                    and str(summary.get("episode_id")) == str(episode_id)
                    and summary.get("split") == "train"
                    and summary.get("mode") == "baseline"
                    and summary.get("public_unseen_accessed") is False
                    and summary.get("controller") is None
                ):
                    continue
            return directory, source_type
    return None


def _utility(metrics: dict) -> float:
    return sum(UTILITY_WEIGHTS[key] * metrics[key] for key in UTILITY_WEIGHTS)


def _trace(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing trace: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _prefix_matches(baseline: list[dict], treatment: list[dict], step: int) -> bool:
    if step < 0 or len(baseline) <= step or len(treatment) <= step:
        return False
    keys = ("act", "ghost_vp", "cur_vp", "front_vp", "back_path_len")
    for left, right in zip(baseline[:step], treatment[:step]):
        if any(left.get(key) != right.get(key) for key in keys):
            return False
    return True


def _feature_evidence(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or ROOT not in path.resolve().parents:
        raise RuntimeError("intervention feature provenance failure")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"instruction", "checkpoint", "native", "alternative"}:
            raise RuntimeError("intervention feature schema drift")
        for key in payload.files:
            value = payload[key]
            if value.shape != (768,) or not np.isfinite(value).all():
                raise RuntimeError("intervention feature value drift")
    return {
        "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def assemble() -> int:
    selection = json.loads(SELECTION.read_text())
    rows = []
    skipped = []
    for source in selection["routes"]:
        episode = str(source["episode_id"])
        treatment_dir = OUT / "runs" / f"ep_{episode}"
        summary_path = treatment_dir / "RUN_SUMMARY.json"
        valid, summary = _summary_valid(summary_path, episode)
        if not valid:
            skipped.append({"episode_id": episode, "reason": "invalid_summary"})
            continue
        controller = summary.get("controller") or {}
        changed = [
            row for row in _trace(treatment_dir / "controller_trace.jsonl")
            if row.get("action_changed") is True
        ]
        if len(changed) != 1 or int(controller.get("actions_changed", -1)) != 1:
            skipped.append({"episode_id": episode, "reason": "no_exact_one_switch"})
            continue
        decision = changed[0]
        score = float(decision.get("policy_risk_adjusted_score"))
        if not LOWER_THRESHOLD < score <= UPPER_THRESHOLD:
            raise RuntimeError(f"proposal score outside sealed band: {episode}")
        tier = "core" if score > CORE_THRESHOLD else "expansion"
        baseline_source = _baseline_source(episode)
        if baseline_source is None:
            skipped.append({"episode_id": episode, "reason": "baseline_missing"})
            continue
        baseline_dir, baseline_source_type = baseline_source
        baseline_stats, baseline_metrics = _read_metrics(baseline_dir, episode)
        treatment_stats, treatment_metrics = _read_metrics(treatment_dir, episode)
        baseline_trace = _trace(baseline_dir / "base_trace.jsonl")
        treatment_trace = _trace(treatment_dir / "base_trace.jsonl")
        step = int(decision["step"])
        if not _prefix_matches(baseline_trace, treatment_trace, step):
            skipped.append({"episode_id": episode, "reason": "baseline_prefix_mismatch"})
            continue
        feature_path = treatment_dir / "intervention_feature.npz"
        feature = _feature_evidence(feature_path)
        delta = {
            key: treatment_metrics[key] - baseline_metrics[key]
            for key in ("success", "spl", "ndtw", "sdtw")
        }
        delta["utility"] = _utility(treatment_metrics) - _utility(baseline_metrics)
        rows.append({
            "row_index": len(rows), "dataset": "R2R", "split": "train",
            "tier": tier, "episode_id": episode,
            "trajectory_id": source["trajectory_id"], "scene_id": source["scene_id"],
            "decision_step": step, "mf3v_score": score,
            "decision": {
                key: decision.get(key) for key in (
                    "step", "minimum_top2_advantage", "median_top2_advantage",
                    "robust_top2_advantage", "ensemble_mad",
                    "cold_start_floor_ratio", "cold_start_relative_mad",
                    "policy_risk_adjusted_score", "native_margin",
                    "current_local_action_ids",
                )
            },
            "delta": delta,
            "baseline_metrics": baseline_metrics,
            "treatment_metrics": treatment_metrics,
            "feature": feature,
            "baseline_stats": {
                "path": str(baseline_stats.relative_to(ROOT)),
                "bytes": baseline_stats.stat().st_size,
                "sha256": sha256_file(baseline_stats),
            },
            "baseline_source": baseline_source_type,
            "treatment_stats": {
                "path": str(treatment_stats.relative_to(ROOT)),
                "bytes": treatment_stats.stat().st_size,
                "sha256": sha256_file(treatment_stats),
            },
            "run_summary": {
                "path": str(summary_path.relative_to(ROOT)),
                "bytes": summary_path.stat().st_size,
                "sha256": sha256_file(summary_path),
            },
            "controller_trace": {
                "path": str((treatment_dir / "controller_trace.jsonl").relative_to(ROOT)),
                "bytes": (treatment_dir / "controller_trace.jsonl").stat().st_size,
                "sha256": sha256_file(treatment_dir / "controller_trace.jsonl"),
            },
            "baseline_prefix_verified": True,
            "future_frames_used": 0,
        })
    counts = {
        "selected_routes": len(selection["routes"]),
        "usable_pairs": len(rows),
        "skipped": len(skipped),
        "scenes": len({row["scene_id"] for row in rows}),
        "core": sum(row["tier"] == "core" for row in rows),
        "expansion": sum(row["tier"] == "expansion" for row in rows),
        "positive_utility": sum(row["delta"]["utility"] > 0 for row in rows),
        "negative_utility": sum(row["delta"]["utility"] < 0 for row in rows),
    }
    if counts["usable_pairs"] < 24 or counts["core"] < 8 or counts["expansion"] < 8:
        status = "R2R_DIRECT_SWITCH_RETURN_DATASET_INSUFFICIENT"
    else:
        status = "R2R_DIRECT_SWITCH_RETURN_DATASET_READY"
    value = {
        "schema_version": "revealnav-mf3zk-r2r-direct-switch-manifest/1",
        "status": status, "revision": "mf3zk", "split": "train",
        "proposal_revision": "mf3zf", "records": rows,
        "counts": counts,
        "mean_utility_delta": (
            sum(row["delta"]["utility"] for row in rows) / len(rows)
            if rows else None
        ),
        "skipped": skipped,
        "selection_sha256": sha256_file(SELECTION),
        "collection_gate_sha256": sha256_file(COLLECTION_GATE),
        "unseen_or_test_read": False,
        "task_metrics_read_only_during_assembly": True,
        "baseline_completion_protocol": (
            {
                "path": str(BASELINE_COMPLETION_PROTOCOL.relative_to(ROOT)),
                "bytes": BASELINE_COMPLETION_PROTOCOL.stat().st_size,
                "sha256": sha256_file(BASELINE_COMPLETION_PROTOCOL),
            }
            if BASELINE_COMPLETION_PROTOCOL.is_file() else None
        ),
        "label_definition": {
            "paired_task_return": "treatment minus deterministic ETP-R1 baseline",
            "utility": "0.50*nDTW + 0.25*SDTW + 0.25*SPL",
            "exactly_one_changed_action": True,
            "train_only": True,
        },
    }
    if MANIFEST.exists():
        raise RuntimeError("refusing to overwrite assembled MF3ZK manifest")
    atomic_json(MANIFEST, value)
    print(json.dumps({"status": status, "counts": counts,
                      "mean_utility_delta": value["mean_utility_delta"]},
                     indent=2, sort_keys=True))
    return 0 if status.endswith("READY") else 2


def monitor() -> int:
    if not PROGRESS.is_file():
        print(json.dumps({"status": "NOT_STARTED", "progress": str(PROGRESS.relative_to(ROOT))}))
        return 1
    print(PROGRESS.read_text())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--gpus", default="0,1")
    run_parser.add_argument("--workers-per-gpu", type=int, default=3)
    run_parser.add_argument("--resume", action="store_true")
    sub.add_parser("assemble")
    sub.add_parser("monitor")
    args = parser.parse_args()
    if args.command == "select":
        return select()
    if args.command == "run":
        return run(tuple(int(value) for value in args.gpus.split(",") if value),
                   args.workers_per_gpu, args.resume)
    if args.command == "assemble":
        return assemble()
    return monitor()


if __name__ == "__main__":
    raise SystemExit(main())
