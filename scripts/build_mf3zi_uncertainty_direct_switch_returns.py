#!/usr/bin/env python3
"""Build exact-return labels for one-shot native-margin interventions.

MF3ZI uses the already frozen MF3ZG learned proposal as its primary path and
adds a separately trained, one-shot safety gate for the native-margin
uncertainty candidate.  This builder is deliberately train-only: it selects
the first sealed MF3V margin crossing from the existing online feature
manifest, rolls out exactly one uncertainty switch, and records the paired
final return.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "artifacts/phase1/mf3q_final_rank23/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
MF3V_GATE = ROOT / (
    "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/"
    "MF3V_SHADOW_GATE.json"
)
WORKER = ROOT / "scripts/rxr_uad_controller_worker_mf3.py"
OUT = ROOT / "artifacts/phase1/mf3zi_uncertainty_direct_switch_returns_v1"
SELECTION = OUT / "MF3ZI_UNCERTAINTY_SELECTION.json"
PROGRESS = OUT / "MF3ZI_UNCERTAINTY_PROGRESS.json"
MANIFEST = OUT / "MF3ZI_UNCERTAINTY_MANIFEST.json"
EXPECTED_EPISODES = 126
EXPECTED_SCENES = 46
SCHEMA = "revealnav-mf3zi-uncertainty-direct-switch"


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_metrics(record: dict) -> tuple[Path, dict]:
    feature = (ROOT / record["path"]).resolve()
    if ROOT not in feature.parents or feature.is_symlink() or not feature.is_file():
        raise RuntimeError("MF3ZI source feature path is unsafe")
    stats = list(feature.parent.rglob("stats_ep_ckpt_1320_train_r0_w1.json"))
    if len(stats) != 1:
        raise RuntimeError("MF3ZI baseline stats cardinality drift")
    payload = json.loads(stats[0].read_text())
    metrics = payload.get(str(record["episode_id"]))
    if not isinstance(metrics, dict):
        raise RuntimeError("MF3ZI baseline episode metric missing")
    for key in ("success", "spl", "ndtw", "sdtw"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError("MF3ZI baseline metric is non-finite")
    return stats[0], metrics


def first_margin_crossing(record: dict, threshold: float) -> tuple[int, float] | None:
    feature = (ROOT / record["path"]).resolve()
    with np.load(feature, allow_pickle=False) as source:
        required = {
            "candidate_mask", "native_index", "native_scores",
        }
        if not required.issubset(source.files):
            raise RuntimeError("MF3ZI source feature keys drift")
        mask = source["candidate_mask"]
        native_index = source["native_index"]
        scores = source["native_scores"]
        if (
            mask.dtype != np.bool_ or mask.ndim != 2
            or native_index.shape != (mask.shape[0],)
            or scores.shape != mask.shape
            or not np.isfinite(scores[mask]).all()
        ):
            raise RuntimeError("MF3ZI source feature axes or values drift")
        for step, native_value in enumerate(native_index):
            native = int(native_value)
            indices = np.flatnonzero(mask[step])
            if (
                len(indices) < 2 or native < 0 or native not in indices
                or int(indices[np.argmax(scores[step, indices])]) != native
            ):
                continue
            alternatives = indices[indices != native]
            margin = float(scores[step, native] - np.max(scores[step, alternatives]))
            if not math.isfinite(margin):
                raise RuntimeError("MF3ZI native margin is non-finite")
            if margin <= threshold:
                return step, margin
    return None


def select() -> int:
    source = json.loads(SOURCE.read_text())
    gate = json.loads(MF3V_GATE.read_text())
    if (
        source.get("status") != "PASS" or len(source.get("records", [])) != 1303
        or gate.get("status") != "SHADOW_GATE_PASS"
        or gate.get("task_metric_run_authorized") is not True
        or gate.get("public_unseen_authorized") is not False
    ):
        raise RuntimeError("MF3ZI source or MF3V gate is not authorized")
    threshold = float(gate["exact_budget_control"]["native_margin_max"])
    selected = []
    for record in source["records"]:
        if record.get("split") != "fit":
            continue
        crossing = first_margin_crossing(record, threshold)
        if crossing is None:
            continue
        stats, metrics = baseline_metrics(record)
        selected.append({
            "episode_id": str(record["episode_id"]),
            "scene_id": str(record["scene_id"]),
            "decision_step": crossing[0],
            "native_margin": crossing[1],
            "baseline_metrics": metrics,
            "baseline_stats": {
                "path": str(stats.relative_to(ROOT)),
                "bytes": stats.stat().st_size,
                "sha256": sha256_file(stats),
            },
            "source_feature": {
                "path": record["path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            },
        })
    if (
        len(selected) != EXPECTED_EPISODES
        or len({row["scene_id"] for row in selected}) != EXPECTED_SCENES
    ):
        raise RuntimeError(
            f"MF3ZI deterministic selection drift: {len(selected)} episodes, "
            f"{len({row['scene_id'] for row in selected})} scenes"
        )
    value = {
        "schema_version": f"{SCHEMA}-selection/1",
        "status": "SEALED_BEFORE_UNCERTAINTY_ROLLOUTS",
        "split": "train",
        "selection": selected,
        "counts": {"episodes": len(selected), "scenes": len({r["scene_id"] for r in selected})},
        "uncertainty_rule": {
            "native_margin_max": threshold,
            "first_crossing_only": True,
            "one_shot": True,
        },
        "sources": {
            "online_manifest": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256_file(SOURCE)},
            "mf3v_gate": {"path": str(MF3V_GATE.relative_to(ROOT)), "sha256": sha256_file(MF3V_GATE)},
            "worker": {"path": str(WORKER.relative_to(ROOT)), "sha256": sha256_file(WORKER)},
        },
        "unseen_or_test_read": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if SELECTION.exists():
        if json.loads(SELECTION.read_text()) != value:
            raise RuntimeError("MF3ZI selection already sealed with different content")
    else:
        atomic_json(SELECTION, value)
    print(json.dumps(value["counts"], indent=2, sort_keys=True))
    return 0


def archive_interrupted(path: Path) -> None:
    if not path.exists():
        return
    parent = OUT / "interrupted"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / path.name
    suffix = 1
    while destination.exists():
        destination = parent / f"{path.name}_{suffix}"
        suffix += 1
    os.replace(path, destination)


def run(gpus: tuple[int, ...], workers_per_gpu: int, resume: bool) -> int:
    protocol = json.loads(SELECTION.read_text())
    if protocol["sources"]["worker"]["sha256"] != sha256_file(WORKER):
        raise RuntimeError("MF3ZI worker changed after selection seal")
    root = OUT / "runs"
    if root.exists() and not resume:
        raise RuntimeError("MF3ZI runs exist; pass --resume explicitly")
    root.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume:
        for directory in sorted(root.glob("ep_*")):
            summary = directory / "RUN_SUMMARY.json"
            if summary.is_file() and json.loads(summary.read_text()).get("status") == "PASS":
                done.add(directory.name.removeprefix("ep_"))
            else:
                archive_interrupted(directory)
    queue = [row for row in protocol["selection"] if row["episode_id"] not in done]
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    if not slots:
        raise ValueError("at least one GPU slot is required")
    active = []
    completed = []
    started = time.time()
    while queue or active:
        while queue and slots:
            row = queue.pop(0)
            gpu = slots.pop(0)
            episode = row["episode_id"]
            directory = root / f"ep_{episode}"
            logs = OUT / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            stdout = (logs / f"ep_{episode}.stdout").open("w")
            stderr = (logs / f"ep_{episode}.stderr").open("w")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "REVEALNAV_MF3_UNCERTAINTY_ONE_SHOT": "1",
                "REVEALNAV_MF3_UNCERTAINTY_FEATURE": str(directory / "intervention_feature.npz"),
            }
            process = subprocess.Popen([
                str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
                "--episode-id", episode, "--mode", "uncertainty",
                "--revision", "mf3v", "--split", "train",
                "--run-dir", str(directory),
            ], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            active.append({"process": process, "row": row, "gpu": gpu, "streams": (stdout, stderr)})
        atomic_json(PROGRESS, {
            "status": "RUNNING", "total": len(protocol["selection"]),
            "completed": len(done) + len(completed),
            "failed": sum(item["returncode"] != 0 for item in completed),
            "queued": len(queue), "active": [
                {"episode_id": item["row"]["episode_id"], "gpu": item["gpu"]}
                for item in active
            ], "elapsed_s": round(time.time() - started, 1),
        })
        time.sleep(1)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({"episode_id": item["row"]["episode_id"], "returncode": code})
            slots.append(item["gpu"]); slots.sort(); active.remove(item)
    failures = [item for item in completed if item["returncode"] != 0]
    atomic_json(PROGRESS, {
        "status": "COMPLETE" if not failures else "FAIL",
        "total": len(protocol["selection"]), "completed": len(done) + len(completed),
        "failed": len(failures), "queued": 0, "active": [],
        "elapsed_s": round(time.time() - started, 1),
    })
    return 0 if not failures else 2


def utility(metrics: dict) -> float:
    return 0.50 * float(metrics["ndtw"]) + 0.25 * float(metrics["sdtw"]) + 0.25 * float(metrics["spl"])


def assemble() -> int:
    protocol = json.loads(SELECTION.read_text())
    rows = []
    for source in protocol["selection"]:
        episode = source["episode_id"]
        run = OUT / "runs" / f"ep_{episode}"
        summary_path = run / "RUN_SUMMARY.json"
        trace_path = run / "controller_trace.jsonl"
        feature = run / "intervention_feature.npz"
        if not (summary_path.is_file() and trace_path.is_file() and feature.is_file()):
            raise RuntimeError(f"MF3ZI missing rollout evidence: {episode}")
        summary = json.loads(summary_path.read_text())
        trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
        changed = [row for row in trace if row.get("action_changed") is True]
        if not (
            summary.get("status") == "PASS" and summary.get("revision") == "mf3v"
            and summary.get("mode") == "uncertainty" and summary.get("split") == "train"
            and summary.get("public_unseen_accessed") is False
            and summary.get("controller", {}).get("actions_changed") == 1
            and len(changed) == 1 and feature.stat().st_size > 0
        ):
            raise RuntimeError(f"MF3ZI one-shot boundary failure: {episode}")
        if changed[0]["step"] < 0 or not math.isfinite(float(changed[0]["native_margin"])):
            raise RuntimeError(f"MF3ZI changed decision is malformed: {episode}")
        stats = (ROOT / source["baseline_stats"]["path"]).resolve()
        if not (
            ROOT in stats.parents and stats.is_file() and not stats.is_symlink()
            and stats.stat().st_size == source["baseline_stats"]["bytes"]
            and sha256_file(stats) == source["baseline_stats"]["sha256"]
        ):
            raise RuntimeError(f"MF3ZI baseline provenance drift: {episode}")
        with np.load(feature, allow_pickle=False) as payload:
            if set(payload.files) != {"instruction", "checkpoint", "native", "alternative"}:
                raise RuntimeError("MF3ZI intervention feature keys drift")
            if any(payload[key].shape != (768,) or not np.isfinite(payload[key]).all() for key in payload.files):
                raise RuntimeError("MF3ZI intervention feature values drift")
        treatment = summary["metrics"]
        baseline = source["baseline_metrics"]
        delta = {key: float(treatment[key]) - float(baseline[key]) for key in ("success", "spl", "ndtw", "sdtw")}
        delta["utility"] = utility(treatment) - utility(baseline)
        rows.append({
            "row_index": len(rows), "episode_id": episode, "scene_id": source["scene_id"],
            "decision_step": int(changed[0]["step"]), "native_margin": float(changed[0]["native_margin"]),
            "delta": delta, "decision": {key: changed[0][key] for key in ("step", "native_margin", "current_local_action_ids")},
            "baseline_metrics": baseline, "treatment_metrics": treatment,
            "feature": {"path": str(feature.relative_to(ROOT)), "bytes": feature.stat().st_size, "sha256": sha256_file(feature)},
            "run_summary": {"path": str(summary_path.relative_to(ROOT)), "bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
            "controller_trace": {"path": str(trace_path.relative_to(ROOT)), "bytes": trace_path.stat().st_size, "sha256": sha256_file(trace_path)},
        })
    if len(rows) != EXPECTED_EPISODES:
        raise RuntimeError("MF3ZI assembled row count drift")
    value = {
        "schema_version": f"{SCHEMA}-manifest/1",
        "status": "UNCERTAINTY_DIRECT_SWITCH_RETURN_DATASET_READY",
        "records": rows,
        "counts": {
            "pairs": len(rows), "scenes": len({r["scene_id"] for r in rows}),
            "positive_utility": sum(r["delta"]["utility"] > 0 for r in rows),
            "negative_utility": sum(r["delta"]["utility"] < 0 for r in rows),
            "ties": sum(abs(r["delta"]["utility"]) <= 1e-8 for r in rows),
        },
        "mean_utility_delta": sum(r["delta"]["utility"] for r in rows) / len(rows),
        "selection_sha256": sha256_file(SELECTION),
        "sources": {
            "selection": {"path": str(SELECTION.relative_to(ROOT)), "sha256": sha256_file(SELECTION)},
            "worker": {"path": str(WORKER.relative_to(ROOT)), "sha256": sha256_file(WORKER)},
        },
        "unseen_or_test_read": False,
    }
    atomic_json(MANIFEST, value)
    print(json.dumps({"status": value["status"], "counts": value["counts"], "mean_utility_delta": value["mean_utility_delta"]}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--gpus", default="0,1")
    run_parser.add_argument("--workers-per-gpu", type=int, default=1)
    run_parser.add_argument("--resume", action="store_true")
    sub.add_parser("assemble")
    args = parser.parse_args()
    if args.command == "select":
        return select()
    if args.command == "run":
        return run(tuple(int(v) for v in args.gpus.split(",") if v), args.workers_per_gpu, args.resume)
    return assemble()


if __name__ == "__main__":
    raise SystemExit(main())
