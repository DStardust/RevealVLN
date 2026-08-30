#!/usr/bin/env python3
"""Evaluate MF3X (MF3V widened train-only score band) on RxR val_seen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

from run_rxr_uad_paired_metrics_mf3 import selection


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/rxr_uad_mf3x_val_seen_worker.py"
FREEZE = ROOT / "artifacts/evaluation/mf3v_horizon_freeze_v1/MF3V_VAL_SEEN_FREEZE.json"
DIAGNOSTIC = ROOT / "artifacts/evaluation/mf3v_widened_band_shadow_diagnostic_v1/MF3V_WIDENED_BAND_SHADOW_DIAGNOSTIC.json"
DATASET = ROOT / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_seen/val_seen_guide.json.gz"
OUT = ROOT / "artifacts/evaluation/mf3x_uad_rxr_val_seen_v1"
PROTOCOL = OUT / "MF3X_RXR_VAL_SEEN_PROTOCOL.json"
PROGRESS = OUT / "MF3X_RXR_VAL_SEEN_PROGRESS.json"
RESULT = OUT / "MF3X_RXR_VAL_SEEN_RESULT.json"
METRICS = ("success", "spl", "ndtw", "sdtw")


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


def jobs(rows):
    return [(mode, row) for mode in ("baseline", "uncertainty", "ensemble") for row in rows]


def protocol_value():
    freeze = json.loads(FREEZE.read_text()); diagnostic = json.loads(DIAGNOSTIC.read_text())
    if freeze.get("status") != "MF3V_VAL_SEEN_FROZEN" or diagnostic.get("status") != "TRAIN_ONLY_CANDIDATE_PASS":
        raise RuntimeError("MF3X prerequisites missing")
    rows = selection()
    return {
        "schema_version": "revealnav-mf3x-rxr-val-seen-protocol/1",
        "status": "SEALED_BEFORE_MF3X_RXR_VAL_SEEN_TASK_METRICS",
        "scope": "RxR English val_seen MF3X train-calibrated widened band",
        "selection": rows, "runs": len(rows) * 3, "revision": "mf3x",
        "candidate": "MF3V horizon model with train-only q98.5-q99.75 score band",
        "threshold_tuned_on_val_seen": False, "public_unseen_accessed": False,
        "sources": {"freeze": sha256_file(FREEZE), "diagnostic": sha256_file(DIAGNOSTIC), "worker": sha256_file(WORKER), "dataset": sha256_file(DATASET)},
    }


def seal():
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value: raise RuntimeError("MF3X protocol drift")
    if not PROTOCOL.exists(): atomic_json(PROTOCOL, value)
    print(json.dumps({"status": value["status"], "runs": value["runs"]}, indent=2)); return 0


def execute(preflight, gpus, workers_per_gpu):
    if json.loads(PROTOCOL.read_text()) != protocol_value(): raise RuntimeError("MF3X protocol not sealed")
    rows = json.loads(PROTOCOL.read_text())["selection"]; rows = rows[:1] if preflight else rows; planned = jobs(rows)
    root = OUT / ("preflight" if preflight else "full")
    if root.exists(): raise RuntimeError(f"refusing to overwrite {root}")
    root.mkdir(parents=True); slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]; queue = list(planned); active = []; completed = []; started = time.time()
    while queue or active:
        while queue and slots:
            mode, row = queue.pop(0); gpu = slots.pop(0); name = f"{mode}_ep_{row['episode_id']}"; run_dir = root / "runs" / name; logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
            stdout = (logs / f"{name}.stdout").open("w"); stderr = (logs / f"{name}.stderr").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            process = subprocess.Popen([str(ROOT / ".envs/etpr1/bin/python"), str(WORKER), "--episode-id", row["episode_id"], "--mode", mode, "--run-dir", str(run_dir)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            active.append({"process": process, "mode": mode, "row": row, "gpu": gpu, "streams": (stdout, stderr)})
        atomic_json(PROGRESS, {"status": "RUNNING", "preflight": preflight, "total": len(planned), "completed": len(completed), "failed": sum(x["returncode"] != 0 for x in completed), "queued": len(queue), "active": [{"mode": x["mode"], "episode_id": x["row"]["episode_id"], "gpu": x["gpu"]} for x in active], "elapsed_s": round(time.time() - started, 1)})
        time.sleep(1)
        for item in list(active):
            code = item["process"].poll()
            if code is None: continue
            for stream in item["streams"]: stream.close()
            completed.append({"mode": item["mode"], "episode_id": item["row"]["episode_id"], "gpu": item["gpu"], "returncode": code}); active.remove(item); slots.append(item["gpu"]); slots.sort()
    failures = [x for x in completed if x["returncode"] != 0]
    atomic_json(PROGRESS, {"status": "COMPLETE" if not failures else "FAIL", "preflight": preflight, "total": len(planned), "completed": len(completed), "failed": len(failures), "queued": 0, "active": [], "elapsed_s": round(time.time() - started, 1)})
    return 0 if not failures else 2


def percentile(values, q): return sorted(values)[round((len(values) - 1) * q)]


def verify(preflight):
    protocol = protocol_value(); rows = protocol["selection"][:1] if preflight else protocol["selection"]; root = OUT / ("preflight" if preflight else "full"); summaries = {}
    for mode, row in jobs(rows):
        path = root / "runs" / f"{mode}_ep_{row['episode_id']}" / "RUN_SUMMARY.json"; value = json.loads(path.read_text())
        if value.get("status") != "PASS" or value.get("split") != "val_seen" or value.get("public_unseen_accessed") is not False or not isinstance(value.get("metrics"), dict): raise RuntimeError("MF3X worker boundary or metric failure")
        if not all(math.isfinite(float(value["metrics"][metric])) for metric in METRICS): raise RuntimeError("MF3X non-finite metric")
        if mode != "baseline" and value["executed_action_validation"]["all_equal"] is not True: raise RuntimeError("MF3X action mismatch")
        summaries[(mode, row["episode_id"])] = value
    if preflight:
        atomic_json(OUT / "MF3X_RXR_VAL_SEEN_PREFLIGHT.json", {"status": "PREFLIGHT_PASS", "runs": len(summaries)}); return 0
    per_scene = []
    for row in rows:
        episode = row["episode_id"]; b = summaries[("baseline", episode)]["metrics"]; t = summaries[("ensemble", episode)]["metrics"]; u = summaries[("uncertainty", episode)]["metrics"]
        delta = {metric: t[metric] - b[metric] for metric in METRICS}; delta["utility"] = .50 * delta["ndtw"] + .25 * delta["sdtw"] + .25 * delta["spl"]; delta["learned_minus_uncertainty_utility"] = .50 * (t["ndtw"] - u["ndtw"]) + .25 * (t["sdtw"] - u["sdtw"]) + .25 * (t["spl"] - u["spl"]); per_scene.append({"scene_id": row["scene_id"], "episode_id": episode, **delta})
    rng = random.Random(20260829); samples = []
    for _ in range(10000):
        sample = [per_scene[rng.randrange(len(per_scene))] for _ in per_scene]; samples.append(sum(x["utility"] for x in sample) / len(sample))
    aggregate = {metric: {"mean": sum(x[metric] for x in per_scene) / len(per_scene)} for metric in (*METRICS, "utility")}; aggregate["utility"]["scene_bootstrap_95pct"] = [percentile(samples, .025), percentile(samples, .975)]
    gates = {"utility_point_positive": aggregate["utility"]["mean"] > 0, "utility_lower_95_positive": aggregate["utility"]["scene_bootstrap_95pct"][0] > 0, "success_point_nonnegative": aggregate["success"]["mean"] >= 0, "spl_point_nonnegative": aggregate["spl"]["mean"] >= 0, "ndtw_point_nonnegative": aggregate["ndtw"]["mean"] >= 0, "learned_utility_exceeds_uncertainty": sum(x["learned_minus_uncertainty_utility"] for x in per_scene) / len(per_scene) > 0}
    passed = all(gates.values()); atomic_json(RESULT, {"schema_version": "revealnav-mf3x-rxr-val-seen-result/1", "status": "TASK_METRIC_GATE_PASS" if passed else "TASK_METRIC_GATE_FAIL", "aggregate_ensemble_minus_baseline": aggregate, "per_scene": per_scene, "gates": gates, "public_unseen_authorized": False, "promote_to_unseen": passed}); return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("seal"); run_parser = sub.add_parser("execute"); run_parser.add_argument("--preflight", action="store_true"); run_parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7"); run_parser.add_argument("--workers-per-gpu", type=int, default=2); verify_parser = sub.add_parser("verify"); verify_parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    if args.command == "seal": return seal()
    if args.command == "execute": return execute(args.preflight, tuple(int(x) for x in args.gpus.split(",") if x), args.workers_per_gpu)
    return verify(args.preflight)


if __name__ == "__main__": raise SystemExit(main())
