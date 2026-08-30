#!/usr/bin/env python3
"""Run the frozen MF3V on a scene-stratified RxR val_unseen pilot."""

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
WORKER = ROOT / "scripts/rxr_uad_mf3v_unseen_worker.py"
DATASET = ROOT / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide.json.gz"
GROUND_TRUTH = ROOT / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide_gt.json.gz"
TRAIN_DATASET = ROOT / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
FREEZE = ROOT / "artifacts/evaluation/mf3v_horizon_freeze_v1/MF3V_VAL_SEEN_FREEZE.json"
OUT = ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_unseen_pilot_v2"
PROTOCOL = OUT / "MF3V_RXR_VAL_UNSEEN_PROTOCOL.json"
PROGRESS = OUT / "MF3V_RXR_VAL_UNSEEN_PROGRESS.json"
RESULT = OUT / "MF3V_RXR_VAL_UNSEEN_RESULT.json"
SELECTION_SALT = "revealnav-mf3v-rxr-val-unseen-scene-pilot/1"
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


def read_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def scene_id(row: dict) -> str:
    return Path(row["scene_id"]).stem


def select_rows() -> tuple[list[dict], dict]:
    payload = read_gzip_json(DATASET)
    episodes = payload.get("episodes", [])
    if len(episodes) != 11006 or len({str(row["episode_id"]) for row in episodes}) != len(episodes):
        raise RuntimeError("RxR val_unseen inventory drift")
    english = [row for row in episodes if row["instruction"]["language"] in ("en-US", "en-IN")]
    if len(english) != 3669:
        raise RuntimeError("RxR val_unseen English inventory drift")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in english:
        grouped[scene_id(row)].append(row)
    if len(grouped) != 11:
        raise RuntimeError("RxR val_unseen scene inventory drift")
    train_scenes = {scene_id(row) for row in read_gzip_json(TRAIN_DATASET).get("episodes", [])}
    if train_scenes & set(grouped):
        raise RuntimeError("train and val_unseen scenes overlap")
    gt = read_gzip_json(GROUND_TRUTH)
    rows = []
    for scene in sorted(grouped):
        row = min(
            grouped[scene],
            key=lambda item: hashlib.sha256(
                f"{SELECTION_SALT}:{scene}:{item['episode_id']}".encode()
            ).hexdigest(),
        )
        episode_id = str(row["episode_id"])
        if episode_id not in gt:
            raise RuntimeError(f"missing ground truth for {episode_id}")
        scene_root = ROOT / f"third_party/ETP-R1/data/scene_datasets/mp3d/{scene}"
        for suffix in (".glb", ".navmesh"):
            asset = scene_root / f"{scene}{suffix}"
            if not asset.is_file() or asset.is_symlink():
                raise RuntimeError(f"missing or unsafe unseen asset: {asset}")
        rows.append({
            "scene_id": scene,
            "episode_id": episode_id,
            "language": row["instruction"]["language"],
            "selection_digest": hashlib.sha256(
                f"{SELECTION_SALT}:{scene}:{episode_id}".encode()
            ).hexdigest(),
        })
    return rows, {
        "all_guide_episodes": len(episodes),
        "english_guide_episodes": len(english),
        "unseen_scenes": len(grouped),
        "selected_episodes": len(rows),
        "train_scenes": len(train_scenes),
    }


def protocol_value() -> dict:
    freeze = json.loads(FREEZE.read_text())
    if freeze.get("status") != "MF3V_VAL_SEEN_FROZEN":
        raise RuntimeError("MF3V freeze is missing")
    if freeze.get("deployment_boundary", {}).get("val_unseen_authorized") is not False:
        raise RuntimeError("freeze boundary drift")
    rows, counts = select_rows()
    return {
        "schema_version": "revealnav-mf3v-rxr-val-unseen-protocol/1",
        "status": "SEALED_BEFORE_MF3V_RXR_VAL_UNSEEN_PILOT",
        "scope": "scene-stratified RxR English val_unseen engineering pilot",
        "selection_salt": SELECTION_SALT,
        "selection": rows,
        "counts": counts,
        "modes": ["baseline", "uncertainty", "ensemble"],
        "runs": len(rows) * 3,
        "revision": "mf3v",
        "threshold_tuned_on_val_unseen": False,
        "future_observations_used_online": False,
        "test_or_test_challenge_accessed": False,
        "paper_result": False,
        "sources": {
            "freeze": sha256_file(FREEZE),
            "worker": sha256_file(WORKER),
            "dataset": sha256_file(DATASET),
            "ground_truth": sha256_file(GROUND_TRUTH),
            "train_dataset": sha256_file(TRAIN_DATASET),
        },
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("unseen protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({"status": value["status"], "runs": value["runs"], "scenes": value["counts"]["unseen_scenes"]}, indent=2))
    return 0


def jobs(rows: list[dict]) -> list[tuple[str, dict]]:
    return [(mode, row) for mode in ("baseline", "uncertainty", "ensemble") for row in rows]


def run(preflight: bool, gpus: tuple[int, ...], workers_per_gpu: int) -> int:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol != protocol_value():
        raise RuntimeError("unseen protocol is not sealed")
    rows = protocol["selection"][:1] if preflight else protocol["selection"]
    planned = jobs(rows)
    root = OUT / ("preflight" if preflight else "full")
    if root.exists():
        raise RuntimeError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    queue = list(planned)
    active = []
    completed = []
    started = time.time()
    while queue or active:
        while queue and slots:
            mode, row = queue.pop(0)
            gpu = slots.pop(0)
            name = f"{mode}_ep_{row['episode_id']}"
            run_dir = root / "runs" / name
            logs = root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            stdout = (logs / f"{name}.stdout").open("w")
            stderr = (logs / f"{name}.stderr").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            process = subprocess.Popen(
                [str(ROOT / ".envs/etpr1/bin/python"), str(WORKER), "--episode-id", row["episode_id"], "--mode", mode, "--run-dir", str(run_dir)],
                cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
            )
            active.append({"process": process, "mode": mode, "row": row, "gpu": gpu, "streams": (stdout, stderr)})
        atomic_json(PROGRESS, {
            "status": "RUNNING", "preflight": preflight, "total": len(planned),
            "completed": len(completed), "failed": sum(x["returncode"] != 0 for x in completed),
            "queued": len(queue), "active": [{"mode": x["mode"], "episode_id": x["row"]["episode_id"], "gpu": x["gpu"]} for x in active],
            "elapsed_s": round(time.time() - started, 1),
        })
        time.sleep(1)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({"mode": item["mode"], "episode_id": item["row"]["episode_id"], "gpu": item["gpu"], "returncode": code})
            active.remove(item)
            slots.append(item["gpu"])
            slots.sort()
    failures = [x for x in completed if x["returncode"] != 0]
    atomic_json(PROGRESS, {"status": "COMPLETE" if not failures else "FAIL", "preflight": preflight, "total": len(planned), "completed": len(completed), "failed": len(failures), "queued": 0, "active": [], "elapsed_s": round(time.time() - started, 1)})
    return 0 if not failures else 2


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[round((len(values) - 1) * q)]


def verify(preflight: bool) -> int:
    protocol = protocol_value()
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("unseen protocol drift")
    rows = protocol["selection"][:1] if preflight else protocol["selection"]
    root = OUT / ("preflight" if preflight else "full")
    summaries = {}
    for mode, row in jobs(rows):
        path = root / "runs" / f"{mode}_ep_{row['episode_id']}" / "RUN_SUMMARY.json"
        value = json.loads(path.read_text())
        if value.get("status") != "PASS" or value.get("split") != "val_unseen" or value.get("public_unseen_accessed") is not True or value.get("threshold_tuned_on_val_unseen") is not False or not isinstance(value.get("metrics"), dict):
            raise RuntimeError("unseen worker boundary or metric failure")
        if not all(math.isfinite(float(value["metrics"][metric])) for metric in METRICS):
            raise RuntimeError("unseen metric is non-finite")
        if mode != "baseline" and value["executed_action_validation"]["all_equal"] is not True:
            raise RuntimeError("unseen controller action mismatch")
        if mode != "baseline":
            trace_path = path.parent / "controller_trace.jsonl"
            trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
            if any(row.get("public_unseen_authorized") is not True for row in trace):
                raise RuntimeError("unseen controller trace authorization drift")
        summaries[(mode, row["episode_id"])] = value
    if preflight:
        atomic_json(OUT / "MF3V_RXR_VAL_UNSEEN_PREFLIGHT.json", {"status": "PREFLIGHT_PASS", "runs": len(summaries), "threshold_tuned_on_val_unseen": False})
        return 0
    per_scene = []
    for row in rows:
        episode = row["episode_id"]
        baseline = summaries[("baseline", episode)]["metrics"]
        treatment = summaries[("ensemble", episode)]["metrics"]
        uncertainty = summaries[("uncertainty", episode)]["metrics"]
        delta = {metric: treatment[metric] - baseline[metric] for metric in METRICS}
        delta["utility"] = 0.50 * delta["ndtw"] + 0.25 * delta["sdtw"] + 0.25 * delta["spl"]
        delta["learned_minus_uncertainty_utility"] = 0.50 * (treatment["ndtw"] - uncertainty["ndtw"]) + 0.25 * (treatment["sdtw"] - uncertainty["sdtw"]) + 0.25 * (treatment["spl"] - uncertainty["spl"])
        per_scene.append({"scene_id": row["scene_id"], "episode_id": episode, **delta})
    rng = random.Random(20260829)
    utility_samples = []
    for _ in range(10000):
        sample = [per_scene[rng.randrange(len(per_scene))] for _ in per_scene]
        utility_samples.append(sum(x["utility"] for x in sample) / len(sample))
    aggregate = {
        metric: {"mean": sum(row[metric] for row in per_scene) / len(per_scene)}
        for metric in (*METRICS, "utility")
    }
    aggregate["utility"]["scene_bootstrap_95pct"] = [percentile(utility_samples, 0.025), percentile(utility_samples, 0.975)]
    result = {
        "schema_version": "revealnav-mf3v-rxr-val-unseen-result/1",
        "status": "UNSEEN_PILOT_PASS",
        "scope": "engineering pilot; not a full public benchmark claim",
        "runs": len(summaries), "failures": 0, "per_scene": per_scene,
        "aggregate_ensemble_minus_baseline": aggregate,
        "threshold_tuned_on_val_unseen": False,
        "test_or_test_challenge_accessed": False,
        "paper_result": False,
        "next_gate": "full RxR val_unseen benchmark with frozen MF3V protocol",
    }
    atomic_json(RESULT, result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--preflight", action="store_true")
    run_parser.add_argument("--gpus", default="0,1")
    run_parser.add_argument("--workers-per-gpu", type=int, default=1)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.command == "seal":
        return seal()
    if args.command == "run":
        return run(args.preflight, tuple(int(x) for x in args.gpus.split(",") if x), args.workers_per_gpu)
    return verify(args.preflight)


if __name__ == "__main__":
    raise SystemExit(main())
