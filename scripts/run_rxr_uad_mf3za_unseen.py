#!/usr/bin/env python3
"""Evaluate frozen MF3ZA on new, non-overlapping RxR val_unseen episodes."""

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
WORKER = ROOT / "scripts/rxr_uad_mf3za_unseen_worker.py"
DATASET = ROOT / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide.json.gz"
GROUND_TRUTH = ROOT / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide_gt.json.gz"
TRAIN_DATASET = ROOT / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
FREEZE = ROOT / "artifacts/evaluation/mf3za_consensus_band_freeze_v1/MF3ZA_VAL_SEEN_FREEZE.json"
PRIOR = ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_unseen_pilot_v2/MF3V_RXR_VAL_UNSEEN_PROTOCOL.json"
OUT = ROOT / "artifacts/evaluation/mf3za_uad_rxr_val_unseen_independent_v1"
PROTOCOL = OUT / "MF3ZA_RXR_VAL_UNSEEN_PROTOCOL.json"
PROGRESS = OUT / "MF3ZA_RXR_VAL_UNSEEN_PROGRESS.json"
RESULT = OUT / "MF3ZA_RXR_VAL_UNSEEN_RESULT.json"
SELECTION_SALT = "revealnav-mf3za-rxr-val-unseen-independent-three-per-scene/1"
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
    episodes = read_gzip_json(DATASET).get("episodes", [])
    if len(episodes) != 11006:
        raise RuntimeError("RxR val_unseen inventory drift")
    english = [
        row for row in episodes
        if row["instruction"]["language"] in ("en-US", "en-IN")
    ]
    if len(english) != 3669:
        raise RuntimeError("RxR val_unseen English inventory drift")
    grouped = defaultdict(list)
    for row in english:
        grouped[scene_id(row)].append(row)
    if len(grouped) != 11:
        raise RuntimeError("RxR val_unseen scene inventory drift")
    train_scenes = {
        scene_id(row)
        for row in read_gzip_json(TRAIN_DATASET).get("episodes", [])
    }
    if train_scenes & set(grouped):
        raise RuntimeError("train and val_unseen scenes overlap")
    prior = json.loads(PRIOR.read_text())
    consumed = {row["episode_id"] for row in prior["selection"]}
    gt = read_gzip_json(GROUND_TRUTH)
    selected = []
    for scene in sorted(grouped):
        ranked = sorted(
            grouped[scene],
            key=lambda row: hashlib.sha256(
                f"{SELECTION_SALT}:{scene}:{row['episode_id']}".encode()
            ).hexdigest(),
        )
        fresh = [row for row in ranked if str(row["episode_id"]) not in consumed]
        if len(fresh) < 3:
            raise RuntimeError(f"insufficient fresh unseen episodes for {scene}")
        scene_root = ROOT / f"third_party/ETP-R1/data/scene_datasets/mp3d/{scene}"
        for suffix in (".glb", ".navmesh"):
            asset = scene_root / f"{scene}{suffix}"
            if not asset.is_file() or asset.is_symlink():
                raise RuntimeError(f"missing or unsafe unseen asset: {asset}")
        for rank, row in enumerate(fresh[:3]):
            episode_id = str(row["episode_id"])
            if episode_id not in gt:
                raise RuntimeError(f"missing ground truth for {episode_id}")
            selected.append({
                "scene_id": scene,
                "episode_id": episode_id,
                "within_scene_rank": rank,
                "language": row["instruction"]["language"],
                "selection_digest": hashlib.sha256(
                    f"{SELECTION_SALT}:{scene}:{episode_id}".encode()
                ).hexdigest(),
            })
    if len(selected) != 33 or consumed & {row["episode_id"] for row in selected}:
        raise RuntimeError("independent unseen selection boundary failure")
    return selected, {
        "all_guide_episodes": len(episodes),
        "english_guide_episodes": len(english),
        "unseen_scenes": len(grouped),
        "selected_episodes": len(selected),
        "episodes_per_scene": 3,
        "excluded_prior_episode_ids": len(consumed),
        "overlap_with_prior": 0,
    }


def protocol_value() -> dict:
    freeze = json.loads(FREEZE.read_text())
    if freeze.get("status") != "MF3ZA_VAL_SEEN_FROZEN":
        raise RuntimeError("MF3ZA freeze is missing")
    if freeze.get("deployment_boundary", {}).get("val_unseen_authorized") is not True:
        raise RuntimeError("MF3ZA unseen authorization is missing")
    rows, counts = select_rows()
    return {
        "schema_version": "revealnav-mf3za-rxr-val-unseen-protocol/1",
        "status": "SEALED_BEFORE_MF3ZA_RXR_VAL_UNSEEN",
        "scope": "independent scene-stratified RxR English val_unseen evaluation",
        "selection_salt": SELECTION_SALT,
        "selection": rows,
        "counts": counts,
        "modes": ["baseline", "uncertainty", "ensemble"],
        "runs": len(rows) * 3,
        "revision": "mf3za",
        "primary_utility": "0.50*nDTW + 0.25*SDTW + 0.25*SPL",
        "uncertainty": "10000 deterministic scene-cluster bootstrap replicates",
        "threshold_tuned_on_val_unseen": False,
        "future_observations_used_online": False,
        "test_or_test_challenge_accessed": False,
        "sources": {
            "freeze": sha256_file(FREEZE),
            "worker": sha256_file(WORKER),
            "dataset": sha256_file(DATASET),
            "ground_truth": sha256_file(GROUND_TRUTH),
            "train_dataset": sha256_file(TRAIN_DATASET),
            "prior_protocol": sha256_file(PRIOR),
        },
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("MF3ZA unseen protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "episodes": value["counts"]["selected_episodes"],
        "runs": value["runs"],
        "overlap_with_prior": value["counts"]["overlap_with_prior"],
    }, indent=2))
    return 0


def jobs(rows: list[dict]) -> list[tuple[str, dict]]:
    return [
        (mode, row)
        for mode in ("baseline", "uncertainty", "ensemble")
        for row in rows
    ]


def run(preflight: bool, gpus: tuple[int, ...], workers_per_gpu: int) -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("MF3ZA unseen protocol is not sealed")
    rows = json.loads(PROTOCOL.read_text())["selection"][:1] if preflight else json.loads(PROTOCOL.read_text())["selection"]
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
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            process = subprocess.Popen(
                [
                    str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
                    "--episode-id", row["episode_id"], "--mode", mode,
                    "--run-dir", str(run_dir),
                ],
                cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
            )
            active.append({
                "process": process, "mode": mode, "row": row, "gpu": gpu,
                "streams": (stdout, stderr),
            })
        atomic_json(PROGRESS, {
            "status": "RUNNING", "preflight": preflight,
            "total": len(planned), "completed": len(completed),
            "failed": sum(row["returncode"] != 0 for row in completed),
            "queued": len(queue),
            "active": [
                {"mode": row["mode"], "episode_id": row["row"]["episode_id"], "gpu": row["gpu"]}
                for row in active
            ],
            "elapsed_s": round(time.time() - started, 1),
        })
        time.sleep(1)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({
                "mode": item["mode"], "episode_id": item["row"]["episode_id"],
                "gpu": item["gpu"], "returncode": code,
            })
            active.remove(item)
            slots.append(item["gpu"])
            slots.sort()
    failures = [row for row in completed if row["returncode"] != 0]
    atomic_json(PROGRESS, {
        "status": "COMPLETE" if not failures else "FAIL",
        "preflight": preflight, "total": len(planned),
        "completed": len(completed), "failed": len(failures),
        "queued": 0, "active": [],
        "elapsed_s": round(time.time() - started, 1),
    })
    return 0 if not failures else 2


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[round((len(values) - 1) * q)]


def verify(preflight: bool) -> int:
    protocol = protocol_value()
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("MF3ZA unseen protocol drift")
    rows = protocol["selection"][:1] if preflight else protocol["selection"]
    root = OUT / ("preflight" if preflight else "full")
    summaries = {}
    for mode, row in jobs(rows):
        path = root / "runs" / f"{mode}_ep_{row['episode_id']}" / "RUN_SUMMARY.json"
        value = json.loads(path.read_text())
        if not (
            value.get("status") == "PASS"
            and value.get("split") == "val_unseen"
            and value.get("public_unseen_accessed") is True
            and value.get("threshold_tuned_on_val_unseen") is False
            and isinstance(value.get("metrics"), dict)
        ):
            raise RuntimeError("MF3ZA unseen worker boundary or metric failure")
        if not all(math.isfinite(float(value["metrics"][metric])) for metric in METRICS):
            raise RuntimeError("MF3ZA unseen metric is non-finite")
        if mode != "baseline" and value["executed_action_validation"]["all_equal"] is not True:
            raise RuntimeError("MF3ZA unseen controller action mismatch")
        summaries[(mode, row["episode_id"])] = value
    if preflight:
        atomic_json(OUT / "MF3ZA_RXR_VAL_UNSEEN_PREFLIGHT.json", {
            "status": "PREFLIGHT_PASS",
            "runs": len(summaries),
            "threshold_tuned_on_val_unseen": False,
        })
        return 0
    per_episode = []
    for row in rows:
        episode = row["episode_id"]
        baseline = summaries[("baseline", episode)]["metrics"]
        treatment = summaries[("ensemble", episode)]["metrics"]
        uncertainty = summaries[("uncertainty", episode)]["metrics"]
        delta = {
            metric: treatment[metric] - baseline[metric]
            for metric in METRICS
        }
        delta["utility"] = (
            0.50 * delta["ndtw"] + 0.25 * delta["sdtw"] + 0.25 * delta["spl"]
        )
        delta["learned_minus_uncertainty_utility"] = (
            0.50 * (treatment["ndtw"] - uncertainty["ndtw"])
            + 0.25 * (treatment["sdtw"] - uncertainty["sdtw"])
            + 0.25 * (treatment["spl"] - uncertainty["spl"])
        )
        per_episode.append({
            "scene_id": row["scene_id"], "episode_id": episode, **delta,
        })
    grouped = defaultdict(list)
    for row in per_episode:
        grouped[row["scene_id"]].append(row)
    scenes = sorted(grouped)
    rng = random.Random(20260829)
    samples = defaultdict(list)
    for _ in range(10000):
        selected_scenes = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        sample = [row for scene in selected_scenes for row in grouped[scene]]
        for metric in (*METRICS, "utility", "learned_minus_uncertainty_utility"):
            samples[metric].append(sum(row[metric] for row in sample) / len(sample))
    aggregate = {}
    for metric in (*METRICS, "utility", "learned_minus_uncertainty_utility"):
        aggregate[metric] = {
            "mean": sum(row[metric] for row in per_episode) / len(per_episode),
            "scene_cluster_bootstrap_95pct": [
                percentile(samples[metric], 0.025), percentile(samples[metric], 0.975),
            ],
        }
    action_changes = sum(
        summaries[("ensemble", row["episode_id"])]["controller"]["actions_changed"]
        for row in rows
    )
    gates = {
        "utility_point_positive": aggregate["utility"]["mean"] > 0,
        "utility_lower_95_positive": aggregate["utility"]["scene_cluster_bootstrap_95pct"][0] > 0,
        "success_point_nonnegative": aggregate["success"]["mean"] >= 0,
        "spl_point_nonnegative": aggregate["spl"]["mean"] >= 0,
        "ndtw_point_nonnegative": aggregate["ndtw"]["mean"] >= 0,
        "learned_utility_exceeds_uncertainty": aggregate["learned_minus_uncertainty_utility"]["mean"] > 0,
        "controller_changes_at_least_one_action": action_changes > 0,
    }
    passed = all(gates.values())
    atomic_json(RESULT, {
        "schema_version": "revealnav-mf3za-rxr-val-unseen-result/1",
        "status": "UNSEEN_ADVANTAGE_PASS" if passed else "UNSEEN_ADVANTAGE_FAIL",
        "scope": "independent non-overlapping 33-episode val_unseen evaluation",
        "runs": len(summaries), "failures": 0,
        "action_changes": action_changes,
        "per_episode": per_episode,
        "aggregate_ensemble_minus_baseline": aggregate,
        "gates": gates,
        "threshold_tuned_on_val_unseen": False,
        "test_or_test_challenge_accessed": False,
        "paper_result": passed,
    })
    return 0 if passed else 2


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
        return run(
            args.preflight,
            tuple(int(value) for value in args.gpus.split(",") if value),
            args.workers_per_gpu,
        )
    return verify(args.preflight)


if __name__ == "__main__":
    raise SystemExit(main())
