#!/usr/bin/env python3
"""Build paired RxR-train final-return labels for actual MF3V switches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

from evaluate_rxr_uad_horizon_mf3v import collect, load_models, manifest_path, score
from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SOURCE = manifest_path("final")
MF3V_GATE = ROOT / "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/MF3V_SHADOW_GATE.json"
WORKER = ROOT / "scripts/rxr_uad_controller_worker_mf3.py"
OUT = ROOT / "artifacts/phase1/mf3zd_direct_switch_returns_v1"
SELECTION = OUT / "MF3ZD_DIRECT_SWITCH_SELECTION.json"
PROGRESS = OUT / "MF3ZD_DIRECT_SWITCH_PROGRESS.json"
MANIFEST = OUT / "MF3ZD_DIRECT_SWITCH_MANIFEST.json"
SCHEMA_TAG = "mf3zd"
WORKER_REVISION = "mf3v"
LOWER_QUANTILE = 0.985
UPPER_QUANTILE = 0.995
EXPECTED_EPISODES = 93
EXPECTED_SCENES = 50


def baseline_summary(feature_record: dict) -> tuple[Path, dict]:
    feature = (ROOT / feature_record["path"]).resolve()
    run = feature.parent
    stats = list(run.rglob("stats_ep_ckpt_1320_train_r0_w1.json"))
    if len(stats) != 1:
        raise RuntimeError("MF3ZD baseline task metric file drift")
    metrics = json.loads(stats[0].read_text()).get(str(feature_record["episode_id"]))
    if not isinstance(metrics, dict):
        raise RuntimeError("MF3ZD baseline episode metric missing")
    for key in ("success", "spl", "ndtw", "sdtw"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError("MF3ZD baseline metric non-finite")
    return stats[0], metrics


def select() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(device)
    sequences = collect(models, "fit", SOURCE, device)
    source = json.loads(SOURCE.read_text())
    records = [row for row in source["records"] if row["split"] == "fit"]
    if len(records) != len(sequences) or len(records) != 1303:
        raise RuntimeError("MF3ZD source alignment drift")
    fit_scores = [score(row) for sequence in sequences for row in sequence if row]
    gate = json.loads(MF3V_GATE.read_text())
    lower = float(gate["selected_rule"]["final_training_threshold"])
    upper = float(gate["selected_rule"]["score_upper_threshold"])
    if not (
        abs(lower - float(torch.quantile(torch.tensor(fit_scores), LOWER_QUANTILE))) < 1e-6
        and abs(upper - float(torch.quantile(torch.tensor(fit_scores), UPPER_QUANTILE))) < 1e-6
    ):
        raise RuntimeError("MF3V score band does not reproduce")
    selected = []
    for metadata, sequence in zip(records, sequences):
        for step, row in enumerate(sequence):
            if row is None or not lower < score(row) <= upper:
                continue
            stats, metrics = baseline_summary(metadata)
            selected.append({
                "episode_id": str(metadata["episode_id"]),
                "scene_id": str(metadata["scene_id"]), "decision_step": step,
                "mf3v_score": score(row), "baseline_metrics": metrics,
                "baseline_stats": {
                    "path": str(stats.relative_to(ROOT)), "bytes": stats.stat().st_size,
                    "sha256": sha256_file(stats),
                },
                "source_feature": {
                    "path": metadata["path"], "bytes": metadata["bytes"],
                    "sha256": metadata["sha256"],
                },
            })
            break
    if (len(selected) != EXPECTED_EPISODES
            or len({row["scene_id"] for row in selected}) != EXPECTED_SCENES):
        raise RuntimeError("MF3ZD selected cohort drift")
    value = {
        "schema_version": f"revealnav-{SCHEMA_TAG}-direct-switch-selection/1",
        "status": "SEALED_BEFORE_DIRECT_SWITCH_ROLLOUTS", "split": "train",
        "selection": selected,
        "counts": {"episodes": len(selected), "scenes": EXPECTED_SCENES},
        "score_band": {"lower_exclusive": lower, "upper_inclusive": upper},
        "checkpoints": checkpoints,
        "sources": {"online_manifest": sha256_file(SOURCE),
                    "mf3v_gate": sha256_file(MF3V_GATE),
                    "worker": sha256_file(WORKER)},
        "unseen_or_test_read": False,
    }
    if SELECTION.exists() and json.loads(SELECTION.read_text()) != value:
        raise RuntimeError("MF3ZD selection drift")
    if not SELECTION.exists():
        atomic_json(SELECTION, value)
    print(json.dumps({"episodes": EXPECTED_EPISODES,
                      "scenes": EXPECTED_SCENES}, indent=2))
    return 0


def run(gpus: tuple[int, ...], workers_per_gpu: int) -> int:
    protocol = json.loads(SELECTION.read_text())
    if protocol["sources"]["worker"] != sha256_file(WORKER):
        raise RuntimeError("MF3ZD worker drift after seal")
    root = OUT / "runs"
    if root.exists():
        raise RuntimeError("refusing to overwrite MF3ZD runs")
    root.mkdir(parents=True)
    queue = list(protocol["selection"])
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    active = []; completed = []; started = time.time()
    while queue or active:
        while queue and slots:
            row = queue.pop(0); gpu = slots.pop(0); episode = row["episode_id"]
            run_dir = root / f"ep_{episode}"; logs = OUT / "logs"; logs.mkdir(exist_ok=True)
            stdout = (logs / f"ep_{episode}.stdout").open("w")
            stderr = (logs / f"ep_{episode}.stderr").open("w")
            env = {
                **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "REVEALNAV_MF3_INTERVENTION_FEATURE": str(run_dir / "intervention_feature.npz"),
            }
            process = subprocess.Popen([
                str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
                "--episode-id", episode, "--mode", "ensemble", "--revision", WORKER_REVISION,
                "--split", "train", "--run-dir", str(run_dir),
            ], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            active.append({"process": process, "row": row, "gpu": gpu,
                           "streams": (stdout, stderr)})
        atomic_json(PROGRESS, {
            "status": "RUNNING", "total": len(protocol["selection"]),
            "completed": len(completed),
            "failed": sum(row["returncode"] != 0 for row in completed),
            "queued": len(queue), "active": [
                {"episode_id": row["row"]["episode_id"], "gpu": row["gpu"]}
                for row in active
            ], "elapsed_s": round(time.time() - started, 1),
        })
        time.sleep(1)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]: stream.close()
            completed.append({"episode_id": item["row"]["episode_id"],
                              "returncode": code})
            active.remove(item); slots.append(item["gpu"]); slots.sort()
    failures = [row for row in completed if row["returncode"] != 0]
    atomic_json(PROGRESS, {
        "status": "COMPLETE" if not failures else "FAIL",
        "total": len(completed), "completed": len(completed),
        "failed": len(failures), "queued": 0, "active": [],
        "elapsed_s": round(time.time() - started, 1),
    })
    return 0 if not failures else 2


def utility(metrics: dict) -> float:
    return 0.50 * metrics["ndtw"] + 0.25 * metrics["sdtw"] + 0.25 * metrics["spl"]


def assemble() -> int:
    protocol = json.loads(SELECTION.read_text()); rows = []
    for source in protocol["selection"]:
        episode = source["episode_id"]; run = OUT / f"runs/ep_{episode}"
        summary_path = run / "RUN_SUMMARY.json"; feature = run / "intervention_feature.npz"
        trace_path = run / "controller_trace.jsonl"
        summary = json.loads(summary_path.read_text())
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        changed = [row for row in trace if row.get("action_changed") is True]
        if not (
            summary.get("status") == "PASS" and summary.get("split") == "train"
            and summary.get("revision") == WORKER_REVISION
            and summary.get("controller", {}).get("actions_changed") == 1
            and len(changed) == 1
            and feature.is_file() and not feature.is_symlink()
        ):
            raise RuntimeError(f"MF3ZD rollout boundary failure: {episode}")
        treatment = summary["metrics"]; baseline = source["baseline_metrics"]
        delta = {key: treatment[key] - baseline[key]
                 for key in ("success", "spl", "ndtw", "sdtw")}
        delta["utility"] = utility(treatment) - utility(baseline)
        rows.append({
            "row_index": len(rows), "episode_id": episode,
            "scene_id": source["scene_id"],
            "selection_decision_index": source["decision_step"],
            "selection_mf3v_score": source["mf3v_score"],
            "decision_step": int(changed[0]["step"]),
            "mf3v_score": float(changed[0]["policy_risk_adjusted_score"]),
            "delta": delta,
            "decision": {
                key: changed[0][key] for key in (
                    "step", "minimum_top2_advantage", "median_top2_advantage",
                    "robust_top2_advantage", "ensemble_mad",
                    "cold_start_floor_ratio", "cold_start_relative_mad",
                    "policy_risk_adjusted_score", "native_margin",
                    "current_local_action_ids",
                )
            },
            "baseline_metrics": baseline, "treatment_metrics": treatment,
            "feature": {"path": str(feature.relative_to(ROOT)),
                        "bytes": feature.stat().st_size,
                        "sha256": sha256_file(feature)},
            "run_summary": {"path": str(summary_path.relative_to(ROOT)),
                            "bytes": summary_path.stat().st_size,
                            "sha256": sha256_file(summary_path)},
            "controller_trace": {"path": str(trace_path.relative_to(ROOT)),
                                 "bytes": trace_path.stat().st_size,
                                 "sha256": sha256_file(trace_path)},
        })
    if len(rows) != EXPECTED_EPISODES:
        raise RuntimeError("MF3ZD assembled row count drift")
    value = {
        "schema_version": f"revealnav-{SCHEMA_TAG}-direct-switch-manifest/1",
        "status": "DIRECT_SWITCH_RETURN_DATASET_READY", "records": rows,
        "counts": {"pairs": len(rows), "scenes": len({r["scene_id"] for r in rows}),
                   "positive_utility": sum(r["delta"]["utility"] > 0 for r in rows),
                   "negative_utility": sum(r["delta"]["utility"] < 0 for r in rows)},
        "mean_utility_delta": sum(r["delta"]["utility"] for r in rows) / len(rows),
        "selection_sha256": sha256_file(SELECTION),
        "unseen_or_test_read": False,
    }
    atomic_json(MANIFEST, value)
    print(json.dumps({"status": value["status"], "counts": value["counts"],
                      "mean_utility_delta": value["mean_utility_delta"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    run_parser = sub.add_parser("run"); run_parser.add_argument("--gpus", default="0,1")
    run_parser.add_argument("--workers-per-gpu", type=int, default=1)
    sub.add_parser("assemble"); args = parser.parse_args()
    if args.command == "select": return select()
    if args.command == "run": return run(tuple(int(x) for x in args.gpus.split(",")),
                                          args.workers_per_gpu)
    return assemble()


if __name__ == "__main__":
    raise SystemExit(main())
