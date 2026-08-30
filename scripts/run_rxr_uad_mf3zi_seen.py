#!/usr/bin/env python3
"""Paired MF3ZI task-metric gate on the fixed RxR val_seen cohort."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
REVISION = "mf3zi"
TAG = "MF3ZI"
WORKER = ROOT / "scripts/rxr_uad_mf3zi_worker.py"
SOURCE_DEPENDENCIES: tuple[Path, ...] = ()
GATE = ROOT / "artifacts/training/mf3zi_uncertainty_return_gate_v1/MF3ZI_CROSSFIT_GATE.json"
OUT = ROOT / "artifacts/evaluation/mf3zi_causal_uncertainty_arbitration_rxr_val_seen_v1"
PROTOCOL = OUT / "MF3ZI_RXR_VAL_SEEN_PROTOCOL.json"
PROGRESS = OUT / "MF3ZI_RXR_VAL_SEEN_PROGRESS.json"
RESULT = OUT / "MF3ZI_RXR_VAL_SEEN_RESULT.json"
MF3ZG_PROTOCOL = ROOT / "artifacts/evaluation/mf3zg_core_preserving_hierarchy_rxr_val_seen_v1/MF3ZG_RXR_VAL_SEEN_PROTOCOL.json"
MF3V_CONTROLS = ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1/full/runs"
METRICS = ("success", "spl", "ndtw", "sdtw")
PRIMARY_TREATMENT = (
    "MF3ZG learned core followed by a train-only gated one-shot "
    "native-margin uncertainty residual"
)


def sha256_file(path: Path) -> str:
    import hashlib
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


def selection() -> list[dict]:
    prior = json.loads(MF3ZG_PROTOCOL.read_text())
    if (
        prior.get("status") != "SEALED_BEFORE_RXR_VAL_SEEN_TASK_METRICS"
        or len(prior.get("selection", [])) != 57
        or prior.get("public_unseen_authorized") is not False
    ):
        raise RuntimeError(f"{TAG} val_seen cohort source drift")
    return prior["selection"]


def protocol_value() -> dict:
    gate = json.loads(GATE.read_text())
    if not (
        gate.get("status") == "SHADOW_GATE_PASS"
        and gate.get("task_metric_run_authorized") is True
        and gate.get("controls", {}).get("unseen_or_test_read") is False
    ):
        raise RuntimeError(f"{TAG} gate does not authorize val_seen")
    rows = selection()
    return {
        "schema_version": f"revealnav-{REVISION}-rxr-val-seen-protocol/1",
        "status": "SEALED_BEFORE_RXR_VAL_SEEN_TASK_METRICS",
        "selection": rows,
        "ensemble_member_seeds": [20260826, 20260827, 20260828],
        "paired_unit": "one deterministic English guide episode per val_seen scene",
        "primary_treatment": PRIMARY_TREATMENT,
        "secondary_treatments": "MF3V native-margin uncertainty control reused as a fixed comparator",
        "primary_utility": "0.50*nDTW + 0.25*SDTW + 0.25*SPL",
        "uncertainty": "10000 deterministic scene bootstrap replicates",
        "success_gate": "primary utility lower 95% bound > 0; success, SPL, nDTW point estimates non-negative; primary utility exceeds MF3V uncertainty-only control",
        "runs": {"baseline_reused_from_mf3v": 57, "uncertainty_reused_from_mf3v": 57, "ensemble": 57, "new_total": 57},
        "reused_control_provenance": {
            "mf3v_result": str((ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1/MF3V_RXR_VAL_SEEN_RESULT.json").relative_to(ROOT)),
            "control_root": str(MF3V_CONTROLS.relative_to(ROOT)),
        },
        "sources": {
            "method_gate": {"path": str(GATE.relative_to(ROOT)), "sha256": sha256_file(GATE)},
            "mf3zg_protocol": {"path": str(MF3ZG_PROTOCOL.relative_to(ROOT)), "sha256": sha256_file(MF3ZG_PROTOCOL)},
            "worker": {"path": str(WORKER.relative_to(ROOT)), "sha256": sha256_file(WORKER)},
            "worker_dependencies": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in SOURCE_DEPENDENCIES
            ],
        },
        "public_unseen_authorized": False,
        "test_or_test_challenge_accessed": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != value:
            raise RuntimeError(f"{TAG} val_seen protocol drift")
    else:
        atomic_json(PROTOCOL, value)
    return 0


def run(preflight: bool, gpus: tuple[int, ...], workers_per_gpu: int, resume: bool) -> int:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol_value() != protocol:
        raise RuntimeError(f"{TAG} protocol is not sealed")
    rows = protocol["selection"][:1] if preflight else protocol["selection"]
    root = OUT / ("preflight" if preflight else "full")
    root.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume:
        for directory in sorted((root / "runs").glob("ensemble_ep_*")):
            summary = directory / "RUN_SUMMARY.json"
            if summary.is_file() and json.loads(summary.read_text()).get("status") == "PASS":
                done.add(directory.name.removeprefix("ensemble_ep_"))
            else:
                interrupted = root / "interrupted"
                interrupted.mkdir(parents=True, exist_ok=True)
                destination = interrupted / directory.name
                suffix = 1
                while destination.exists():
                    destination = interrupted / f"{directory.name}_{suffix}"
                    suffix += 1
                os.replace(directory, destination)
    queue = [row for row in rows if row["episode_id"] not in done]
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    if not slots:
        raise ValueError("no GPU slots")
    active = []
    completed = []
    started = time.time()
    while queue or active:
        while queue and slots:
            row = queue.pop(0); gpu = slots.pop(0); episode = row["episode_id"]
            directory = root / "runs" / f"ensemble_ep_{episode}"
            logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
            stdout = (logs / f"ensemble_ep_{episode}.stdout").open("w")
            stderr = (logs / f"ensemble_ep_{episode}.stderr").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            process = subprocess.Popen([
                str(PYTHON), str(WORKER), "--episode-id", episode,
                "--split", "val_seen", "--run-dir", str(directory),
            ], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            active.append({"process": process, "row": row, "gpu": gpu, "streams": (stdout, stderr)})
        atomic_json(PROGRESS, {"status": "RUNNING", "preflight": preflight, "total": len(rows), "completed": len(done) + len(completed), "failed": sum(item["returncode"] != 0 for item in completed), "queued": len(queue), "active": [{"episode_id": item["row"]["episode_id"], "gpu": item["gpu"]} for item in active], "elapsed_s": round(time.time() - started, 1)})
        time.sleep(1)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]: stream.close()
            completed.append({"episode_id": item["row"]["episode_id"], "returncode": code})
            slots.append(item["gpu"]); slots.sort(); active.remove(item)
    failures = [item for item in completed if item["returncode"] != 0]
    atomic_json(PROGRESS, {"status": "COMPLETE" if not failures else "FAIL", "preflight": preflight, "total": len(rows), "completed": len(done) + len(completed), "failed": len(failures), "queued": 0, "active": [], "elapsed_s": round(time.time() - started, 1)})
    return 0 if not failures else 2


def utility(metrics: dict) -> float:
    return 0.50 * float(metrics["ndtw"]) + 0.25 * float(metrics["sdtw"]) + 0.25 * float(metrics["spl"])


def load_summary(path: Path, episode: str, mode: str, split: str) -> dict:
    value = json.loads(path.read_text())
    if not (
        value.get("status") == "PASS" and value.get("episode_id") == episode
        and value.get("mode") == mode and value.get("split") == split
        and value.get("public_unseen_accessed") is False
        and isinstance(value.get("metrics"), dict)
        and all(math.isfinite(float(value["metrics"][key])) for key in METRICS)
    ):
        raise RuntimeError(f"{TAG} summary boundary failure: {path}")
    return value


def verify(preflight: bool) -> int:
    protocol = json.loads(PROTOCOL.read_text())
    rows = protocol["selection"][:1] if preflight else protocol["selection"]
    summaries = {}
    for row in rows:
        episode = row["episode_id"]
        for mode in ("baseline", "uncertainty"):
            path = MF3V_CONTROLS / f"{mode}_ep_{episode}" / "RUN_SUMMARY.json"
            value = load_summary(path, episode, mode, "val_seen")
            summaries[(mode, episode)] = value
        path = OUT / ("preflight" if preflight else "full") / "runs" / f"ensemble_ep_{episode}" / "RUN_SUMMARY.json"
        value = load_summary(path, episode, "ensemble", "val_seen")
        if value.get("revision") != REVISION or value.get("executed_action_validation", {}).get("all_equal") is not True:
            raise RuntimeError(f"{TAG} controller validation failed")
        summaries[("ensemble", episode)] = value
    if preflight:
        atomic_json(OUT / f"{TAG}_RXR_VAL_SEEN_PREFLIGHT.json", {"status": "PREFLIGHT_PASS", "runs": len(summaries), "public_unseen_authorized": False})
        return 0
    per_scene = []
    for row in rows:
        episode = row["episode_id"]
        baseline = summaries[("baseline", episode)]["metrics"]
        uncertainty = summaries[("uncertainty", episode)]["metrics"]
        treatment = summaries[("ensemble", episode)]["metrics"]
        delta = {key: float(treatment[key]) - float(baseline[key]) for key in METRICS}
        delta["utility"] = utility(treatment) - utility(baseline)
        compare = {key: float(treatment[key]) - float(uncertainty[key]) for key in METRICS}
        compare["utility"] = utility(treatment) - utility(uncertainty)
        per_scene.append({"scene_id": row["scene_id"], "episode_id": episode, **delta, **{f"learned_minus_uncertainty_{key}": value for key, value in compare.items()}})
    rng = random.Random(20260830)
    metrics = (*METRICS, "utility")
    boots = {key: [] for key in metrics}; compare_boots = {key: [] for key in metrics}
    for _ in range(10000):
        sample = [per_scene[rng.randrange(len(per_scene))] for _ in per_scene]
        for key in metrics:
            boots[key].append(sum(item[key] for item in sample) / len(sample))
            compare_boots[key].append(sum(item[f"learned_minus_uncertainty_{key}"] for item in sample) / len(sample))
    def pct(values, q): return sorted(values)[round((len(values) - 1) * q)]
    aggregate = {key: {"mean": sum(item[key] for item in per_scene) / len(per_scene), "scene_bootstrap_95pct": [pct(boots[key], .025), pct(boots[key], .975)]} for key in metrics}
    comparison = {key: {"mean": sum(item[f"learned_minus_uncertainty_{key}"] for item in per_scene) / len(per_scene), "scene_bootstrap_95pct": [pct(compare_boots[key], .025), pct(compare_boots[key], .975)]} for key in metrics}
    gates = {
        "utility_point_positive": aggregate["utility"]["mean"] > 0,
        "utility_lower_95_positive": aggregate["utility"]["scene_bootstrap_95pct"][0] > 0,
        "success_point_nonnegative": aggregate["success"]["mean"] >= 0,
        "spl_point_nonnegative": aggregate["spl"]["mean"] >= 0,
        "ndtw_point_nonnegative": aggregate["ndtw"]["mean"] >= 0,
        "learned_utility_exceeds_uncertainty": comparison["utility"]["mean"] > 0,
    }
    passed = all(gates.values())
    atomic_json(RESULT, {"schema_version": f"revealnav-{REVISION}-rxr-val-seen-result/1", "status": "TASK_METRIC_GATE_PASS" if passed else "TASK_METRIC_GATE_FAIL", "aggregate_ensemble_minus_baseline": aggregate, "aggregate_ensemble_minus_uncertainty": comparison, "per_scene": per_scene, "gates": gates, "public_unseen_authorized": False})
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    for command in ("run",):
        p = sub.add_parser(command); p.add_argument("--preflight", action="store_true"); p.add_argument("--gpus", default="0,1"); p.add_argument("--workers-per-gpu", type=int, default=1); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("verify"); p.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.command == "seal": return seal()
    if args.command == "run": return run(args.preflight, tuple(int(v) for v in args.gpus.split(",") if v), args.workers_per_gpu, args.resume)
    return verify(args.preflight)


if __name__ == "__main__":
    raise SystemExit(main())
