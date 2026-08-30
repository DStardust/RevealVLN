#!/usr/bin/env python3
"""Seal and run the paired RxR val_seen MF3 UAD task-metric gate."""

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
WORKER = ROOT / "scripts/rxr_uad_controller_worker_mf3.py"
REVISION = "mf3s"
TAG = "MF3S"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/"
    "val_seen/val_seen_guide.json.gz"
)
GATE = ROOT / (
    "artifacts/evaluation/mf3s_policy_risk_shadow_gate_v1/"
    "MF3S_SHADOW_GATE.json"
)
OUT = ROOT / "artifacts/evaluation/mf3s_uad_rxr_val_seen_v1"
PROTOCOL = OUT / "MF3S_RXR_VAL_SEEN_PROTOCOL.json"
PROGRESS = OUT / "MF3S_RXR_VAL_SEEN_PROGRESS.json"
RESULT = OUT / "MF3S_RXR_VAL_SEEN_RESULT.json"
SEEDS = (20260826, 20260827, 20260828)
SALT = "revealnav-mf3s-rxr-val-seen-all-scenes/1"
METRICS = ("success", "spl", "ndtw", "sdtw")
MF3V_CONTROLS = ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1/full/runs"


def configure_revision(revision: str) -> None:
    global REVISION, TAG, GATE, OUT, PROTOCOL, PROGRESS, RESULT, SALT
    if revision not in ("mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"):
        raise ValueError("unsupported MF3 task-metric revision")
    REVISION = revision
    TAG = revision.upper()
    if revision == "mf3zh":
        GATE = ROOT / (
            "artifacts/training/mf3zh_uncertainty_floor_residual_gate_v1/"
            "MF3ZH_SHADOW_GATE.json"
        )
        OUT = ROOT / (
            "artifacts/evaluation/"
            "mf3zh_uncertainty_floor_residual_rxr_val_seen_v1"
        )
    elif revision == "mf3zg":
        GATE = ROOT / (
            "artifacts/training/mf3zg_hierarchical_core_preserving_gate_v1/"
            "MF3ZG_SHADOW_GATE.json"
        )
        OUT = ROOT / (
            "artifacts/evaluation/"
            "mf3zg_core_preserving_hierarchy_rxr_val_seen_v1"
        )
    elif revision == "mf3zf":
        GATE = ROOT / (
            "artifacts/training/mf3zf_action_aligned_return_gate_v1/"
            "MF3ZF_CROSSFIT_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3zf_coverage_safety_rxr_val_seen_v1"
    elif revision == "mf3ze":
        GATE = ROOT / (
            "artifacts/training/mf3ze_action_aligned_return_gate_v1/"
            "MF3ZE_CROSSFIT_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3ze_action_aligned_rxr_val_seen_v1"
    elif revision == "mf3zc":
        GATE = ROOT / (
            "artifacts/evaluation/mf3zc_calibrated_dissent_shadow_gate_v1/"
            "MF3ZC_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3zc_calibrated_dissent_rxr_val_seen_v1"
    elif revision == "mf3zb":
        GATE = ROOT / (
            "artifacts/evaluation/mf3zb_temporal_maturity_shadow_gate_v1/"
            "MF3ZB_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3zb_temporal_maturity_rxr_val_seen_v1"
    elif revision == "mf3za":
        GATE = ROOT / (
            "artifacts/evaluation/mf3za_consensus_band_shadow_gate_v1/"
            "MF3ZA_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3za_consensus_band_rxr_val_seen_v1"
    elif revision == "mf3z":
        GATE = ROOT / (
            "artifacts/evaluation/mf3z_adaptive_tail_shadow_gate_v1/"
            "MF3Z_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3z_adaptive_tail_rxr_val_seen_aligned_v1"
    elif revision == "mf3y":
        GATE = ROOT / (
            "artifacts/evaluation/mf3y_consensus_tail_shadow_gate_v1/"
            "MF3Y_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3y_uad_rxr_val_seen_v1"
    elif revision == "mf3v":
        GATE = ROOT / (
            "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/"
            "MF3V_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1"
    elif revision == "mf3u":
        GATE = ROOT / (
            "artifacts/evaluation/mf3u_policy_anchor_shadow_gate_v1/"
            "MF3U_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3u_uad_rxr_val_seen_dev_v1"
    elif revision == "mf3t":
        GATE = ROOT / (
            "artifacts/evaluation/mf3t_coverage_shadow_gate_v2/"
            "MF3T_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3t_uad_rxr_val_seen_dev_v1"
    else:
        GATE = ROOT / (
            "artifacts/evaluation/mf3s_policy_risk_shadow_gate_v1/"
            "MF3S_SHADOW_GATE.json"
        )
        OUT = ROOT / "artifacts/evaluation/mf3s_uad_rxr_val_seen_v1"
    PROTOCOL = OUT / f"{TAG}_RXR_VAL_SEEN_PROTOCOL.json"
    PROGRESS = OUT / f"{TAG}_RXR_VAL_SEEN_PROGRESS.json"
    RESULT = OUT / f"{TAG}_RXR_VAL_SEEN_RESULT.json"
    # New candidates use the frozen MF3V scene/episode pairing so that every
    # extension is compared on exactly the same 57 val_seen episodes.
    SALT = (
        "revealnav-mf3v-rxr-val-seen-all-scenes/1"
        if REVISION in ("mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
        else f"revealnav-{REVISION}-rxr-val-seen-all-scenes/1"
    )


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


def digest(value: str) -> str:
    return hashlib.sha256(f"{SALT}:{value}".encode()).hexdigest()


def selection() -> list[dict]:
    with gzip.open(DATASET, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    english = [
        row for row in episodes
        if row["instruction"]["language"] in ("en-US", "en-IN")
    ]
    if len(episodes) != 6746 or len(english) != 2255:
        raise RuntimeError("RxR val_seen inventory drift")
    grouped = defaultdict(list)
    for row in english:
        grouped[Path(row["scene_id"]).stem].append(row)
    if len(grouped) != 57:
        raise RuntimeError("RxR val_seen scene inventory drift")
    result = []
    for scene in sorted(grouped):
        episode = min(
            grouped[scene],
            key=lambda row: digest(f"{scene}:{row['episode_id']}"),
        )
        result.append({
            "scene_id": scene,
            "episode_id": str(episode["episode_id"]),
            "language": episode["instruction"]["language"],
            "selection_digest": digest(f"{scene}:{episode['episode_id']}"),
        })
    return result


def protocol_value() -> dict:
    gate = json.loads(GATE.read_text())
    if not (
        gate.get("status") == "SHADOW_GATE_PASS"
        and gate.get("task_metric_run_authorized") is True
    ):
        raise RuntimeError("MF3 shadow gate does not authorize val_seen")
    rows = selection()
    worker_sha = sha256_file(WORKER)
    # The MF3S protocol was sealed before the MF3T branch was added to the
    # shared worker.  Preserve that already-consumed seal byte-for-byte; the
    # MF3T protocol is new and records the current worker hash.
    if REVISION == "mf3s" and PROTOCOL.exists():
        sealed = json.loads(PROTOCOL.read_text())
        if sealed.get("schema_version") == "revealnav-mf3s-rxr-val-seen-protocol/1":
            worker_sha = sealed["worker_sha256"]
    return {
        "schema_version": (
            f"revealnav-{REVISION}-rxr-val-seen-protocol/2"
            if REVISION in ("mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh")
            else f"revealnav-{REVISION}-rxr-val-seen-protocol/1"
        ),
        "status": "SEALED_BEFORE_RXR_VAL_SEEN_TASK_METRICS",
        "selection_salt": SALT,
        "selection": rows,
        "ensemble_member_seeds": list(SEEDS),
        "runs": ({
            "baseline_reused_from_mf3v": len(rows),
            "uncertainty_reused_from_mf3v": len(rows),
            "ensemble": len(rows), "new_total": len(rows),
        } if REVISION in ("mf3ze", "mf3zf", "mf3zg", "mf3zh") else {
            "baseline": len(rows), "uncertainty": len(rows),
            "ensemble": len(rows), "total": len(rows) * 3,
        }),
        "paired_unit": "one deterministic English guide episode per val_seen scene",
        "primary_treatment": {
            "mf3s": (
                "online first-crossing policy-risk-adjusted rescue-versus-harm "
                "top-2 correction from the three fixed MF3Q final members"
            ),
            "mf3z": (
                "online first-crossing horizon-consistent adaptive relative-margin "
                "consensus-gated tail recovery from three fixed MF3V final members"
            ),
            "mf3za": (
                "online first-crossing horizon-consistent consensus-band "
                "relative-margin tail recovery from three fixed MF3V final members"
            ),
            "mf3zb": (
                "online temporal-maturity-gated horizon-consistent top-2 "
                "correction from three fixed MF3V final members"
            ),
            "mf3zc": (
                "online cold-start calibrated-dissent horizon-consistent top-2 "
                "correction from three fixed MF3V final members"
            ),
            "mf3ze": (
                "online MF3V top-2 proposal filtered by the train-only, "
                "action-aligned counterfactual return safety gate"
            ),
            "mf3zf": (
                "online coverage-expanded MF3V top-2 proposal filtered by the "
                "train-only, action-aligned counterfactual return safety gate"
            ),
            "mf3zg": (
                "online core-preserving hierarchical MF3V top-2 proposal with "
                "independent train-only core and expansion return gates"
            ),
            "mf3zh": (
                "online MF3V uncertainty floor with a priority action-aligned "
                "core-preserving learned residual"
            ),
            "mf3y": (
                "online first-crossing horizon-consistent consensus-gated tail "
                "recovery from three fixed MF3V final members"
            ),
            "mf3v": (
                "online first-crossing horizon-consistent coverage-constrained "
                "policy-risk-adjusted rescue-versus-harm top-2 correction from "
                "three fixed MF3V final members"
            ),
            "mf3u": (
                "online first-crossing coverage-constrained policy-risk-adjusted "
                "rescue-versus-harm top-2 correction with a native-margin policy "
                "anchor from three fixed MF3U members"
            ),
            "mf3t": (
                "online first-crossing coverage-constrained policy-risk-adjusted "
                "rescue-versus-harm top-2 correction from the three fixed MF3T "
                "final members"
            ),
        }[REVISION],
        "secondary_treatments": (
            "native-margin uncertainty control"
        ),
        "primary_utility": "0.50*nDTW + 0.25*SDTW + 0.25*SPL",
        "uncertainty": "10000 deterministic scene bootstrap replicates",
        "success_gate": (
            "primary utility lower 95% bound > 0; success, SPL, nDTW "
            "point estimates non-negative; ensemble utility exceeds the "
            "uncertainty-only control"
        ),
        "shadow_gate_sha256": sha256_file(GATE),
        "worker_sha256": worker_sha,
        **({"reused_control_provenance": {
            "mf3v_result": sha256_file(
                ROOT / "artifacts/evaluation/mf3v_uad_rxr_val_seen_dev_v1/"
                "MF3V_RXR_VAL_SEEN_RESULT.json"
            ),
            "control_root": str(MF3V_CONTROLS.relative_to(ROOT)),
        }} if REVISION in ("mf3ze", "mf3zf", "mf3zg", "mf3zh") else {}),
        **({"controller_revision": (
            "mf3zh_uncertainty_floor_residual_v1" if REVISION == "mf3zh"
            else "mf3zg_core_preserving_hierarchy_v1" if REVISION == "mf3zg"
            else "mf3zf_coverage_safety_v1" if REVISION == "mf3zf"
            else "mf3ze_action_aligned_return_gate_v1" if REVISION == "mf3ze"
            else "mf3zc_calibrated_dissent_v1" if REVISION == "mf3zc"
            else "mf3zb_temporal_maturity_v1" if REVISION == "mf3zb"
            else "mf3za_consensus_band_tail_v1" if REVISION == "mf3za"
            else "mf3z_adaptive_relative_margin_tail_v1" if REVISION == "mf3z"
            else "mf3v_horizon_consistent_v1" if REVISION == "mf3v"
            else "mf3y_consensus_tail_v1" if REVISION == "mf3y"
            else "mf3u_policy_anchor_v1" if REVISION == "mf3u"
            else "mf3t_gate_schema_adapter_v1"
        )} if REVISION in ("mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh") else {}),
        "dataset_sha256": sha256_file(DATASET),
        "public_unseen_authorized": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed MF3 RxR val_seen protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    return 0


def jobs(rows: list[dict]):
    if REVISION in ("mf3ze", "mf3zf", "mf3zg", "mf3zh"):
        return [("ensemble", None, row) for row in rows]
    result = [("baseline", None, row) for row in rows]
    result.extend(("uncertainty", None, row) for row in rows)
    result.extend(("ensemble", None, row) for row in rows)
    return result


def name(mode, seed, row):
    return (
        f"{mode}_ep_{row['episode_id']}" if mode != "uad"
        else f"uad_seed_{seed}_ep_{row['episode_id']}"
    )


def execute(
    preflight: bool, gpus: tuple[int, ...], resume: bool,
    workers_per_gpu: int,
) -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("MF3 RxR val_seen protocol is not sealed")
    rows = json.loads(PROTOCOL.read_text())["selection"]
    planned = jobs(rows[:1]) if preflight else jobs(rows)
    root = OUT / ("preflight" if preflight else "full")
    root.mkdir(parents=True, exist_ok=resume)
    done = set()
    completed = []
    if resume:
        for run_dir in sorted((root / "runs").glob("*")):
            summary = run_dir / "RUN_SUMMARY.json"
            if summary.is_file() and json.loads(summary.read_text()).get("status") == "PASS":
                row = json.loads(summary.read_text())
                key = (row["mode"], row.get("seed"), row["episode_id"])
                done.add(key); completed.append({"key": key, "returncode": 0})
            else:
                interrupted = root / "interrupted"
                interrupted.mkdir(parents=True, exist_ok=True)
                destination = interrupted / run_dir.name
                suffix = 1
                while destination.exists():
                    destination = interrupted / f"{run_dir.name}_{suffix}"
                    suffix += 1
                os.replace(run_dir, destination)
    queue = [row for row in planned if (row[0], row[1], row[2]["episode_id"]) not in done]
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    free = sorted(
        gpu for gpu in set(gpus) for _ in range(workers_per_gpu)
    )
    active = []; started = time.time()
    while queue or active:
        while queue and free:
            mode, seed, row = queue.pop(0); gpu = free.pop(0)
            job_name = name(mode, seed, row)
            run_dir = root / "runs" / job_name
            logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
            stdout = (logs / f"{job_name}.stdout").open("w")
            stderr = (logs / f"{job_name}.stderr").open("w")
            command = [
                str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
                "--episode-id", row["episode_id"], "--mode", mode,
                "--revision", REVISION, "--run-dir", str(run_dir),
            ]
            if seed is not None:
                command.extend(("--seed", str(seed)))
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                           "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"}
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
            )
            active.append({
                "mode": mode, "seed": seed, "row": row, "gpu": gpu,
                "process": process, "streams": (stdout, stderr),
            })
        atomic_json(PROGRESS, {
            "status": "RUNNING", "preflight": preflight,
            "total": len(planned), "completed": len(completed),
            "failed": sum(row["returncode"] != 0 for row in completed),
            "queued": len(queue), "elapsed_s": round(time.time() - started, 1),
            "active": [{
                "gpu": row["gpu"], "mode": row["mode"], "seed": row["seed"],
                "episode_id": row["row"]["episode_id"],
            } for row in active],
        })
        time.sleep(1)
        for row in list(active):
            code = row["process"].poll()
            if code is None:
                continue
            for stream in row["streams"]:
                stream.close()
            completed.append({
                "key": (row["mode"], row["seed"], row["row"]["episode_id"]),
                "returncode": code,
            })
            free.append(row["gpu"]); free.sort(); active.remove(row)
    failures = [row for row in completed if row["returncode"]]
    atomic_json(PROGRESS, {
        "status": "COMPLETE" if not failures else "FAIL",
        "preflight": preflight, "total": len(planned),
        "completed": len(completed), "failed": len(failures),
        "queued": 0, "active": [], "elapsed_s": round(time.time() - started, 1),
    })
    return 0 if not failures else 2


def percentile(values, q):
    return sorted(values)[round((len(values) - 1) * q)]


def verify(preflight: bool) -> int:
    protocol = json.loads(PROTOCOL.read_text())
    rows = protocol["selection"][:1] if preflight else protocol["selection"]
    planned = jobs(rows) if preflight else jobs(rows)
    root = OUT / ("preflight" if preflight else "full")
    summaries = {}
    for mode, seed, row in planned:
        path = root / "runs" / name(mode, seed, row) / "RUN_SUMMARY.json"
        value = json.loads(path.read_text())
        if (
            value.get("status") != "PASS" or value.get("split") != "val_seen"
            or value.get("public_unseen_accessed") is not False
            or not isinstance(value.get("metrics"), dict)
        ):
            raise RuntimeError("MF3 RxR worker boundary or metric failure")
        if not all(math.isfinite(float(value["metrics"][key])) for key in METRICS):
            raise RuntimeError("MF3 RxR non-finite metric")
        if mode in ("ensemble", "uncertainty") and value[
            "executed_action_validation"
        ]["all_equal"] is not True:
            raise RuntimeError("MF3 controller declared/executed action mismatch")
        summaries[(mode, seed, row["episode_id"])] = value
    if REVISION in ("mf3ze", "mf3zf", "mf3zg", "mf3zh"):
        for row in rows:
            for mode in ("baseline", "uncertainty"):
                path = MF3V_CONTROLS / f"{mode}_ep_{row['episode_id']}" / "RUN_SUMMARY.json"
                value = json.loads(path.read_text())
                if not (
                    value.get("status") == "PASS"
                    and value.get("revision") == "mf3v"
                    and value.get("mode") == mode
                    and value.get("episode_id") == row["episode_id"]
                    and value.get("split") == "val_seen"
                    and value.get("public_unseen_accessed") is False
                    and isinstance(value.get("metrics"), dict)
                ):
                    raise RuntimeError(f"{TAG} reused MF3V control drift")
                summaries[(mode, None, row["episode_id"])] = value
    if preflight:
        passed = len(summaries) == 3
        atomic_json(OUT / f"{TAG}_RXR_VAL_SEEN_PREFLIGHT.json", {
            "status": "PREFLIGHT_PASS" if passed else "PREFLIGHT_FAIL",
            "runs": len(summaries), "public_unseen_authorized": False,
        })
        return 0 if passed else 2
    per_scene = []
    for row in rows:
        episode = row["episode_id"]
        baseline = summaries[("baseline", None, episode)]["metrics"]
        uncertainty = summaries[("uncertainty", None, episode)]["metrics"]
        treatment = summaries[("ensemble", None, episode)]["metrics"]
        delta = {metric: treatment[metric] - float(baseline[metric]) for metric in METRICS}
        delta["utility"] = (
            0.50 * delta["ndtw"] + 0.25 * delta["sdtw"] + 0.25 * delta["spl"]
        )
        learned_minus_uncertainty = {
            metric: treatment[metric] - float(uncertainty[metric])
            for metric in METRICS
        }
        learned_minus_uncertainty["utility"] = (
            0.50 * learned_minus_uncertainty["ndtw"]
            + 0.25 * learned_minus_uncertainty["sdtw"]
            + 0.25 * learned_minus_uncertainty["spl"]
        )
        per_scene.append({
            "scene_id": row["scene_id"], "episode_id": episode,
            **delta,
            **{f"learned_minus_uncertainty_{key}": value
               for key, value in learned_minus_uncertainty.items()},
        })
    rng = random.Random(20260828)
    metrics = (*METRICS, "utility")
    bootstrap = {metric: [] for metric in metrics}
    uncertainty_bootstrap = {metric: [] for metric in metrics}
    for _ in range(10000):
        sample = [per_scene[rng.randrange(len(per_scene))] for _ in per_scene]
        for metric in metrics:
            bootstrap[metric].append(sum(row[metric] for row in sample) / len(sample))
            key = f"learned_minus_uncertainty_{metric}"
            uncertainty_bootstrap[metric].append(
                sum(row[key] for row in sample) / len(sample)
            )
    aggregate = {
        metric: {
            "mean": sum(row[metric] for row in per_scene) / len(per_scene),
            "scene_bootstrap_95pct": [
                percentile(bootstrap[metric], 0.025),
                percentile(bootstrap[metric], 0.975),
            ],
        }
        for metric in metrics
    }
    learned_minus_uncertainty = {
        metric: {
            "mean": sum(
                row[f"learned_minus_uncertainty_{metric}"] for row in per_scene
            ) / len(per_scene),
            "scene_bootstrap_95pct": [
                percentile(uncertainty_bootstrap[metric], 0.025),
                percentile(uncertainty_bootstrap[metric], 0.975),
            ],
        }
        for metric in metrics
    }
    gates = {
        "utility_point_positive": aggregate["utility"]["mean"] > 0,
        "utility_lower_95_positive": aggregate["utility"]["scene_bootstrap_95pct"][0] > 0,
        "success_point_nonnegative": aggregate["success"]["mean"] >= 0,
        "spl_point_nonnegative": aggregate["spl"]["mean"] >= 0,
        "ndtw_point_nonnegative": aggregate["ndtw"]["mean"] >= 0,
        "learned_utility_exceeds_uncertainty": (
            learned_minus_uncertainty["utility"]["mean"] > 0
        ),
    }
    passed = all(gates.values())
    atomic_json(RESULT, {
        "schema_version": f"revealnav-{REVISION}-rxr-val-seen-result/1",
        "status": "TASK_METRIC_GATE_PASS" if passed else "TASK_METRIC_GATE_FAIL",
        "aggregate_uad_ensemble_minus_baseline": aggregate,
        "aggregate_uad_ensemble_minus_uncertainty": learned_minus_uncertainty,
        "per_scene": per_scene, "gates": gates,
        "public_unseen_authorized": False,
    })
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--revision", choices=("mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"), default="mf3s")
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--revision", choices=("mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"), default="mf3s")
    execute_parser.add_argument("--preflight", action="store_true")
    execute_parser.add_argument("--resume", action="store_true")
    execute_parser.add_argument("--gpus", default="0,1,2,3,4,5")
    execute_parser.add_argument("--workers-per-gpu", type=int, default=1)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--revision", choices=("mf3s", "mf3t", "mf3u", "mf3v", "mf3y", "mf3z", "mf3za", "mf3zb", "mf3zc", "mf3ze", "mf3zf", "mf3zg", "mf3zh"), default="mf3s")
    verify_parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    configure_revision(args.revision)
    if args.command == "seal":
        return seal()
    if args.command == "execute":
        return execute(args.preflight, tuple(
            int(value) for value in args.gpus.split(",") if value
        ), args.resume, args.workers_per_gpu)
    return verify(args.preflight)


if __name__ == "__main__":
    raise SystemExit(main())
