#!/usr/bin/env python3
"""Seal and run the outcome-blind V5.4 OPP shadow diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/r2r_continuous_controller_worker_v5_4.py"
INTEGRATED = ROOT / "revealnav_mf2r4/integrated_controller.py"
CALIBRATION = ROOT / (
    "artifacts/evaluation/mf2_ecog_opp_shared_calibration_v3/"
    "RXR_ECOG_OPP_SHARED_CALIBRATION_RESULT_V3.json"
)
ACTIVE = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_3_activation_screen_v2/"
    "R2R_V5_3_ACTIVATION_SCREEN_PARTIAL_RESULT_V2.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_4_opp_shadow_seen_active_dev"
PROTOCOL = OUT / "R2R_V5_4_OPP_SHADOW_PROTOCOL.json"
RESULT = OUT / "R2R_V5_4_OPP_SHADOW_RESULT.json"
SEEDS = (20260826, 20260827, 20260828)


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


def protocol_value() -> dict:
    active = json.loads(ACTIVE.read_text())
    if not (
        active.get("status")
        == "PARTIAL_SCREEN_ENGINEERING_PASS_ACTIVE_COHORT_READY"
        and active.get("active_cohort_size") == 24
        and active.get("selection_used_task_metrics") is False
        and active.get("result_contains_task_metrics") is False
    ):
        raise RuntimeError("V5.3 outcome-blind active cohort is invalid")
    selection = [{
        "episode_id": str(row["episode_id"]),
        "scene_id": row["scene_id"],
        "trajectory_id": row.get("trajectory_id"),
    } for row in active["active_cohort"]]
    if len(selection) != len({row["episode_id"] for row in selection}):
        raise RuntimeError("active cohort contains duplicate episodes")
    return {
        "schema_version": "revealnav-r2r-v5.4-opp-shadow-protocol/1",
        "status": "SEALED_BEFORE_SHADOW_RUNS",
        "scope": "R2R val_seen internal repair cohort",
        "selection": selection,
        "seeds": list(SEEDS),
        "expected_runs": len(selection) * len(SEEDS),
        "actions": "frozen ETP-R1 only; V5.4 observes but never acts",
        "selection_uses_task_metrics": False,
        "worker_reads_task_metric_payload": False,
        "threshold_search_allowed": False,
        "sources": {
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            str(INTEGRATED.relative_to(ROOT)): sha256_file(INTEGRATED),
            str(CALIBRATION.relative_to(ROOT)): sha256_file(CALIBRATION),
            str(ACTIVE.relative_to(ROOT)): sha256_file(ACTIVE),
        },
        "paper_result": False,
    }


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.4 shadow protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "expected_runs": value["expected_runs"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }))


def name(seed: int, episode_id: str) -> str:
    return f"shadow_seed_{seed}_ep_{episode_id}"


def execute(gpus: tuple[int, ...], resume: bool) -> None:
    protocol = protocol_value()
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.4 shadow protocol must be sealed")
    run_root = OUT / "runs"
    log_root = OUT / "logs"
    if run_root.exists() and not resume:
        raise RuntimeError("shadow run directory exists; use --resume")
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    completed = []
    queue = []
    for seed in SEEDS:
        for row in protocol["selection"]:
            job = name(seed, row["episode_id"])
            summary = run_root / job / "RUN_SUMMARY.json"
            if resume and summary.is_file():
                value = json.loads(summary.read_text())
                if value.get("status") == "PASS":
                    completed.append({"name": job, "returncode": 0, "recovered": True})
                    continue
            if (run_root / job).exists():
                destination = OUT / "interrupted" / f"{job}_{int(time.time())}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(run_root / job, destination)
            queue.append((seed, row["episode_id"], job))
    free = list(gpus)
    active = []
    while queue or active:
        while queue and free:
            seed, episode_id, job = queue.pop(0)
            gpu = free.pop(0)
            stdout = (log_root / f"{job}.stdout.log").open("w")
            stderr = (log_root / f"{job}.stderr.log").open("w")
            command = [
                str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
                "--episode-id", episode_id, "--mode", "shadow",
                "--seed", str(seed), "--split", "val_seen",
                "--run-dir", str(run_root / job),
            ]
            process = subprocess.Popen(command, cwd=ROOT, env={
                **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            }, stdout=stdout, stderr=stderr)
            active.append({
                "name": job, "gpu": gpu, "process": process,
                "streams": (stdout, stderr),
            })
        time.sleep(0.5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({
                "name": item["name"], "gpu": item["gpu"],
                "returncode": code,
            })
            print(json.dumps(completed[-1]), flush=True)
            free.append(item["gpu"])
            free.sort()
            active.remove(item)
            atomic_json(OUT / "RUN_STATUS.json", {
                "status": "RUNNING" if queue or active else "COMPLETE",
                "completed_count": len(completed),
                "expected_runs": protocol["expected_runs"],
                "failures": [row for row in completed if row["returncode"]],
            })
    if any(row["returncode"] for row in completed):
        raise RuntimeError("one or more V5.4 shadow workers failed")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def valid_chain(rows: list[dict]) -> bool:
    previous = "0" * 64
    for row in rows:
        if row.get("previous_hash") != previous:
            return False
        value = dict(row)
        claimed = value.pop("record_hash", None)
        digest = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if digest != claimed:
            return False
        previous = claimed
    return True


def verify() -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("sealed V5.4 shadow protocol drift")
    summaries = []
    suppressions: dict[str, int] = {}
    by_seed = {}
    for path in sorted((OUT / "runs").glob("*/RUN_SUMMARY.json")):
        row = json.loads(path.read_text())
        trace = load_jsonl(path.parent / "controller_trace.jsonl")
        controller = row.get("controller") or {}
        if not (
            row.get("status") == "PASS"
            and row.get("mode") == "shadow"
            and row.get("task_metric_payload_read") is False
            and row.get("metrics") is None
            and controller.get("checkpointed_excursions") == 0
            and controller.get("continue_decisions") == 0
            and controller.get("backtrack_decisions") == 0
            and valid_chain(trace)
        ):
            raise RuntimeError(f"invalid shadow evidence: {path.parent.name}")
        summaries.append(row)
        seed = str(row["seed"])
        seed_row = by_seed.setdefault(seed, {
            "runs": 0, "ree_q_proposals": 0, "opp_accepted": 0,
        })
        seed_row["runs"] += 1
        accepted = int(controller["opp_checkpoint_acceptances"])
        rejected = sum(controller["opp_checkpoint_suppressions"].values())
        seed_row["ree_q_proposals"] += accepted + rejected
        seed_row["opp_accepted"] += accepted
        for reason, count in controller["opp_checkpoint_suppressions"].items():
            suppressions[reason] = suppressions.get(reason, 0) + count
    if len(summaries) != protocol["expected_runs"]:
        raise RuntimeError("V5.4 shadow run count is incomplete")
    proposals = sum(
        row["ree_q_proposals"] for row in by_seed.values()
    )
    accepted = sum(row["opp_accepted"] for row in by_seed.values())
    result = {
        "schema_version": "revealnav-r2r-v5.4-opp-shadow-result/1",
        "status": (
            "SHADOW_PASS_ACTION_GATE_READY" if accepted
            else "SHADOW_FAIL_ZERO_FULL_OPP_ACTIVATION"
        ),
        "protocol_sha256": sha256_file(PROTOCOL),
        "completed_runs": len(summaries),
        "ree_q_proposals": proposals,
        "opp_accepted": accepted,
        "opp_acceptance_rate": accepted / proposals if proposals else None,
        "opp_suppressions": suppressions,
        "by_seed": by_seed,
        "task_metric_payload_read": False,
        "actions_modified": False,
        "paper_result": False,
    }
    atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "execute", "verify", "all"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain unique GPU indices")
    if args.command in ("seal", "all"):
        seal()
    if args.command in ("execute", "all"):
        execute(gpus, args.resume)
    if args.command in ("verify", "all"):
        verify()


if __name__ == "__main__":
    main()
