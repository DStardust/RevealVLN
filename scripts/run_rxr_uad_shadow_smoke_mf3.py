#!/usr/bin/env python3
"""Run a small persistent RxR-train UAD shadow cohort on multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
WORKER = ROOT / "scripts/rxr_uad_shadow_worker_mf3.py"


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    episodes = tuple(value.strip() for value in args.episodes.split(",") if value.strip())
    gpus = tuple(int(value) for value in args.gpus.split(","))
    output = args.output_dir.resolve()
    if (
        not episodes or len(set(episodes)) != len(episodes) or not gpus
        or ROOT not in output.parents or output.exists()
    ):
        raise SystemExit("invalid episodes, GPUs, or new project-local output")
    output.mkdir(parents=True)
    progress_path = output / "MF3B_UAD_SHADOW_PROGRESS.json"
    pending = deque(episodes)
    active = {}
    completed = []
    failed = []

    def write_progress() -> None:
        atomic_json(progress_path, {
            "schema_version": "revealnav-mf3b-uad-shadow-progress/1",
            "status": "RUNNING" if pending or active else "COMPLETE",
            "total": len(episodes),
            "completed": len(completed),
            "failed": list(failed),
            "remaining": len(pending),
            "active": {
                str(gpu): {"episode_id": value["episode"], "pid": value["process"].pid}
                for gpu, value in active.items()
            },
            "updated_at_unix": time.time(),
        })

    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            episode = pending.popleft()
            run_dir = output / f"ep_{episode}"
            stdout = (output / f"ep_{episode}.stdout").open("w")
            stderr = (output / f"ep_{episode}.stderr").open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                [
                    str(PYTHON), str(WORKER), "--episode-id", episode,
                    "--run-dir", str(run_dir),
                ],
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            active[gpu] = {
                "episode": episode,
                "process": process,
                "stdout": stdout,
                "stderr": stderr,
            }
        write_progress()
        if active:
            time.sleep(1)
        for gpu, value in list(active.items()):
            return_code = value["process"].poll()
            if return_code is None:
                continue
            value["stdout"].close()
            value["stderr"].close()
            episode = value["episode"]
            if return_code == 0:
                completed.append(episode)
            else:
                failed.append({"episode_id": episode, "return_code": return_code})
            del active[gpu]

    summaries = []
    for episode in completed:
        path = output / f"ep_{episode}/RUN_SUMMARY.json"
        summaries.append(json.loads(path.read_text()))
    outcomes = {str(seed): Counter() for seed in (20260826, 20260827, 20260828)}
    for summary in summaries:
        for seed, counts in summary["outcome_counts"].items():
            outcomes[seed].update(counts)
    result = {
        "schema_version": "revealnav-mf3b-uad-shadow-cohort/1",
        "status": (
            "SHADOW_COHORT_PASS"
            if not failed and len(summaries) == len(episodes)
            and all(row["status"] == "SHADOW_PASS" for row in summaries)
            else "FAIL"
        ),
        "episodes": list(episodes),
        "episode_count": len(summaries),
        "decision_rows": sum(row["decision_rows"] for row in summaries),
        "verified_native_decisions": sum(
            row["native_action_verification"]["checked_decisions"]
            for row in summaries
        ),
        "actions_changed": sum(row["actions_changed"] for row in summaries),
        "outcome_counts": {
            seed: dict(sorted(counts.items())) for seed, counts in outcomes.items()
        },
        "failed": failed,
        "gate_decision": "PENDING_COHORT_EVIDENCE_NOT_POLICY_AUTHORIZATION",
    }
    atomic_json(output / "MF3B_UAD_SHADOW_COHORT.json", result)
    write_progress()
    return 0 if result["status"] == "SHADOW_COHORT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
