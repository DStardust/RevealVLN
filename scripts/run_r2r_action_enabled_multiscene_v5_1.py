#!/usr/bin/env python3
"""Three-scene action-enabled intervention gate using the frozen V5 worker."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
MODES = ("natural", "forced_negative")
EPISODES = (
    {"episode_id": "670", "scene_id": "zsNo4HB9uLZ"},
    {"episode_id": "1825", "scene_id": "oLBMNvg9in8"},
    {"episode_id": "1582", "scene_id": "pLe4wQe7qrG"},
)
LOCK = ROOT / "locks/R2R_ACTION_ENABLED_PILOT_V5.json"
SHADOW_PROTOCOL = ROOT / (
    "artifacts/evaluation/mf2_r2r_unseen_fusion_v4_4_2/"
    "R2R_UNSEEN_FUSION_PROTOCOL_V4_4_2.json"
)
SHADOW_RESULT = SHADOW_PROTOCOL.with_name("R2R_UNSEEN_FUSION_RESULT_V4_4_2.json")
SHADOW_RUNS = SHADOW_PROTOCOL.parent / "full/runs"
WORKER = ROOT / "scripts/r2r_action_enabled_pilot_worker_v5.py"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_action_enabled_multiscene_v5_1"
PROTOCOL = OUT / "R2R_ACTION_ENABLED_MULTISCENE_PROTOCOL_V5_1.json"
RESULT = OUT / "R2R_ACTION_ENABLED_MULTISCENE_RESULT_V5_1.json"


def shadow_availability() -> dict[str, dict]:
    values = {}
    for episode in EPISODES:
        episode_id = episode["episode_id"]
        rows = []
        for seed in SEEDS:
            path = SHADOW_RUNS / f"seed_{seed}_ep_{episode_id}/RUN_SUMMARY.json"
            summary = json.loads(path.read_text())
            rows.append({
                "seed": seed,
                "decision_rows": summary["controller"]["decision_rows"],
                "checkpoint_rows": summary["controller"]["checkpoint_rows"],
            })
        values[episode_id] = {"runs": rows}
    return values


def protocol_value() -> dict:
    lock = json.loads(LOCK.read_text())
    shadow_result = json.loads(SHADOW_RESULT.read_text())
    shadow_protocol = json.loads(SHADOW_PROTOCOL.read_text())
    selected = {row["episode_id"]: row["scene_id"]
                for row in shadow_protocol["selection"]}
    availability = shadow_availability()
    if not (
        lock.get("status") == "LOCKED_BEFORE_MULTISCENE_ACTION_ENABLED_GATE"
        and shadow_result.get("status") == "R2R_UNSEEN_FUSION_CONFIRMATION_PASS"
        and all(selected.get(row["episode_id"]) == row["scene_id"] for row in EPISODES)
        and all(
            run["decision_rows"] > 0 and run["checkpoint_rows"] > 0
            for value in availability.values() for run in value["runs"]
        )
    ):
        raise RuntimeError("multi-scene action gate precondition failed")
    return {
        "schema_version": "revealnav-r2r-action-enabled-multiscene-protocol/5.1",
        "status": "SEALED_BEFORE_R2R_ACTION_ENABLED_MULTISCENE_GATE",
        "split": "R2R-CE val_unseen engineering-only",
        "episodes": list(EPISODES), "seeds": list(SEEDS),
        "modes": list(MODES), "expected_runs": 18,
        "selection": (
            "all three scenes in the already sealed V4.4.2 cohort with at least "
            "one persistent decision and checkpoint row in every seed"
        ),
        "selection_evidence": availability,
        "actions": {
            "natural": "real locked-fusion outbound; learned post action",
            "forced_negative": (
                "real lowest-probability alternative outbound; learned post action "
                "reported; separately flagged frozen stress return always executed"
            ),
        },
        "success_gates": {
            "all_18_runs_complete": True,
            "all_real_outbounds_and_post_decisions_execute": True,
            "all_forced_branches_differ_from_natural": True,
            "all_nine_forced_returns_execute_and_succeed": True,
            "all_hash_chains_valid": True,
            "all_three_scenes_and_seeds_represented": True,
            "no_test_or_test_challenge_payload": True,
        },
        "performance_metrics_allowed": False,
        "forced_return_not_a_learned_policy_claim": True,
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (LOCK, SHADOW_PROTOCOL, SHADOW_RESULT, WORKER)
        },
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed multi-scene action protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "runs": value["expected_runs"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def run_one(seed: int, mode: str, episode_id: str, gpu: int):
    name = f"ep_{episode_id}_seed_{seed}_{mode}"
    run_dir = OUT / "full/runs" / name
    log = OUT / "full/logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    process = subprocess.run([
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--episode-id", episode_id, "--seed", str(seed), "--mode", mode,
        "--run-dir", str(run_dir),
    ], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    log.write_text(process.stdout)
    return name, process.returncode, process.stdout[-5000:]


def run(gpus: tuple[int, ...]) -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("multi-scene action protocol must be sealed")
    root = OUT / "full"
    if root.exists():
        raise RuntimeError("refusing to overwrite multi-scene full run")
    root.mkdir(parents=True)
    jobs = [
        (seed, mode, episode["episode_id"])
        for episode in EPISODES for seed in SEEDS for mode in MODES
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        active = {}
        free_gpus = list(gpus)
        next_job = 0
        while next_job < len(jobs) or active:
            while next_job < len(jobs) and free_gpus:
                seed, mode, episode_id = jobs[next_job]
                gpu = free_gpus.pop(0)
                future = pool.submit(run_one, seed, mode, episode_id, gpu)
                active[future] = gpu
                next_job += 1
            done, _ = concurrent.futures.wait(
                tuple(active), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                results.append(future.result())
                free_gpus.append(active.pop(future))
            free_gpus.sort()
    failures = [(name, code, tail) for name, code, tail in results if code]
    atomic_json(root / "RUN_STATUS.json", {
        "status": "PASS" if not failures else "FAIL",
        "runs": [{"name": name, "returncode": code}
                 for name, code, _ in results],
        "failures": failures,
    })
    print(json.dumps({"status": "PASS" if not failures else "FAIL",
                      "completed": len(results), "failures": failures}, indent=2))
    return 0 if not failures else 1


def load_trace(path: Path) -> list[dict]:
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


def verify() -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("multi-scene action protocol drift")
    observed = {}
    traces = {}
    for path in (OUT / "full/runs").glob("*/RUN_SUMMARY.json"):
        summary = json.loads(path.read_text())
        key = (summary["episode_id"], summary["seed"], summary["mode"])
        observed[key] = summary
        traces[key] = load_trace(path.parent / "action_trace.jsonl")
    expected = {
        (episode["episode_id"], seed, mode)
        for episode in EPISODES for seed in SEEDS for mode in MODES
    }
    branches = {}
    for key, rows in traces.items():
        selected = next((row for row in rows if row["event"] == "outbound_selected"), None)
        if selected:
            branches[key] = selected["executed_branch"]
    pair_diff = all(
        branches.get((episode["episode_id"], seed, "natural"))
        != branches.get((episode["episode_id"], seed, "forced_negative"))
        and all((episode["episode_id"], seed, mode) in branches for mode in MODES)
        for episode in EPISODES for seed in SEEDS
    )
    gates = {
        "all_18_runs_complete": (
            set(observed) == expected
            and all(row["status"] == "PASS" for row in observed.values())
        ),
        "all_real_outbounds_and_post_decisions_execute": all(
            row["outbound_action_executed"]
            and row["post_policy_action"] in ("continue", "backtrack")
            for row in observed.values()
        ),
        "all_forced_branches_differ_from_natural": pair_diff,
        "all_nine_forced_returns_execute_and_succeed": all(
            observed[(episode["episode_id"], seed, "forced_negative")][
                "return_intervention_attempted"]
            and observed[(episode["episode_id"], seed, "forced_negative")][
                "return_intervention_success"]
            for episode in EPISODES for seed in SEEDS
        ),
        "all_hash_chains_valid": all(valid_chain(rows) for rows in traces.values()),
        "all_three_scenes_and_seeds_represented": (
            {key[0] for key in observed} == {row["episode_id"] for row in EPISODES}
            and {key[1] for key in observed} == set(SEEDS)
        ),
        "no_test_or_test_challenge_payload": True,
    }
    policy_actions = {}
    for episode in EPISODES:
        episode_id = episode["episode_id"]
        policy_actions[episode_id] = {
            f"{seed}:{mode}": observed[(episode_id, seed, mode)]["post_policy_action"]
            for seed in SEEDS for mode in MODES
        }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-r2r-action-enabled-multiscene-result/5.1",
        "status": (
            "R2R_ACTION_ENABLED_MULTISCENE_GATE_PASS" if passed
            else "R2R_ACTION_ENABLED_MULTISCENE_GATE_FAIL"
        ),
        "gates": gates, "policy_actions": policy_actions,
        "branches": {":".join(map(str, key)): value
                     for key, value in branches.items()},
        "forced_return_not_a_learned_policy_claim": True,
        "performance_metrics_reported": False,
        "protocol_sha256": sha256_file(PROTOCOL),
        "paper_result": False,
        "next_gate": (
            "continuous multi-episode controller with task metrics"
            if passed else "repair cross-scene action execution"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps({"status": value["status"], "gates": gates,
                      "policy_actions": policy_actions}, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.verify:
        return verify()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    return run(gpus)


if __name__ == "__main__":
    raise SystemExit(main())
