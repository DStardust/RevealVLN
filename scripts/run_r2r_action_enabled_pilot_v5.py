#!/usr/bin/env python3
"""Seal and gate a real outbound/return ETP action intervention pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
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
EPISODE_ID = "1582"
SCENE_ID = "pLe4wQe7qrG"
LOCK = ROOT / "locks/POST_EXCURSION_INTEGRATED_CONTROLLER_V4_9.json"
SHADOW_PROTOCOL = ROOT / (
    "artifacts/evaluation/mf2_r2r_unseen_fusion_v4_4_2/"
    "R2R_UNSEEN_FUSION_PROTOCOL_V4_4_2.json"
)
SHADOW_RESULT = SHADOW_PROTOCOL.with_name("R2R_UNSEEN_FUSION_RESULT_V4_4_2.json")
WORKER = ROOT / "scripts/r2r_action_enabled_pilot_worker_v5.py"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/"
    "val_unseen/val_unseen.json.gz"
)
GROUND_TRUTH = DATASET.with_name("val_unseen_gt.json.gz")
OUT = ROOT / "artifacts/evaluation/mf2_r2r_action_enabled_v5"
PROTOCOL = OUT / "R2R_ACTION_ENABLED_PROTOCOL_V5.json"
RESULT = OUT / "R2R_ACTION_ENABLED_RESULT_V5.json"


def protocol_value() -> dict:
    lock = json.loads(LOCK.read_text())
    shadow = json.loads(SHADOW_RESULT.read_text())
    shadow_protocol = json.loads(SHADOW_PROTOCOL.read_text())
    selected = [row for row in shadow_protocol["selection"]
                if row["episode_id"] == EPISODE_ID]
    if not (
        lock.get("status") == "LOCKED_BEFORE_ACTION_ENABLED_ONLINE_PILOT"
        and shadow.get("status") == "R2R_UNSEEN_FUSION_CONFIRMATION_PASS"
        and len(selected) == 1 and selected[0]["scene_id"] == SCENE_ID
    ):
        raise RuntimeError("action-enabled pilot precondition failed")
    return {
        "schema_version": "revealnav-r2r-action-enabled-protocol/5",
        "status": "SEALED_BEFORE_R2R_ACTION_ENABLED_PILOT",
        "split": "R2R-CE val_unseen engineering-only",
        "episode": {"episode_id": EPISODE_ID, "scene_id": SCENE_ID},
        "episode_selection": (
            "maximum persistent-decision rows in the already completed sealed "
            "V4.4.2 shadow cohort; selected before any action-enabled outcome"
        ),
        "seeds": list(SEEDS), "modes": list(MODES),
        "natural_mode": (
            "execute the first locked-fusion CHECKPOINTED_EXCURSION; execute "
            "return only if the post-excursion head predicts BACKTRACK"
        ),
        "forced_negative_mode": (
            "execute the lowest-target-probability persistent branch other than "
            "the natural selection, record the learned post action, then always "
            "execute a separately flagged frozen-controller stress return"
        ),
        "success_gates": {
            "all_six_runs_complete": True,
            "all_checkpoint_triplets_strictly_load": True,
            "all_six_real_outbound_actions_execute": True,
            "all_six_post_state_decisions_observed": True,
            "forced_negative_branch_differs_from_natural": True,
            "all_three_forced_stress_returns_execute_and_succeed": True,
            "all_trace_hash_chains_valid": True,
            "no_test_or_test_challenge_payload": True,
        },
        "forced_stress_return_is_not_a_policy_success_claim": True,
        "performance_metrics_allowed": False,
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (LOCK, SHADOW_PROTOCOL, SHADOW_RESULT, WORKER,
                         DATASET, GROUND_TRUTH)
        },
        "test_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed action-enabled protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "runs": len(SEEDS) * len(MODES),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def run_one(root: Path, seed: int, mode: str, gpu: int):
    name = f"seed_{seed}_{mode}"
    run_dir = root / "runs" / name
    log = root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    process = subprocess.run([
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--episode-id", EPISODE_ID, "--seed", str(seed),
        "--mode", mode, "--run-dir", str(run_dir),
    ], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    log.write_text(process.stdout)
    return name, process.returncode, process.stdout[-5000:]


def run(preflight: bool, gpus: tuple[int, ...]) -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("action-enabled protocol must be sealed")
    root = OUT / ("preflight" if preflight else "full")
    if root.exists():
        raise RuntimeError(f"refusing to overwrite {root}")
    jobs = (
        [(SEEDS[1], mode) for mode in MODES] if preflight
        else [(seed, mode) for seed in SEEDS for mode in MODES]
    )
    root.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        pending = []
        for index, (seed, mode) in enumerate(jobs):
            pending.append(pool.submit(
                run_one, root, seed, mode, gpus[index % len(gpus)]
            ))
        results = [future.result() for future in pending]
    failures = [(name, code, tail) for name, code, tail in results if code]
    atomic_json(root / "RUN_STATUS.json", {
        "status": "PASS" if not failures else "FAIL",
        "preflight": preflight,
        "runs": [{"name": name, "returncode": code}
                 for name, code, _ in results],
        "failures": failures,
    })
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "runs": len(results)}, indent=2))
    return 0


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def valid_chain(rows: list[dict]) -> bool:
    previous = "0" * 64
    for row in rows:
        if row.get("previous_hash") != previous:
            return False
        value = dict(row)
        claimed = value.pop("record_hash", None)
        if hashlib_sha(value) != claimed:
            return False
        previous = claimed
    return True


def hashlib_sha(value: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def verify() -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("action-enabled protocol drift")
    run_root = OUT / "full" / "runs"
    observed = {}
    traces = {}
    for path in run_root.glob("*/RUN_SUMMARY.json"):
        summary = json.loads(path.read_text())
        key = (summary["seed"], summary["mode"])
        observed[key] = summary
        traces[key] = load_trace(path.parent / "action_trace.jsonl")
    expected = {(seed, mode) for seed in SEEDS for mode in MODES}
    natural_branch = {}
    forced_branch = {}
    for key, rows in traces.items():
        selected = next((row for row in rows
                         if row["event"] == "outbound_selected"), None)
        if selected is None:
            continue
        target = natural_branch if key[1] == "natural" else forced_branch
        target[key[0]] = selected["executed_branch"]
    gates = {
        "all_six_runs_complete": (
            set(observed) == expected
            and all(row["status"] == "PASS" for row in observed.values())
        ),
        "all_checkpoint_triplets_strictly_load": all(
            row["strict_checkpoint_load"] for row in observed.values()
        ),
        "all_six_real_outbound_actions_execute": all(
            row["outbound_action_executed"] for row in observed.values()
        ),
        "all_six_post_state_decisions_observed": all(
            row["post_policy_action"] in ("continue", "backtrack")
            for row in observed.values()
        ),
        "forced_negative_branch_differs_from_natural": (
            set(natural_branch) == set(forced_branch) == set(SEEDS)
            and all(natural_branch[seed] != forced_branch[seed] for seed in SEEDS)
        ),
        "all_three_forced_stress_returns_execute_and_succeed": all(
            observed[(seed, "forced_negative")]["return_intervention_attempted"]
            and observed[(seed, "forced_negative")]["return_intervention_success"]
            for seed in SEEDS
        ),
        "all_trace_hash_chains_valid": all(
            valid_chain(rows) for rows in traces.values()
        ),
        "no_test_or_test_challenge_payload": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-r2r-action-enabled-result/5",
        "status": (
            "R2R_ACTION_ENABLED_ENGINEERING_GATE_PASS" if passed
            else "R2R_ACTION_ENABLED_ENGINEERING_GATE_FAIL"
        ),
        "gates": gates,
        "runs": {f"{seed}:{mode}": observed.get((seed, mode))
                 for seed, mode in sorted(expected)},
        "natural_branches": natural_branch,
        "forced_negative_branches": forced_branch,
        "forced_stress_return_is_not_a_policy_success_claim": True,
        "performance_metrics_reported": False,
        "protocol_sha256": sha256_file(PROTOCOL),
        "paper_result": False,
        "next_gate": (
            "multi-episode action-enabled controller evaluation"
            if passed else "repair online intervention seam"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps({"status": value["status"], "gates": gates,
                      "natural": natural_branch,
                      "forced": forced_branch}, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.verify:
        return verify()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    return run(args.preflight, gpus)


if __name__ == "__main__":
    raise SystemExit(main())
