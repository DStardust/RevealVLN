#!/usr/bin/env python3
"""Seal, run, and verify the locked V4 RxR val_unseen shadow gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "locks/RXR_UNSEEN_CHECKPOINT_LOCK_V4_2.json"
WORKER = ROOT / "scripts/rxr_unseen_controller_worker.py"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/"
    "val_unseen/val_unseen_guide.json.gz"
)
GROUND_TRUTH = ROOT / (
    "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/"
    "val_unseen/val_unseen_guide_gt.json.gz"
)
TRAIN_DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/"
    "train/train_guide.json.gz"
)
EXTRACTION = ROOT / "artifacts/upstream/RXR_VAL_UNSEEN_EXTRACTION.json"
OUT = ROOT / "artifacts/evaluation/mf2_unseen_controller_v4_2_1"
PROTOCOL = OUT / "RXR_UNSEEN_CONTROLLER_PROTOCOL_V4_2_1.json"
RESULT = OUT / "RXR_UNSEEN_CONTROLLER_RESULT_V4_2_1.json"
SEEDS = (20260826, 20260827, 20260828)
GPUS = (0, 1)
SELECTION_SALT = "revealnav-rxr-unseen-controller-pilot/1"


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


def read_gzip_json(path: Path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def scene_id(episode: dict) -> str:
    return Path(episode["scene_id"]).stem


def select_episodes() -> tuple[list[dict], dict]:
    payload = read_gzip_json(DATASET)
    episodes = payload.get("episodes", [])
    ids = [str(row["episode_id"]) for row in episodes]
    if len(episodes) != 11006 or len(ids) != len(set(ids)):
        raise RuntimeError("RxR val_unseen episode inventory drift")
    english = [
        row for row in episodes
        if row["instruction"]["language"] in {"en-US", "en-IN"}
    ]
    if len(english) != 3669:
        raise RuntimeError("RxR val_unseen English count drift")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in english:
        grouped[scene_id(row)].append(row)
    if len(grouped) != 11:
        raise RuntimeError("RxR val_unseen scene count drift")
    train_scenes = {
        scene_id(row) for row in read_gzip_json(TRAIN_DATASET)["episodes"]
    }
    if train_scenes & set(grouped):
        raise RuntimeError("train and val_unseen scenes overlap")
    ground_truth = read_gzip_json(GROUND_TRUTH)
    selected = []
    for scene in sorted(grouped):
        row = min(
            grouped[scene],
            key=lambda item: hashlib.sha256(
                f"{SELECTION_SALT}:{scene}:{item['episode_id']}".encode()
            ).hexdigest(),
        )
        episode_id = str(row["episode_id"])
        if episode_id not in ground_truth:
            raise RuntimeError("selected episode lacks ground truth")
        glb = ROOT / (
            f"third_party/ETP-R1/data/scene_datasets/mp3d/{scene}/{scene}.glb"
        )
        navmesh = glb.with_suffix(".navmesh")
        if not glb.is_file() or glb.is_symlink() or not navmesh.is_file() \
                or navmesh.is_symlink():
            raise RuntimeError(f"selected scene assets missing: {scene}")
        selected.append({
            "episode_id": episode_id,
            "scene_id": scene,
            "language": row["instruction"]["language"],
            "selection_digest": hashlib.sha256(
                f"{SELECTION_SALT}:{scene}:{episode_id}".encode()
            ).hexdigest(),
        })
    counts = {
        "all_guide_episodes": len(episodes),
        "english_guide_episodes": len(english),
        "unseen_scenes": len(grouped),
        "selected_episodes": len(selected),
        "train_scenes": len(train_scenes),
    }
    return selected, counts


def protocol_value() -> dict:
    lock = json.loads(LOCK.read_text())
    extraction = json.loads(EXTRACTION.read_text())
    if (
        lock.get("status") != "LOCKED_BEFORE_UNSEEN_EVALUATION"
        or tuple(row["seed"] for row in lock["checkpoints"]) != SEEDS
        or extraction.get("status") != "RXR_VAL_UNSEEN_EXTRACTION_PASS"
        or extraction.get("test_or_test_challenge_extracted") is not False
    ):
        raise RuntimeError("unseen controller gate precondition failed")
    for row in lock["checkpoints"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError("locked checkpoint drift")
    selected, counts = select_episodes()
    return {
        "schema_version": "revealnav-rxr-unseen-controller-protocol/4.2.1",
        "status": "SEALED_BEFORE_RXR_VAL_UNSEEN_CONTROLLER_RUNS",
        "scope": "RxR-CE English guide val_unseen engineering pilot",
        "checkpoint_lock_commit": "5eef9a7e95b267fd87cd6a7ec0c06af8d39b1403",
        "seeds": list(SEEDS),
        "physical_gpus": list(GPUS),
        "selection_salt": SELECTION_SALT,
        "selection": selected,
        "counts": counts,
        "controller": {
            "mode": "locked V4 macro-action shadow; ETP-R1 remains authoritative",
            "actions": ["commit", "checkpointed_excursion"],
            "persistence_k": 3,
            "persistence_definition": (
                "each persistent branch id must be consecutively observed K times; "
                "new unstable branches do not reset established identities"
            ),
            "choice": "global minimum predicted macro cost",
            "exact_tie": "prefer commit, then lexical branch id",
            "threshold_tuning_on_val_unseen": False,
            "shadow_actions_executed": 0,
        },
        "success_gates": {
            "all_33_seed_episode_runs_complete": True,
            "all_checkpoints_strict_loaded": True,
            "all_controller_hash_chains_valid": True,
            "all_controller_costs_finite": True,
            "base_action_and_metric_invariant_across_shadow_seeds": True,
            "at_least_one_persistent_multibranch_decision_observed": True,
            "no_shadow_action_executed": True,
        },
        "sources": {
            str(LOCK.relative_to(ROOT)): sha256_file(LOCK),
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            "revealnav_mf2r4/controller.py": sha256_file(
                ROOT / "revealnav_mf2r4/controller.py"
            ),
            "revealnav_mf2r4/model.py": sha256_file(
                ROOT / "revealnav_mf2r4/model.py"
            ),
            str(EXTRACTION.relative_to(ROOT)): sha256_file(EXTRACTION),
            str(DATASET.relative_to(ROOT)): sha256_file(DATASET),
            str(GROUND_TRUTH.relative_to(ROOT)): sha256_file(GROUND_TRUTH),
        },
        "forbidden_splits": ["test", "test_challenge"],
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != value:
            raise RuntimeError("sealed unseen controller protocol drift")
    else:
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "episodes": len(value["selection"]),
        "runs": len(value["selection"]) * len(SEEDS),
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def launch(run_root: Path, seed: int, episode: dict, gpu: int):
    name = f"seed_{seed}_ep_{episode['episode_id']}"
    run_dir = run_root / "runs" / name
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"{name}.stdout.log").open("w")
    stderr = (logs / f"{name}.stderr.log").open("w")
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    command = [
        sys.executable, str(WORKER),
        "--episode-id", episode["episode_id"],
        "--seed", str(seed),
        "--exp-name", name,
        "--run-dir", str(run_dir),
    ]
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
    )
    return {
        "name": name, "seed": seed, "episode": episode,
        "gpu": gpu, "process": process, "streams": (stdout, stderr),
    }


def run(preflight: bool) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("unseen controller protocol must be sealed")
    protocol = json.loads(PROTOCOL.read_text())
    episodes = protocol["selection"][:1] if preflight else protocol["selection"]
    run_root = OUT / ("preflight" if preflight else "full")
    if run_root.exists():
        raise RuntimeError(f"refusing to overwrite {run_root}")
    run_root.mkdir(parents=True)
    queue = [(seed, episode) for seed in SEEDS for episode in episodes]
    free_gpus = list(GPUS)
    active = []
    completed = []
    while queue or active:
        while queue and free_gpus:
            seed, episode = queue.pop(0)
            gpu = free_gpus.pop(0)
            active.append(launch(run_root, seed, episode, gpu))
        time.sleep(1)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({
                "name": item["name"], "seed": item["seed"],
                "episode_id": item["episode"]["episode_id"],
                "gpu": item["gpu"], "returncode": code,
            })
            active.remove(item)
            free_gpus.append(item["gpu"])
            free_gpus.sort()
            print(json.dumps(completed[-1]), flush=True)
    failures = [row for row in completed if row["returncode"] != 0]
    atomic_json(run_root / "RUN_STATUS.json", {
        "status": "PASS" if not failures else "FAIL",
        "preflight": preflight,
        "completed": completed,
        "failures": failures,
    })
    return 0 if not failures else 1


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def valid_chain(rows: list[dict]) -> bool:
    previous = "0" * 64
    for row in rows:
        if row.get("previous_hash") != previous:
            return False
        claimed = row.get("record_hash")
        value = dict(row)
        value.pop("record_hash", None)
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != claimed:
            return False
        previous = claimed
    return True


def verify() -> int:
    protocol = protocol_value()
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("unseen controller protocol drift")
    run_root = OUT / "full"
    expected = {
        (seed, row["episode_id"])
        for seed in SEEDS for row in protocol["selection"]
    }
    observed = {}
    traces_by_key = {}
    all_rows = []
    for summary_path in sorted((run_root / "runs").glob("*/RUN_SUMMARY.json")):
        summary = json.loads(summary_path.read_text())
        key = (summary["seed"], summary["episode_id"])
        if key in observed:
            raise RuntimeError("duplicate unseen seed/episode result")
        observed[key] = summary
        trace = load_jsonl(summary_path.parent / "v4_controller.jsonl")
        traces_by_key[key] = trace
        all_rows.extend(trace)
        summary["_chain_valid"] = valid_chain(trace)
    checks = {
        "all_33_seed_episode_runs_complete": (
            set(observed) == expected
            and all(row.get("status") == "PASS" for row in observed.values())
        ),
        "all_checkpoints_strict_loaded": all(
            row.get("controller", {}).get("strict_load") is True
            for row in observed.values()
        ),
        "all_controller_hash_chains_valid": all(
            row.get("_chain_valid") for row in observed.values()
        ),
        "all_controller_costs_finite": all(
            (not row["decision_eligible"])
            or (
                row["predicted_cost"] is not None
                and math.isfinite(row["predicted_cost"])
                and row["preservation_gain"] is not None
                and math.isfinite(row["preservation_gain"])
            )
            for row in all_rows
        ),
        "base_action_and_metric_invariant_across_shadow_seeds": True,
        "at_least_one_persistent_multibranch_decision_observed": any(
            row["decision_eligible"] for row in all_rows
        ),
        "no_shadow_action_executed": all(
            row.get("shadow_actions_executed") == 0
            for row in observed.values()
        ) and all(row["shadow_only_not_executed"] for row in all_rows),
    }
    per_episode = []
    for episode in protocol["selection"]:
        episode_id = episode["episode_id"]
        group = [observed[(seed, episode_id)] for seed in SEEDS]
        trace_hashes = {row["base_trace_sha256"] for row in group}
        metrics = [row["metrics"] for row in group]
        invariant = len(trace_hashes) == 1 and metrics[1:] == metrics[:-1]
        checks["base_action_and_metric_invariant_across_shadow_seeds"] &= invariant
        per_episode.append({
            "episode_id": episode_id,
            "scene_id": episode["scene_id"],
            "base_invariant_across_seeds": invariant,
            "high_level_trace_sha256": next(iter(trace_hashes)),
            "metrics": metrics[0],
        })
    per_seed = []
    for seed in SEEDS:
        group = [row for (s, _), row in observed.items() if s == seed]
        traces = [
            record
            for (s, episode_id), _ in observed.items() if s == seed
            for record in traces_by_key[(s, episode_id)]
        ]
        per_seed.append({
            "seed": seed,
            "episodes": len(group),
            "controller_rows": len(traces),
            "decision_rows": sum(row["decision_eligible"] for row in traces),
            "checkpointed_excursion_rows": sum(
                row["macro_action"] == "checkpointed_excursion" for row in traces
            ),
            "commit_rows": sum(row["macro_action"] == "commit" for row in traces),
        })
    passed = all(checks.values())
    value = {
        "schema_version": "revealnav-rxr-unseen-controller-result/4.2.1",
        "status": "RXR_UNSEEN_CONTROLLER_GATE_PASS" if passed else
                  "RXR_UNSEEN_CONTROLLER_GATE_FAIL",
        "checks": checks,
        "runs": len(observed),
        "per_seed": per_seed,
        "per_episode": per_episode,
        "protocol_sha256": sha256_file(PROTOCOL),
        "checkpoint_lock_sha256": sha256_file(LOCK),
        "controller_mode": "shadow_only",
        "shadow_actions_executed": 0,
        "test_or_test_challenge_accessed": False,
        "threshold_tuned_on_val_unseen": False,
        "paper_result": False,
        "next_gate": (
            "freeze a state-conditioned return executor before closed-loop runs"
            if passed else "diagnose unseen controller integration"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "preflight", "run", "verify"))
    args = parser.parse_args()
    if args.mode == "seal":
        return seal()
    if args.mode == "preflight":
        return run(True)
    if args.mode == "run":
        return run(False)
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
