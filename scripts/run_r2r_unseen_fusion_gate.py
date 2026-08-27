#!/usr/bin/env python3
"""Seal, run, and verify fresh R2R val_unseen REE+Q shadow confirmation."""

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
LOCK = ROOT / "locks/REE_Q_FUSION_CONTROLLER_V4_4.json"
WORKER = ROOT / "scripts/r2r_unseen_fusion_worker.py"
FUSION = ROOT / "revealnav_mf2r4/fusion.py"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/"
    "val_unseen/val_unseen.json.gz"
)
GROUND_TRUTH = ROOT / (
    "third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/"
    "val_unseen/val_unseen_gt.json.gz"
)
TRAIN_DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/"
    "train/train.json.gz"
)
EXTRACTION = ROOT / "artifacts/upstream/R2R_VAL_UNSEEN_EXTRACTION.json"
FAILED_PREFLIGHT = ROOT / (
    "artifacts/evaluation/mf2_r2r_unseen_fusion_v4_4/"
    "R2R_UNSEEN_GLOBAL_FRONTIER_PREFLIGHT_DIAGNOSIS_V4_4.json"
)
FAILED_PERSISTENCE_PREFLIGHT = ROOT / (
    "artifacts/evaluation/mf2_r2r_unseen_fusion_v4_4_1/"
    "R2R_UNSEEN_PERSISTENCE_PREFLIGHT_DIAGNOSIS_V4_4_1.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_r2r_unseen_fusion_v4_4_2"
PROTOCOL = OUT / "R2R_UNSEEN_FUSION_PROTOCOL_V4_4_2.json"
RESULT = OUT / "R2R_UNSEEN_FUSION_RESULT_V4_4_2.json"
SEEDS = (20260826, 20260827, 20260828)
GPUS = (0, 1)
SELECTION_SALT = "revealnav-r2r-unseen-fusion-confirmation/1"
EXCLUDED_PREFLIGHT_EPISODES = ("90", "262")


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
    episodes = read_gzip_json(DATASET).get("episodes", [])
    ids = [str(row["episode_id"]) for row in episodes]
    if len(episodes) != 1839 or len(ids) != len(set(ids)):
        raise RuntimeError("R2R val_unseen episode inventory drift")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        grouped[scene_id(row)].append(row)
    if len(grouped) != 11:
        raise RuntimeError("R2R val_unseen scene count drift")
    train_episodes = read_gzip_json(TRAIN_DATASET)["episodes"]
    train_scenes = {scene_id(row) for row in train_episodes}
    if train_scenes & set(grouped):
        raise RuntimeError("R2R train and val_unseen scenes overlap")
    ground_truth = read_gzip_json(GROUND_TRUTH)
    selected = []
    for scene in sorted(grouped):
        ranked = sorted(
            grouped[scene],
            key=lambda item: hashlib.sha256(
                f"{SELECTION_SALT}:{scene}:{item['episode_id']}".encode()
            ).hexdigest(),
        )
        row = next(
            item for item in ranked
            if str(item["episode_id"]) not in EXCLUDED_PREFLIGHT_EPISODES
        )
        episode_id = str(row["episode_id"])
        if episode_id not in ground_truth:
            raise RuntimeError("selected R2R episode lacks ground truth")
        glb = ROOT / f"third_party/ETP-R1/data/scene_datasets/mp3d/{scene}/{scene}.glb"
        navmesh = glb.with_suffix(".navmesh")
        if (
            glb.is_symlink() or not glb.is_file()
            or navmesh.is_symlink() or not navmesh.is_file()
        ):
            raise RuntimeError(f"selected scene assets missing: {scene}")
        selected.append({
            "episode_id": episode_id,
            "scene_id": scene,
            "selection_digest": hashlib.sha256(
                f"{SELECTION_SALT}:{scene}:{episode_id}".encode()
            ).hexdigest(),
        })
    return selected, {
        "val_unseen_episodes": len(episodes),
        "val_unseen_scenes": len(grouped),
        "selected_episodes": len(selected),
        "train_episodes": len(train_episodes),
        "train_scenes": len(train_scenes),
    }


def protocol_value() -> dict:
    lock = json.loads(LOCK.read_text())
    extraction = json.loads(EXTRACTION.read_text())
    failed_preflight = json.loads(FAILED_PREFLIGHT.read_text())
    failed_persistence = json.loads(FAILED_PERSISTENCE_PREFLIGHT.read_text())
    if not (
        lock.get("status") == "LOCKED_BEFORE_FRESH_R2R_UNSEEN_CONFIRMATION"
        and tuple(row["seed"] for row in lock["checkpoint_pairs"]) == SEEDS
        and extraction.get("status") == "R2R_VAL_UNSEEN_EXTRACTION_PASS"
        and extraction.get("raw_payload_extracted") is False
        and extraction.get("test_or_test_challenge_extracted") is False
        and failed_preflight.get("status")
        == "GLOBAL_FRONTIER_INTERFACE_MISMATCH_CONFIRMED"
        and failed_preflight.get("full_run_started") is False
        and failed_persistence.get("status")
        == "PERSISTENCE_CLOCK_INTERFACE_MISMATCH_CONFIRMED"
        and failed_persistence.get("full_run_started") is False
    ):
        raise RuntimeError("R2R fusion gate precondition failed")
    for pair in lock["checkpoint_pairs"]:
        for name in ("ree", "q"):
            row = pair[name]
            path = ROOT / row["path"]
            if (
                path.is_symlink() or not path.is_file()
                or path.stat().st_size != row["bytes"]
                or sha256_file(path) != row["sha256"]
            ):
                raise RuntimeError(f"locked {name} checkpoint drift")
    selected, counts = select_episodes()
    return {
        "schema_version": "revealnav-r2r-unseen-fusion-protocol/4.4.2",
        "status": "SEALED_BEFORE_FRESH_R2R_VAL_UNSEEN_MATCHED_FRONTIER_RUNS",
        "scope": "R2R-CE val_unseen cross-style engineering confirmation",
        "seeds": list(SEEDS),
        "physical_gpus": list(GPUS),
        "selection_salt": SELECTION_SALT,
        "excluded_interface_preflight_episodes": list(
            EXCLUDED_PREFLIGHT_EPISODES
        ),
        "selection": selected,
        "counts": counts,
        "controller": {
            "mode": "locked REE+Q fusion shadow; ETP-R1 remains authoritative",
            "formula": "q + 5.0 * (1 - p_target)",
            "persistence_k": 3,
            "persistence_definition": (
                "frozen ETP GraphMap ghost matched-observation count >= 3"
            ),
            "raw_q_recorded_on_same_base_trajectory": True,
            "candidate_scope": (
                "local ghost frontier whose ghost_fronts include the current "
                "ETP graph node; never the full global ghost memory"
            ),
            "threshold_or_checkpoint_tuning": False,
            "shadow_actions_executed": 0,
        },
        "engineering_gates": {
            "all_33_seed_episode_runs_complete": True,
            "all_six_checkpoints_strict_loaded": True,
            "all_controller_hash_chains_valid": True,
            "all_fused_costs_finite": True,
            "base_action_and_metric_invariant_across_shadow_seeds": True,
            "at_least_one_persistent_multibranch_decision_observed": True,
            "no_shadow_action_executed": True,
        },
        "predeclared_scientific_gates": {
            "at_least_10_aligned_persistent_decisions": True,
            "fused_all_three_branch_agreement_above_raw_q": True,
            "fused_all_three_branch_agreement_at_least_0_60": True,
        },
        "sources": {
            str(LOCK.relative_to(ROOT)): sha256_file(LOCK),
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            str(FUSION.relative_to(ROOT)): sha256_file(FUSION),
            str(EXTRACTION.relative_to(ROOT)): sha256_file(EXTRACTION),
            str(DATASET.relative_to(ROOT)): sha256_file(DATASET),
            str(GROUND_TRUTH.relative_to(ROOT)): sha256_file(GROUND_TRUTH),
            str(FAILED_PREFLIGHT.relative_to(ROOT)): sha256_file(FAILED_PREFLIGHT),
            str(FAILED_PERSISTENCE_PREFLIGHT.relative_to(ROOT)): sha256_file(
                FAILED_PERSISTENCE_PREFLIGHT
            ),
        },
        "forbidden_splits": ["test", "test_challenge"],
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed R2R fusion protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "episodes": len(value["selection"]),
        "runs": len(value["selection"]) * len(SEEDS),
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def launch(run_root: Path, seed: int, episode: dict, gpu: int) -> dict:
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
    process = subprocess.Popen([
        sys.executable, str(WORKER),
        "--episode-id", episode["episode_id"],
        "--seed", str(seed), "--exp-name", name,
        "--run-dir", str(run_dir),
    ], cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    return {
        "name": name, "seed": seed, "episode": episode, "gpu": gpu,
        "process": process, "streams": (stdout, stderr),
    }


def execute(preflight: bool, resume: bool = False) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("R2R fusion protocol must be sealed")
    protocol = json.loads(PROTOCOL.read_text())
    episodes = protocol["selection"][:1] if preflight else protocol["selection"]
    run_root = OUT / ("preflight" if preflight else "full")
    if run_root.exists() and not resume:
        raise RuntimeError(f"refusing to overwrite {run_root}")
    run_root.mkdir(parents=True, exist_ok=resume)
    completed = []
    completed_keys = set()
    if resume:
        interrupted = run_root / "interrupted"
        for run_dir in sorted((run_root / "runs").glob("*")):
            summary_path = run_dir / "RUN_SUMMARY.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                if summary.get("status") != "PASS":
                    raise RuntimeError(f"completed resume result is not PASS: {run_dir}")
                key = (int(summary["seed"]), str(summary["episode_id"]))
                completed_keys.add(key)
                completed.append({
                    "name": run_dir.name, "seed": key[0],
                    "episode_id": key[1], "gpu": None, "returncode": 0,
                    "recovered_complete": True,
                })
                continue
            destination = interrupted / run_dir.name
            interrupted.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError(f"interrupted evidence destination exists: {destination}")
            os.replace(run_dir, destination)
            for suffix in ("stdout.log", "stderr.log"):
                log = run_root / "logs" / f"{run_dir.name}.{suffix}"
                if log.exists():
                    os.replace(log, interrupted / log.name)
    queue = [
        (seed, episode) for seed in SEEDS for episode in episodes
        if (seed, episode["episode_id"]) not in completed_keys
    ]
    free_gpus = list(GPUS)
    active = []
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
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        raise RuntimeError("R2R fusion protocol drift")
    run_root = OUT / "full"
    expected = {
        (seed, row["episode_id"]) for seed in SEEDS for row in protocol["selection"]
    }
    observed = {}
    traces = {}
    evidence_files = []
    for path in sorted((run_root / "runs").glob("*/RUN_SUMMARY.json")):
        summary = json.loads(path.read_text())
        key = (summary["seed"], summary["episode_id"])
        if key in observed:
            raise RuntimeError("duplicate R2R fusion result")
        observed[key] = summary
        trace = load_jsonl(path.parent / "fusion_controller.jsonl")
        traces[key] = trace
        summary["_chain_valid"] = valid_chain(trace)
        evidence_files.extend((path, path.parent / "fusion_controller.jsonl"))
    all_rows = [row for trace in traces.values() for row in trace]
    engineering = {
        "all_33_seed_episode_runs_complete": (
            set(observed) == expected
            and all(row.get("status") == "PASS" for row in observed.values())
        ),
        "all_six_checkpoints_strict_loaded": all(
            row.get("controller", {}).get("strict_load") is True
            for row in observed.values()
        ),
        "all_controller_hash_chains_valid": all(
            row.get("_chain_valid") for row in observed.values()
        ),
        "all_fused_costs_finite": all(
            not row["decision_eligible"] or (
                row["predicted_fused_cost"] is not None
                and math.isfinite(row["predicted_fused_cost"])
            ) for row in all_rows
        ),
        "base_action_and_metric_invariant_across_shadow_seeds": True,
        "at_least_one_persistent_multibranch_decision_observed": any(
            row["decision_eligible"] for row in all_rows
        ),
        "no_shadow_action_executed": all(
            row.get("shadow_actions_executed") == 0 for row in observed.values()
        ) and all(row["shadow_only_not_executed"] for row in all_rows),
    }
    aligned = []
    per_episode = []
    for episode in protocol["selection"]:
        episode_id = episode["episode_id"]
        summaries = [observed[(seed, episode_id)] for seed in SEEDS]
        trace_hashes = {row["base_trace_sha256"] for row in summaries}
        metrics = [row["metrics"] for row in summaries]
        invariant = len(trace_hashes) == 1 and metrics[1:] == metrics[:-1]
        engineering["base_action_and_metric_invariant_across_shadow_seeds"] &= invariant
        by_seed = {
            seed: {row["step"]: row for row in traces[(seed, episode_id)]
                   if row["decision_eligible"]}
            for seed in SEEDS
        }
        common_steps = set.intersection(*(set(rows) for rows in by_seed.values()))
        for step in sorted(common_steps):
            values = [by_seed[seed][step] for seed in SEEDS]
            aligned.append({
                "fused_branch_agree": len({row["branch_id"] for row in values}) == 1,
                "raw_branch_agree": len({row["raw_q_branch_id"] for row in values}) == 1,
                "fused_action_agree": len({row["macro_action"] for row in values}) == 1,
                "raw_action_agree": len({row["raw_q_macro_action"] for row in values}) == 1,
            })
        per_episode.append({
            "episode_id": episode_id,
            "scene_id": episode["scene_id"],
            "base_invariant_across_seeds": invariant,
            "aligned_persistent_decisions": len(common_steps),
            "metrics": metrics[0],
        })
    fused_branch = statistics_mean(
        row["fused_branch_agree"] for row in aligned
    )
    raw_branch = statistics_mean(row["raw_branch_agree"] for row in aligned)
    scientific = {
        "at_least_10_aligned_persistent_decisions": len(aligned) >= 10,
        "fused_all_three_branch_agreement_above_raw_q": fused_branch > raw_branch,
        "fused_all_three_branch_agreement_at_least_0_60": fused_branch >= 0.60,
    }
    passed = all(engineering.values()) and all(scientific.values())
    evidence_digest = hashlib.sha256()
    for path in sorted(evidence_files):
        evidence_digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        evidence_digest.update(sha256_file(path).encode() + b"\0")
    value = {
        "schema_version": "revealnav-r2r-unseen-fusion-result/4.4.2",
        "status": (
            "R2R_UNSEEN_FUSION_CONFIRMATION_PASS" if passed
            else "R2R_UNSEEN_FUSION_CONFIRMATION_FAIL"
        ),
        "runs": len(observed),
        "engineering_gates": engineering,
        "scientific_gates": scientific,
        "stability": {
            "aligned_persistent_decisions": len(aligned),
            "fused_all_three_branch_agreement": fused_branch,
            "raw_q_all_three_branch_agreement": raw_branch,
            "fused_all_three_action_agreement": statistics_mean(
                row["fused_action_agree"] for row in aligned
            ),
            "raw_q_all_three_action_agreement": statistics_mean(
                row["raw_action_agree"] for row in aligned
            ),
        },
        "run_evidence": {
            "files": len(evidence_files),
            "path_sha256_chain": evidence_digest.hexdigest(),
        },
        "per_episode": per_episode,
        "protocol_sha256": sha256_file(PROTOCOL),
        "checkpoint_lock_sha256": sha256_file(LOCK),
        "controller_mode": "shadow_only",
        "shadow_actions_executed": 0,
        "test_or_test_challenge_accessed": False,
        "threshold_tuned_on_val_unseen": False,
        "paper_result": False,
        "next_gate": (
            "state-conditioned return executor" if passed
            else "preserve result and diagnose cross-style fusion"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if passed else 1


def statistics_mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("seal", "preflight", "run", "resume", "verify")
    )
    args = parser.parse_args()
    if args.mode == "seal":
        return seal()
    if args.mode == "preflight":
        return execute(True)
    if args.mode == "run":
        return execute(False)
    if args.mode == "resume":
        return execute(False, resume=True)
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
