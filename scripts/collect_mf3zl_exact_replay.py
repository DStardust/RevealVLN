#!/usr/bin/env python3
"""Seal, execute, assemble, and audit MF3ZL dense exact replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.exact_replay import (  # noqa: E402
    ProposalEventIdentity,
    validate_exact_prefix,
    validate_forced_switch,
    validate_shadow_event,
)


OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1"
PROTOCOL = OUT / "MF3ZL_RCSP_PROTOCOL.json"
SELECTION = OUT / "MF3ZL_EXACT_REPLAY_SELECTION.json"
TARGETS = OUT / "MF3ZL_EXACT_REPLAY_TARGETS.json"
MANIFEST = OUT / "MF3ZL_EXACT_REPLAY_MANIFEST.json"
AUDIT = OUT / "MF3ZL_DATA_SUPPORT_AUDIT.json"
NATIVE_PROGRESS = OUT / "MF3ZL_NATIVE_SHADOW_PROGRESS.json"
TARGET_PROGRESS = OUT / "MF3ZL_TARGETED_SWITCH_PROGRESS.json"
WORKER = ROOT / "scripts/mf3zl_exact_replay_worker.py"
PYTHON = ROOT / ".envs/etpr1/bin/python"
DSR_PROTOCOL = OUT.parent / "mf3zk_dsr_v1/MF3ZK_DSR_PROTOCOL.json"
R2R_DATA = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz"
)
RXR_DATA = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
R2R_OLD_SELECTION = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_collection/"
    "MF3ZK_R2R_COLLECTION_SELECTION.json"
)
RXR_OLD_SELECTIONS = (
    ROOT / "artifacts/phase1/mf3zd_direct_switch_returns_v1/MF3ZD_DIRECT_SWITCH_SELECTION.json",
    ROOT / "artifacts/phase1/mf3zf_expanded_direct_switch_returns_v1/MF3ZF_DIRECT_SWITCH_SELECTION.json",
)
SCHEMA = "revealnav-mf3zl-exact-replay"
UTILITY_WEIGHTS = {"ndtw": 0.50, "sdtw": 0.25, "spl": 0.25}
PUBLIC_TOKENS = {"val_seen", "val_unseen", "test", "test_challenge"}
STOP_REQUESTED = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError(f"stale atomic partial: {part}")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if ROOT not in resolved.parents or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid project-local source: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _scene(episode: dict) -> str:
    parts = str(episode["scene_id"]).split("/")
    if len(parts) < 2 or len(parts[-2]) != 11:
        raise RuntimeError("train episode scene identity drift")
    return parts[-2]


def _sort_id(value: object) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _load_episodes(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        value = json.load(stream)
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"invalid train episode payload: {path}")
    return episodes


def _used_trajectories(
    r2r_episodes: list[dict], rxr_episodes: list[dict],
) -> dict[str, set[tuple[str, str]]]:
    r2r = json.loads(R2R_OLD_SELECTION.read_text())
    used_r2r = {
        (str(row["scene_id"]), str(row["trajectory_id"]))
        for row in r2r["routes"]
    }
    selected_rxr_episodes = set()
    for path in RXR_OLD_SELECTIONS:
        value = json.loads(path.read_text())
        selected_rxr_episodes.update(str(row["episode_id"]) for row in value["selection"])
    episode_lookup = {str(row["episode_id"]): row for row in rxr_episodes}
    if not selected_rxr_episodes <= set(episode_lookup):
        raise RuntimeError("old RxR selection episode is absent from train payload")
    used_rxr = {
        (_scene(episode_lookup[episode]), str(episode_lookup[episode]["trajectory_id"]))
        for episode in selected_rxr_episodes
    }
    # The DSR canonical inventory is an independent closure check. It cannot
    # add outcome-dependent routes; it can only enlarge the historical exclude set.
    dsr = json.loads(DSR_PROTOCOL.read_text())
    lookups = {
        "RxR": {str(row["episode_id"]): row for row in rxr_episodes},
        "R2R": {str(row["episode_id"]): row for row in r2r_episodes},
    }
    used = {"RxR": used_rxr, "R2R": used_r2r}
    for item in dsr["source_inventory"]["canonical_rows"]:
        identity = item["identity"]
        dataset = str(identity["dataset"])
        episode = lookups[dataset].get(str(identity["episode_id"]))
        if episode is None:
            raise RuntimeError("DSR canonical episode is absent from train payload")
        used[dataset].add((_scene(episode), str(episode["trajectory_id"])))
    return used


def _representatives(
    dataset: str,
    episodes: list[dict],
    allowed_scenes: set[str],
    excluded: set[tuple[str, str]],
) -> tuple[list[dict], dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    eligible_before_exclusion = set()
    short_routes = set()
    non_english_routes = set()
    for episode in episodes:
        scene = _scene(episode)
        trajectory = str(episode["trajectory_id"])
        key = (scene, trajectory)
        if scene not in allowed_scenes:
            continue
        if len(episode.get("reference_path", [])) < 4:
            short_routes.add(key)
            continue
        language = episode.get("instruction", {}).get("language")
        if dataset == "RxR" and language not in {"en-US", "en-IN"}:
            non_english_routes.add(key)
            continue
        eligible_before_exclusion.add(key)
        if key not in excluded:
            groups[key].append(episode)
    selected = []
    language_order = {"en-US": 0, "en-IN": 1, None: 0}
    for (scene, trajectory), candidates in sorted(groups.items()):
        candidates.sort(key=lambda row: (
            language_order.get(row.get("instruction", {}).get("language"), 9),
            _sort_id(row["episode_id"]),
        ))
        episode = candidates[0]
        instruction = episode.get("instruction", {})
        instruction_digest = hashlib.sha256(
            str(instruction.get("instruction_text", "")).encode()
        ).hexdigest()
        row = {
            "dataset": dataset,
            "split": "train",
            "scene_id": scene,
            "trajectory_id": trajectory,
            "episode_id": str(episode["episode_id"]),
            "reference_points": len(episode["reference_path"]),
            "language": instruction.get("language"),
            "instruction_sha256": instruction_digest,
        }
        row["selection_digest"] = stable_hash(row)
        selected.append(row)
    selected.sort(key=lambda row: (
        row["dataset"], row["scene_id"], _sort_id(row["trajectory_id"])
    ))
    selected_keys = {(row["scene_id"], row["trajectory_id"]) for row in selected}
    expected = eligible_before_exclusion - excluded
    if selected_keys != expected or len(selected_keys) != len(selected):
        raise RuntimeError(f"{dataset} complete unused population closure failed")
    return selected, {
        "eligible_trajectories_before_historical_exclusion": len(eligible_before_exclusion),
        "historically_excluded_trajectories": len(eligible_before_exclusion & excluded),
        "selected_unused_trajectories": len(selected),
        "selected_scenes": len({row["scene_id"] for row in selected}),
        "short_route_trajectories": len(short_routes),
        "non_english_trajectory_candidates": len(non_english_routes) if dataset == "RxR" else 0,
    }


def build_selection() -> dict:
    dsr = json.loads(DSR_PROTOCOL.read_text())
    if dsr.get("status") != "SEALED_BEFORE_MF3ZK_DSR_TRAINING":
        raise RuntimeError("DSR source protocol status drift")
    allowed_scenes = set(dsr["selection"]["outer_scene_assignment"])
    consumed = set(dsr["known_consumed_scene_ids"])
    if len(allowed_scenes) != 39 or allowed_scenes & consumed:
        raise RuntimeError("MF3ZL development/consumed scene boundary drift")
    r2r_episodes = _load_episodes(R2R_DATA)
    rxr_episodes = _load_episodes(RXR_DATA)
    used = _used_trajectories(r2r_episodes, rxr_episodes)
    routes = []
    counts = {}
    for dataset, episodes in (("R2R", r2r_episodes), ("RxR", rxr_episodes)):
        selected, detail = _representatives(
            dataset, episodes, allowed_scenes, used[dataset]
        )
        routes.extend(selected)
        counts[dataset] = detail
    routes.sort(key=lambda row: (
        row["dataset"], row["scene_id"], _sort_id(row["trajectory_id"])
    ))
    if any(row["split"] != "train" for row in routes):
        raise RuntimeError("public split entered exact-replay selection")
    identities = [
        (row["dataset"], row["scene_id"], row["trajectory_id"]) for row in routes
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate route in exact-replay selection")
    return {
        "schema_version": f"{SCHEMA}-selection/1",
        "status": "SEALED_COMPLETE_UNUSED_TRAIN_POPULATION",
        "revision": "mf3zl_rcsp_v1",
        "selection_rule": (
            "all unused deterministic trajectory representatives in the 39 "
            "already consumed development scenes; reference path >=4; RxR English only"
        ),
        "outcome_fields_used_for_selection": [],
        "adaptive_stopping_allowed": False,
        "untouched_scenes_consumed": False,
        "public_split_access": False,
        "allowed_scene_ids": sorted(allowed_scenes),
        "excluded_consumed_scene_ids": sorted(consumed),
        "counts": {
            "total_routes": len(routes),
            "datasets": counts,
        },
        "routes": routes,
        "route_identity_commitment_sha256": stable_hash(identities),
        "sources": {
            "dsr_protocol": inventory(DSR_PROTOCOL),
            "r2r_train": inventory(R2R_DATA),
            "rxr_train_guide": inventory(RXR_DATA),
            "r2r_old_selection": inventory(R2R_OLD_SELECTION),
            "rxr_old_selections": [inventory(path) for path in RXR_OLD_SELECTIONS],
        },
    }


def _implementation_paths() -> tuple[Path, ...]:
    return (
        ROOT / "METHOD_REVISION_3ZL_RCSP.md",
        ROOT / "revealnav_mf3/rcsp.py",
        ROOT / "revealnav_mf3/rcsp_selection.py",
        ROOT / "revealnav_mf3/exact_replay.py",
        ROOT / "scripts/collect_mf3zl_exact_replay.py",
        ROOT / "scripts/mf3zl_exact_replay_worker.py",
        ROOT / "scripts/train_mf3zl_rcsp.py",
        ROOT / "tests/test_mf3zl_rcsp_model.py",
        ROOT / "tests/test_mf3zl_rcsp_selection.py",
        ROOT / "tests/test_mf3zl_exact_replay.py",
    )


def build_protocol(selection: dict) -> dict:
    dsr = json.loads(DSR_PROTOCOL.read_text())
    sources = {
        "mf3zg_gate": ROOT / (
            "artifacts/training/mf3zg_hierarchical_core_preserving_gate_v1/"
            "MF3ZG_SHADOW_GATE.json"
        ),
        "r2r_checkpoint": ROOT / (
            "third_party/ETP-R1/data/logs/checkpoints/"
            "release_r2r_grpo/store/ckpt.iter270.pth"
        ),
        "rxr_checkpoint": ROOT / (
            "third_party/ETP-R1/data/logs/checkpoints/"
            "release_rxr_grpo/store/ckpt.iter1320.pth"
        ),
        "joint_pretrained": ROOT / (
            "third_party/ETP-R1/pretrained/r2r_rxr_ce/"
            "mlm.sap_habitat_depth/store2/model_step_367500.pt"
        ),
    }
    return {
        "schema_version": "revealnav-mf3zl-rcsp-protocol/1",
        "status": "SEALED_BEFORE_MF3ZL_COLLECTION_AND_TRAINING",
        "revision": "mf3zl_rcsp_v1",
        "method": "Risk-Constrained Counterfactual Switch Policy",
        "historical_evidence": {
            "mf3zk_dsr_v1_consumed": True,
            "old_confirmation_consumed": True,
            "old_confirmation_reused": False,
        },
        "frozen_components": [
            "ETP-R1 policy and visual-language backbone",
            "MF3V proposal ranker",
            "MF3ZG core/expansion hierarchy",
            "native and runner-up action identities",
            "one-switch intervention budget",
            "utility and catastrophe definitions",
        ],
        "selection": {
            "path": str(SELECTION.relative_to(ROOT)),
            "bytes": SELECTION.stat().st_size,
            "sha256": sha256_file(SELECTION),
            "route_identity_commitment_sha256": selection[
                "route_identity_commitment_sha256"
            ],
            "complete_population_required": True,
            "adaptive_stopping": False,
        },
        "data_gate": {
            "minimum_unique_exact_events_per_domain": 300,
            "minimum_development_scenes_per_domain": 30,
            "maximum_conflicting_identities": 0,
            "untouched_scene_expansion_on_failure": False,
        },
        "model": {
            "policy_dim": 10,
            "embedding_dim": 768,
            "rank": 4,
            "dataset_scene_tier_inputs": False,
            "decision_rule": "switch_logit > 0",
            "decision_threshold": 0.0,
        },
        "loss": {
            "preference": "abs_delta_utility_weighted_BCEWithLogits",
            "catastrophic_threshold": -0.10,
            "risk_constraint": "per_domain_soft_selected_rate_not_above_ungated",
            "weighting": "domain_scene_episode_event",
        },
        "training": {
            "outer_folds": 5,
            "outer_scene_assignment": dsr["selection"]["outer_scene_assignment"],
            "inner_folds": 4,
            "inner_fold_salt": "mf3zl-rcsp-v1-inner-scenes/1",
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [20260830, 20260831, 20260832],
            "optimizer": "Adam_full_batch_projected_primal_dual",
            "learning_rate": 0.005,
            "dual_learning_rate": 0.05,
            "training_steps": 800,
            "only_selected_hyperparameter": "weight_decay",
            "selection_metric": "inner_oof_utility_weighted_preference_loss",
        },
        "utility": UTILITY_WEIGHTS,
        "failure_criteria": [
            "source_or_feature_provenance_drift",
            "incomplete_sealed_population",
            "data_support_below_300_events_or_30_scenes_per_domain",
            "duplicate_or_conflicting_exact_identity",
            "public_or_consumed_scene_access",
            "non_exact_pair_or_action_identity_failure",
            "joint_outer_fold_missing_or_zero_intervention",
            "domain_nonpositive_utility_or_leave_one_scene_utility",
            "domain_catastrophic_rate_above_ungated_or_matched_baseline",
            "utility_not_above_low_margin_and_high_score_matched_baselines",
        ],
        "public_split_access": {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        },
        "authorization": {
            "trainer_may_authorize_confirmation": False,
            "trainer_may_authorize_public_unseen": False,
            "deployment_integration_before_pass": False,
        },
        "source_files": {name: inventory(path) for name, path in sources.items()},
        "implementation_files": {
            str(path.relative_to(ROOT)): inventory(path)
            for path in _implementation_paths()
        },
    }


def seal() -> int:
    if PROTOCOL.exists() or SELECTION.exists():
        raise RuntimeError("MF3ZL protocol/selection already exists; refusing reseal")
    OUT.mkdir(parents=True, exist_ok=True)
    selection = build_selection()
    atomic_json(SELECTION, selection)
    protocol = build_protocol(selection)
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({
        "status": protocol["status"],
        "routes": selection["counts"],
        "selection_sha256": sha256_file(SELECTION),
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def verify_protocol() -> tuple[dict, dict]:
    if not PROTOCOL.is_file() or PROTOCOL.is_symlink():
        raise RuntimeError("MF3ZL pre-sealed protocol is unavailable")
    protocol = json.loads(PROTOCOL.read_text())
    selection = json.loads(SELECTION.read_text())
    if (
        protocol.get("status") != "SEALED_BEFORE_MF3ZL_COLLECTION_AND_TRAINING"
        or selection.get("status") != "SEALED_COMPLETE_UNUSED_TRAIN_POPULATION"
        or protocol["selection"]["sha256"] != sha256_file(SELECTION)
        or protocol["selection"]["bytes"] != SELECTION.stat().st_size
        or protocol["public_split_access"] != {
            "test": False, "test_challenge": False,
            "val_seen": False, "val_unseen": False,
        }
    ):
        raise RuntimeError("MF3ZL sealed protocol semantics drift")
    for section in ("source_files", "implementation_files"):
        for item in protocol[section].values():
            path = ROOT / item["path"]
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"MF3ZL sealed file drift: {item['path']}")
    if any(row["split"] != "train" for row in selection["routes"]):
        raise RuntimeError("MF3ZL selection contains a public split")
    return protocol, selection


def _attempts(job_root: Path) -> list[Path]:
    return sorted(path for path in job_root.glob("attempt_*") if path.is_dir())


def _summary(path: Path) -> dict | None:
    summary = path / "RUN_SUMMARY.json"
    try:
        return json.loads(summary.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _job_state(job_root: Path) -> tuple[str, Path | None, dict | None]:
    attempts = _attempts(job_root)
    if not attempts:
        return "missing", None, None
    latest = attempts[-1]
    value = _summary(latest)
    if value is not None and value.get("status") == "PASS":
        return "pass", latest, value
    return "fail", latest, value


def _next_attempt(job_root: Path) -> Path:
    number = len(_attempts(job_root)) + 1
    return job_root / f"attempt_{number:03d}"


def _handle_signal(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _run_jobs(
    stage: str,
    jobs: list[dict],
    progress_path: Path,
    gpus: tuple[int, ...],
    workers_per_gpu: int,
    retry_failed: bool,
) -> int:
    if not gpus or workers_per_gpu < 1:
        raise ValueError("MF3ZL runner needs at least one GPU slot")
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    completed = []
    failed_existing = []
    queue = []
    for job in jobs:
        state, attempt, summary = _job_state(job["job_root"])
        if state == "pass":
            completed.append(job["job_id"])
        elif state == "fail" and not retry_failed:
            failed_existing.append(job["job_id"])
        else:
            queue.append(job)
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    active = []
    stage_started = time.time()
    logs = OUT / "logs" / stage
    logs.mkdir(parents=True, exist_ok=True)

    def write_progress(status: str) -> None:
        elapsed = max(0.001, time.time() - stage_started)
        newly_done = len(completed) + len(failed_existing)
        rate = newly_done / elapsed
        remaining = len(queue) + len(active)
        atomic_json(progress_path, {
            "schema_version": f"{SCHEMA}-{stage}-progress/1",
            "status": status,
            "stage": stage,
            "total": len(jobs),
            "completed_pass": len(completed),
            "failed": len(failed_existing),
            "queued": len(queue),
            "active": [
                {"job_id": item["job"]["job_id"], "gpu": item["gpu"]}
                for item in active
            ],
            "elapsed_s": round(elapsed, 1),
            "completion_rate_per_hour": round(rate * 3600, 2),
            "eta_s": round(remaining / rate, 1) if rate > 0 else None,
            "retry_failed": retry_failed,
            "public_split_access": False,
        })

    while queue or active:
        while queue and slots and not STOP_REQUESTED:
            job = queue.pop(0)
            gpu = slots.pop(0)
            run_dir = _next_attempt(job["job_root"])
            stdout = (logs / f"{job['job_id']}_a{run_dir.name[-3:]}.stdout").open("w")
            stderr = (logs / f"{job['job_id']}_a{run_dir.name[-3:]}.stderr").open("w")
            command = [
                str(PYTHON), str(WORKER),
                "--dataset", job["dataset"],
                "--episode-id", job["episode_id"],
                "--scene-id", job["scene_id"],
                "--mode", job["mode"],
                "--run-dir", str(run_dir),
            ]
            if job.get("target") is not None:
                command.extend(["--target", str(job["target"])])
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment,
                stdout=stdout, stderr=stderr,
            )
            active.append({
                "process": process, "job": job, "gpu": gpu,
                "streams": (stdout, stderr), "run_dir": run_dir,
            })
        write_progress("STOPPING" if STOP_REQUESTED else "RUNNING")
        if STOP_REQUESTED and active:
            for item in active:
                item["process"].terminate()
        if not active:
            break
        time.sleep(2)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            job_id = item["job"]["job_id"]
            value = _summary(item["run_dir"])
            if code == 0 and value is not None and value.get("status") == "PASS":
                completed.append(job_id)
            else:
                failed_existing.append(job_id)
            active.remove(item)
            slots.append(item["gpu"])
            slots.sort()
    status = (
        "STOPPED" if STOP_REQUESTED else
        "COMPLETE" if not failed_existing and len(completed) == len(jobs)
        else "COMPLETE_WITH_FAILURES"
    )
    write_progress(status)
    return 0 if status == "COMPLETE" else 2


def _native_jobs(selection: dict) -> list[dict]:
    return [{
        "job_id": f"{row['dataset'].lower()}_ep_{row['episode_id']}",
        "dataset": row["dataset"],
        "episode_id": row["episode_id"],
        "scene_id": row["scene_id"],
        "mode": "native_shadow",
        "job_root": OUT / "runs/native" / row["dataset"].lower() / f"ep_{row['episode_id']}",
    } for row in selection["routes"]]


def run_native(args) -> int:
    _, selection = verify_protocol()
    return _run_jobs(
        "native_shadow", _native_jobs(selection), NATIVE_PROGRESS,
        tuple(args.gpus), args.workers_per_gpu, args.retry_failed,
    )


def _passed_attempt(job_root: Path) -> tuple[Path, dict]:
    state, attempt, summary = _job_state(job_root)
    if state != "pass" or attempt is None or summary is None:
        raise RuntimeError(f"required MF3ZL rollout is not PASS: {job_root}")
    return attempt, summary


def seal_targets(selection: dict) -> dict:
    if TARGETS.exists():
        value = json.loads(TARGETS.read_text())
        if value.get("source_selection_sha256") != sha256_file(SELECTION):
            raise RuntimeError("MF3ZL target seal source drift")
        return value
    targets = []
    for job, route in zip(_native_jobs(selection), selection["routes"], strict=True):
        attempt, summary = _passed_attempt(job["job_root"])
        if (
            summary.get("mode") != "native_shadow"
            or summary.get("dataset") != route["dataset"]
            or summary.get("episode_id") != route["episode_id"]
            or summary.get("scene_id") != route["scene_id"]
            or summary.get("task_metric_payload_read") is not False
            or summary.get("public_split_access") is not False
        ):
            raise RuntimeError("MF3ZL native-shadow boundary drift")
        for event in summary["proposal_events"]:
            identity = ProposalEventIdentity(**event["event_identity"])
            item = {
                "event_identity": event["event_identity"],
                "decision": event["decision"],
                "native_feature": event["feature"],
                "native_run_summary": inventory(attempt / "RUN_SUMMARY.json"),
                "selection_digest": route["selection_digest"],
            }
            item["target_digest"] = stable_hash(item["event_identity"])
            targets.append(item)
    identities = [item["event_identity"] for item in targets]
    if len({stable_hash(value) for value in identities}) != len(identities):
        raise RuntimeError("duplicate MF3ZL target identity")
    value = {
        "schema_version": f"{SCHEMA}-targets/1",
        "status": "SEALED_AFTER_COMPLETE_NATIVE_SHADOW_BEFORE_TREATMENTS",
        "source_selection_sha256": sha256_file(SELECTION),
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "task_metric_payload_read": False,
        "outcome_fields_used_for_target_selection": [],
        "all_observed_first_core_and_first_expansion_events": True,
        "counts": {
            "events": len(targets),
            "datasets": dict(Counter(
                item["event_identity"]["dataset"] for item in targets
            )),
            "tiers": dict(Counter(
                item["event_identity"]["tier"] for item in targets
            )),
        },
        "targets": targets,
    }
    atomic_json(TARGETS, value)
    item_root = OUT / "target_items"
    item_root.mkdir()
    for item in targets:
        atomic_json(item_root / f"{item['target_digest']}.json", {
            "schema_version": f"{SCHEMA}-target-item/1",
            "event_identity": item["event_identity"],
            "target_digest": item["target_digest"],
            "source_targets_sha256": sha256_file(TARGETS),
        })
    return value


def _target_jobs(targets: dict) -> list[dict]:
    jobs = []
    for item in targets["targets"]:
        identity = item["event_identity"]
        digest = item["target_digest"]
        jobs.append({
            "job_id": f"target_{digest[:16]}",
            "dataset": identity["dataset"],
            "episode_id": identity["episode_id"],
            "scene_id": identity["scene_id"],
            "mode": "targeted_switch",
            "target": OUT / "target_items" / f"{digest}.json",
            "job_root": OUT / "runs/targeted" / identity["dataset"].lower() / digest,
        })
    return jobs


def run_targets(args) -> int:
    _, selection = verify_protocol()
    native = json.loads(NATIVE_PROGRESS.read_text())
    if native.get("status") != "COMPLETE":
        raise RuntimeError("all sealed native shadows must pass before targets are sealed")
    targets = seal_targets(selection)
    return _run_jobs(
        "targeted_switch", _target_jobs(targets), TARGET_PROGRESS,
        tuple(args.gpus), args.workers_per_gpu, args.retry_failed,
    )


def _read_trace(artifact: dict) -> list[dict]:
    path = ROOT / artifact["path"]
    if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
        raise RuntimeError("MF3ZL trace provenance drift")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _metrics(summary: dict, episode_id: str) -> tuple[dict, dict]:
    items = summary.get("stats_inventory", [])
    if len(items) != 1 or items[0].get("payload_read_by_worker") is not False:
        raise RuntimeError("MF3ZL task metric inventory drift")
    item = items[0]
    path = ROOT / item["path"]
    if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
        raise RuntimeError("MF3ZL task metric provenance drift")
    value = json.loads(path.read_text()).get(str(episode_id))
    if not isinstance(value, dict):
        raise RuntimeError("MF3ZL episode task metrics missing")
    for key in ("success", "spl", "ndtw", "sdtw"):
        if not math.isfinite(float(value[key])):
            raise RuntimeError("MF3ZL task metric is non-finite")
    return value, item


def _utility(metrics: dict) -> float:
    return sum(float(metrics[key]) * weight for key, weight in UTILITY_WEIGHTS.items())


def assemble() -> int:
    verify_protocol()
    if not TARGETS.is_file() or json.loads(TARGET_PROGRESS.read_text()).get("status") != "COMPLETE":
        raise RuntimeError("all MF3ZL targeted treatments must pass before assembly")
    targets = json.loads(TARGETS.read_text())
    native_jobs = {
        (job["dataset"], job["episode_id"]): job
        for job in _native_jobs(json.loads(SELECTION.read_text()))
    }
    records = []
    seen = {}
    for item, job in zip(targets["targets"], _target_jobs(targets), strict=True):
        identity = ProposalEventIdentity(**item["event_identity"])
        native_attempt, native_summary = _passed_attempt(
            native_jobs[(identity.dataset, identity.episode_id)]["job_root"]
        )
        treatment_attempt, treatment_summary = _passed_attempt(job["job_root"])
        if (
            treatment_summary.get("mode") != "targeted_switch"
            or treatment_summary.get("target") != item["event_identity"]
            or treatment_summary.get("changed_actions") != 1
            or treatment_summary.get("task_metric_payload_read") is not False
            or treatment_summary.get("public_split_access") is not False
        ):
            raise RuntimeError("MF3ZL targeted treatment boundary drift")
        native_controller = _read_trace(native_summary["controller_trace"])
        treatment_controller = _read_trace(treatment_summary["controller_trace"])
        native_event_records = [
            row for row in native_controller
            if row.get("event_identity") == item["event_identity"]
        ]
        if len(native_event_records) != 1:
            raise RuntimeError("MF3ZL native event trace cardinality drift")
        validate_shadow_event(native_event_records[0])
        validate_forced_switch(treatment_controller, identity)
        native_actions = _read_trace(native_summary["base_trace"])
        treatment_actions = _read_trace(treatment_summary["base_trace"])
        validate_exact_prefix(native_actions, treatment_actions, identity.step)
        treatment_events = [
            event for event in treatment_summary["proposal_events"]
            if event["event_identity"] == item["event_identity"]
        ]
        if len(treatment_events) != 1:
            raise RuntimeError("MF3ZL treatment event feature cardinality drift")
        native_feature = item["native_feature"]
        treatment_feature = treatment_events[0]["feature"]
        for feature in (native_feature, treatment_feature):
            path = ROOT / feature["path"]
            if path.stat().st_size != feature["bytes"] or sha256_file(path) != feature["sha256"]:
                raise RuntimeError("MF3ZL event feature provenance drift")
        if (
            native_feature["bytes"] != treatment_feature["bytes"]
            or native_feature["sha256"] != treatment_feature["sha256"]
        ):
            raise RuntimeError("MF3ZL causal target feature changed on exact replay")
        baseline, baseline_stats = _metrics(native_summary, identity.episode_id)
        treatment, treatment_stats = _metrics(treatment_summary, identity.episode_id)
        delta = {
            key: float(treatment[key]) - float(baseline[key])
            for key in ("success", "spl", "ndtw", "sdtw")
        }
        delta["utility"] = _utility(treatment) - _utility(baseline)
        record = {
            "dataset": identity.dataset,
            "episode_id": identity.episode_id,
            "scene_id": identity.scene_id,
            "decision_step": identity.step,
            "tier": identity.tier,
            "native_action_id": identity.native_action_id,
            "runner_action_id": identity.runner_action_id,
            "target": delta["utility"],
            "catastrophic": delta["utility"] <= -0.10,
            "delta": delta,
            "decision": item["decision"],
            "baseline_metrics": baseline,
            "treatment_metrics": treatment,
            "feature": native_feature,
            "baseline_stats": baseline_stats,
            "treatment_stats": treatment_stats,
            "native_run_summary": inventory(native_attempt / "RUN_SUMMARY.json"),
            "treatment_run_summary": inventory(treatment_attempt / "RUN_SUMMARY.json"),
            "exact_prefix_verified": True,
            "exact_one_switch_verified": True,
        }
        key = stable_hash(item["event_identity"])
        content = stable_hash(record)
        if key in seen and seen[key] != content:
            raise RuntimeError("conflicting MF3ZL exact event identity")
        seen[key] = content
        records.append(record)
    value = {
        "schema_version": f"{SCHEMA}-manifest/1",
        "status": "DENSE_EXACT_REPLAY_READY",
        "revision": "mf3zl_rcsp_v1",
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "source_selection_sha256": sha256_file(SELECTION),
        "source_targets_sha256": sha256_file(TARGETS),
        "complete_population_executed": True,
        "task_metric_payload_read_only_during_assembly": True,
        "public_split_access": False,
        "counts": {
            "exact_events": len(records),
            "datasets": dict(Counter(row["dataset"] for row in records)),
            "scenes": {
                dataset: len({row["scene_id"] for row in records if row["dataset"] == dataset})
                for dataset in ("RxR", "R2R")
            },
            "positive": dict(Counter(
                row["dataset"] for row in records if row["target"] > 0
            )),
            "catastrophic": dict(Counter(
                row["dataset"] for row in records if row["catastrophic"]
            )),
            "conflicting_identities": 0,
        },
        "records": records,
    }
    if MANIFEST.exists():
        raise RuntimeError("refusing to overwrite MF3ZL exact replay manifest")
    atomic_json(MANIFEST, value)
    print(json.dumps(value["counts"], indent=2, sort_keys=True))
    return 0


def audit() -> int:
    protocol, _ = verify_protocol()
    manifest = json.loads(MANIFEST.read_text())
    dsr = json.loads(DSR_PROTOCOL.read_text())
    if (
        manifest.get("status") != "DENSE_EXACT_REPLAY_READY"
        or manifest.get("complete_population_executed") is not True
        or manifest.get("public_split_access") is not False
        or manifest.get("source_protocol_sha256") != sha256_file(PROTOCOL)
    ):
        raise RuntimeError("MF3ZL dense manifest boundary drift")
    existing = dsr["source_inventory"]["canonical_rows"]
    dense = manifest["records"]
    failures = []
    domains = {}
    for domain in ("RxR", "R2R"):
        old_rows = [
            row for row in existing if row["identity"]["dataset"] == domain
        ]
        new_rows = [row for row in dense if row["dataset"] == domain]
        old_episodes = {row["identity"]["episode_id"] for row in old_rows}
        new_episodes = {row["episode_id"] for row in new_rows}
        if old_episodes & new_episodes:
            failures.append(f"{domain}:historical_dense_episode_overlap")
        scenes = {row["scene_id"] for row in old_rows} | {
            row["scene_id"] for row in new_rows
        }
        events = len(old_rows) + len(new_rows)
        domains[domain] = {
            "existing_exact_events": len(old_rows),
            "dense_exact_events": len(new_rows),
            "combined_unique_exact_events": events,
            "combined_development_scenes": len(scenes),
        }
        if events < int(protocol["data_gate"]["minimum_unique_exact_events_per_domain"]):
            failures.append(f"{domain}:fewer_than_300_exact_events")
        if len(scenes) < int(protocol["data_gate"]["minimum_development_scenes_per_domain"]):
            failures.append(f"{domain}:fewer_than_30_development_scenes")
    if int(manifest["counts"]["conflicting_identities"]) != 0:
        failures.append("conflicting_exact_identity")
    value = {
        "schema_version": "revealnav-mf3zl-data-support-audit/1",
        "status": "TRAIN_DATA_SUPPORT_PASS" if not failures else "TRAIN_DATA_SUPPORT_FAIL",
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "source_manifest": inventory(MANIFEST),
        "complete_population_executed": True,
        "adaptive_stopping_used": False,
        "untouched_scenes_consumed": False,
        "public_split_access": False,
        "domains": domains,
        "failure_reasons": failures,
        "rcsp_training_authorized": not failures,
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
    }
    if AUDIT.exists():
        raise RuntimeError("refusing to overwrite MF3ZL data support audit")
    atomic_json(AUDIT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if not failures else 2


def monitor() -> int:
    values = {}
    for name, path in (
        ("native_shadow", NATIVE_PROGRESS),
        ("targeted_switch", TARGET_PROGRESS),
        ("targets", TARGETS),
        ("data_audit", AUDIT),
    ):
        if path.is_file():
            value = json.loads(path.read_text())
            values[name] = value.get("counts", value) if name == "targets" else value
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seal")
    for name in ("run-native-shadow", "run-targeted-switches"):
        current = subparsers.add_parser(name)
        current.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
        current.add_argument("--workers-per-gpu", type=int, default=1)
        current.add_argument("--retry-failed", action="store_true")
    subparsers.add_parser("assemble")
    subparsers.add_parser("audit")
    subparsers.add_parser("monitor")
    args = parser.parse_args()
    return {
        "seal": lambda: seal(),
        "run-native-shadow": lambda: run_native(args),
        "run-targeted-switches": lambda: run_targets(args),
        "assemble": lambda: assemble(),
        "audit": lambda: audit(),
        "monitor": lambda: monitor(),
    }[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
