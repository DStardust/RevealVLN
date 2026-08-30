#!/usr/bin/env python3
"""Replay only V6.3.1 causal features and strictly join V6.2 labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rxr_v6_counterfactual_worker as worker_base  # noqa: E402
from revealnav_mf2r6.protocol import scene_fold  # noqa: E402
from run_rxr_v6_counterfactual_pipeline import (  # noqa: E402
    atomic_json, atomic_npz, sha256_file,
)


PYTHON = ROOT / ".envs/etpr1/bin/python"
SOURCE = ROOT / "artifacts/phase1/rxr_v6/full_v6_2"
OUTPUT = ROOT / "artifacts/phase1/rxr_v6/full_v6_3_1"
POST_Q = ROOT / "artifacts/phase1/rxr_v6/v6_3_1/post_q_outer"
SELECTION = SOURCE / "RXR_V6_EPISODE_SELECTION.json"
SOURCE_PROTOCOL = SOURCE / "RXR_V6_PAIR_PROTOCOL.json"
SOURCE_MANIFEST = SOURCE / "RXR_V6_PAIRED_DATASET_MANIFEST.json"
SOURCE_ARRAYS = SOURCE / "RXR_V6_PAIRED_DATASET.npz"
DESIGN = ROOT / (
    "artifacts/design/MF2_POLICY_RELATIVE_REVERSIBLE_ADVANTAGE_V6_3_1.md"
)
WORKER = ROOT / "scripts/rxr_v6_3_1_feature_worker.py"
TRAINER = ROOT / "scripts/train_rxr_v6_relative_advantage.py"
MODEL = ROOT / "revealnav_mf2r6/model.py"
PARTITION = ROOT / "revealnav_mf2r6/protocol.py"
PROTOCOL = OUTPUT / "RXR_V6_3_1_FEATURE_PROTOCOL.json"
PROGRESS = OUTPUT / "RXR_V6_3_1_FEATURE_PROGRESS.json"
MANIFEST = OUTPUT / "RXR_V6_3_1_PAIRED_DATASET_MANIFEST.json"
ARRAYS = OUTPUT / "RXR_V6_3_1_PAIRED_DATASET.npz"
BASE_KEYS = (
    "instruction", "post_observation", "temporal_history", "checkpoint",
    "native", "alternative", "scalars",
)
EMBEDDING_KEYS = BASE_KEYS[:-1]
IDENTITY_KEYS = (
    "event_id", "event_index", "episode_id", "trajectory_id", "scene_id",
    "language", "controller_seed", "post_navigation_step",
    "prefix_action_count", "checkpoint_id", "native_branch_id",
    "alternative_branch_id", "alternative_source", "candidate_branch_ids",
    "online_return_path_length_m", "causal_prefix_only",
)


def safe_project_file(relative: str, evidence: dict | None = None) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError("unsafe V6.3.1 project-relative path")
    raw = ROOT / value
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError("missing or linked V6.3.1 evidence")
    path = raw.resolve()
    if ROOT not in path.parents:
        raise RuntimeError("V6.3.1 evidence escaped project")
    if evidence is not None and (
        path.stat().st_size != evidence["bytes"]
        or sha256_file(path) != evidence["sha256"]
    ):
        raise RuntimeError("V6.3.1 file provenance drift")
    return path


def post_q_evidence() -> dict[str, dict]:
    rows = {}
    for fold in range(5):
        result_path = POST_Q / f"fold_{fold}/RESULT.json"
        result = json.loads(result_path.read_text())
        checkpoint = safe_project_file(
            result["checkpoint"]["path"], result["checkpoint"]
        )
        if not (
            result.get("status") == "V6_3_1_POST_Q_OUTER_READY"
            and result.get("outer_fold") == fold
            and result.get("calibration_or_evaluation_scene_leakage") is False
        ):
            raise RuntimeError("post-Q outer result is not authorized")
        rows[str(fold)] = {
            "outer_fold": fold,
            "result_path": str(result_path.relative_to(ROOT)),
            "result_bytes": result_path.stat().st_size,
            "result_sha256": sha256_file(result_path),
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            **{
                key: result[key] for key in result
                if key.endswith("_v6_scene_ids_sha256")
            },
        }
    return rows


def protocol_value() -> dict:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_protocol = json.loads(SOURCE_PROTOCOL.read_text())
    selection = json.loads(SELECTION.read_text())
    if not (
        source_manifest.get("status") == "RXR_V6_PAIRED_DATASET_READY"
        and source_manifest.get("metadata", {}).get("pairs") == 339
        and source_protocol.get("maximum_events_per_episode") == 3
        and selection.get("episode_count") == 120
        and source_manifest.get("metadata", {}).get("unseen_or_test_read") is False
    ):
        raise RuntimeError("V6.2 parent evidence drift")
    sources = (
        SELECTION, SOURCE_PROTOCOL, SOURCE_MANIFEST, SOURCE_ARRAYS, DESIGN,
        WORKER, TRAINER, MODEL, PARTITION, Path(__file__).resolve(),
    )
    return {
        "schema_version": "revealnav-rxr-v6.3.1-feature-protocol/1",
        "status": "SEALED_BEFORE_V6_3_1_FEATURE_REPLAY",
        "parent_cohort": "full_v6_2",
        "episodes": 120,
        "attempted_events": 340,
        "accepted_pairs": 339,
        "rejected_unexecutable_pairs": 1,
        "maximum_events_per_episode": 3,
        "replay": "native shadow only; no macro or label regeneration",
        "outer_post_q": post_q_evidence(),
        "sources": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
            for path in sources
        },
        "unseen_or_test_read": False,
        "paper_result": False,
    }


def seal() -> dict:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V6.3.1 feature protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    return value


def feature_arrays(event: dict) -> dict[str, np.ndarray]:
    path = safe_project_file(event["feature_path"], {
        "bytes": event["feature_bytes"], "sha256": event["feature_sha256"],
    })
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def source_run(row: dict) -> Path:
    return SOURCE / "runs" / f"shadow_ep{row['episode_id']}_s{row['seed']}"


def replay_run(row: dict) -> Path:
    return OUTPUT / "runs" / f"shadow_ep{row['episode_id']}_s{row['seed']}"


def verify_episode(row: dict, protocol: dict) -> list[dict]:
    old_dir = source_run(row)
    new_dir = replay_run(row)
    old = json.loads((old_dir / "RUN_SUMMARY.json").read_text())
    new = json.loads((new_dir / "RUN_SUMMARY.json").read_text())
    for key in (
        "status", "episode_id", "trajectory_id", "scene_id", "language",
        "seed", "mode", "split", "candidate_event_count", "metrics",
    ):
        if old.get(key) != new.get(key):
            raise RuntimeError(f"V6.3.1 replay episode drift: {key}")
    if not (
        new.get("status") == "PASS" and new.get("mode") == "shadow"
        and new.get("split") == "train"
        and new.get("unseen_or_test_read") is False
        and new.get("future_information_used_for_online_input") is False
        and (old_dir / "base_trace.jsonl").read_bytes()
        == (new_dir / "base_trace.jsonl").read_bytes()
        and old.get("base_trace_sha256") == new.get("base_trace_sha256")
    ):
        raise RuntimeError("V6.3.1 native trajectory replay drift")
    old_events = old["candidate_events"]
    new_events = new["candidate_events"]
    if len(old_events) != len(new_events):
        raise RuntimeError("V6.3.1 candidate count drift")
    for old_event, new_event in zip(old_events, new_events):
        if any(old_event.get(key) != new_event.get(key) for key in IDENTITY_KEYS):
            raise RuntimeError("V6.3.1 candidate identity drift")
        if not (
            new_event.get("runtime_scene_fold")
            == scene_fold(str(row["scene_id"]))
            and new_event.get("post_q_outer_evidence")
            == protocol["outer_post_q"]
        ):
            raise RuntimeError("V6.3.1 post-Q fold provenance drift")
        old_arrays = feature_arrays(old_event)
        new_arrays = feature_arrays(new_event)
        if set(old_arrays) != set(BASE_KEYS) or set(new_arrays) != (
            set(BASE_KEYS) | {"outer_fold_scalars"}
        ):
            raise RuntimeError("V6.3.1 feature schema drift")
        if worker_base.stable_array_hash(old_arrays) != old_event[
            "causal_state_sha256"
        ]:
            raise RuntimeError("V6.2 source feature hash drift")
        base_new = {key: new_arrays[key] for key in BASE_KEYS}
        if not (
            worker_base.stable_array_hash(new_arrays)
            == new_event["causal_state_sha256"]
            and worker_base.stable_array_hash(base_new)
            == new_event["base_causal_state_sha256"]
            == old_event["causal_state_sha256"]
            and all(np.array_equal(old_arrays[key], new_arrays[key])
                    for key in BASE_KEYS)
            and new_arrays["outer_fold_scalars"].shape == (5, 20)
            and new_arrays["outer_fold_scalars"].dtype == np.float32
            and np.isfinite(new_arrays["outer_fold_scalars"]).all()
            and all(np.array_equal(
                new_arrays["outer_fold_scalars"][fold, :3],
                old_arrays["scalars"],
            ) for fold in range(5))
        ):
            raise RuntimeError("V6.3.1 causal feature replay drift")
    return new_events[:3]


def atomic_progress(stage: str, completed: int, active: dict, failures: list) -> None:
    atomic_json(PROGRESS, {
        "schema_version": "revealnav-rxr-v6.3.1-feature-progress/1",
        "stage": stage, "selected": 120, "completed": completed,
        "remaining": 120 - completed, "active": active,
        "failures": failures,
    })


def execute(row: dict, gpu: int) -> int:
    run_dir = replay_run(row)
    command = [
        str(PYTHON), str(WORKER), "--episode-id", str(row["episode_id"]),
        "--seed", str(row["seed"]), "--mode", "shadow",
        "--run-dir", str(run_dir),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE="1")
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"shadow_ep{row['episode_id']}.out").open("w") as stdout, (
        logs / f"shadow_ep{row['episode_id']}.err"
    ).open("w") as stderr:
        return subprocess.run(
            command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
            text=True, check=False,
        ).returncode


def replay(gpus: tuple[int, ...]) -> None:
    protocol = seal()
    selection = json.loads(SELECTION.read_text())["episodes"]
    pending = []
    completed = 0
    for row in selection:
        run_dir = replay_run(row)
        if run_dir.exists():
            verify_episode(row, protocol)
            completed += 1
        else:
            pending.append(row)
    active = {}
    failures = []
    atomic_progress("feature_replay", completed, active, failures)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        running = {}
        rows = iter(pending)
        for slot, gpu in enumerate(gpus):
            row = next(rows, None)
            if row is None:
                break
            future = pool.submit(execute, row, gpu)
            running[future] = (slot, gpu, row)
            active[str(slot)] = {"gpu": gpu, "episode_id": row["episode_id"]}
        atomic_progress("feature_replay", completed, active, failures)
        while running:
            done, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                slot, gpu, row = running.pop(future)
                returncode = future.result()
                try:
                    if returncode:
                        raise RuntimeError(f"worker exit {returncode}")
                    verify_episode(row, protocol)
                except Exception as error:
                    failures.append({
                        "episode_id": row["episode_id"],
                        "error": f"{type(error).__name__}: {error}",
                    })
                completed += 1
                active.pop(str(slot), None)
                next_row = next(rows, None)
                if next_row is not None:
                    future = pool.submit(execute, next_row, gpu)
                    running[future] = (slot, gpu, next_row)
                    active[str(slot)] = {
                        "gpu": gpu, "episode_id": next_row["episode_id"],
                    }
                atomic_progress("feature_replay", completed, active, failures)
    if failures:
        raise RuntimeError(f"V6.3.1 feature replay failures: {len(failures)}")


def assemble() -> dict:
    protocol = seal()
    selection = json.loads(SELECTION.read_text())["episodes"]
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    if not (
        SOURCE_ARRAYS.stat().st_size == source_manifest["arrays"]["bytes"]
        and sha256_file(SOURCE_ARRAYS) == source_manifest["arrays"]["sha256"]
    ):
        raise RuntimeError("V6.2 consolidated arrays drift")
    with np.load(SOURCE_ARRAYS, allow_pickle=False) as source:
        parent_arrays = {key: source[key].copy() for key in source.files}
    if set(parent_arrays) != set(BASE_KEYS) | {"target"}:
        raise RuntimeError("V6.2 consolidated schema drift")
    old_events = {}
    new_events = {}
    for episode in selection:
        old_summary = json.loads(
            (source_run(episode) / "RUN_SUMMARY.json").read_text()
        )
        replay_events = verify_episode(episode, protocol)
        for old_event, new_event in zip(
            old_summary["candidate_events"][:3], replay_events
        ):
            if old_event["event_id"] in old_events:
                raise RuntimeError("duplicate parent V6 event")
            old_events[old_event["event_id"]] = old_event
            new_events[new_event["event_id"]] = new_event
    record_ids = {row["event_id"] for row in source_manifest["records"]}
    rejection_ids = {row["event_id"] for row in source_manifest["rejections"]}
    if not (
        len(old_events) == 340 and set(old_events) == set(new_events)
        and not record_ids & rejection_ids
        and record_ids | rejection_ids == set(old_events)
        and len(record_ids) == 339 and len(rejection_ids) == 1
    ):
        raise RuntimeError("V6.3.1 attempted-event accounting drift")
    scalar_rows = []
    failure_rows = []
    records = []
    for index, source_record in enumerate(source_manifest["records"]):
        if int(source_record["row_index"]) != index:
            raise RuntimeError("V6.2 parent row order drift")
        old_event = old_events[source_record["event_id"]]
        new_event = new_events[source_record["event_id"]]
        old_feature = feature_arrays(old_event)
        new_feature = feature_arrays(new_event)
        for key in EMBEDDING_KEYS:
            if not (
                np.array_equal(parent_arrays[key][index], old_feature[key])
                and np.array_equal(parent_arrays[key][index], new_feature[key])
            ):
                raise RuntimeError("V6.3.1 consolidated embedding drift")
        if not (
            np.array_equal(parent_arrays["scalars"][index], old_feature["scalars"])
            and parent_arrays["target"][index]
            == np.float32(source_record["relative_advantage"])
        ):
            raise RuntimeError("V6.3.1 parent scalar/target drift")
        scalar_rows.append(new_feature["outer_fold_scalars"])
        failure_rows.append(float(source_record["native_metrics"]["success"] <= 0.0))
        record = copy.deepcopy(source_record)
        record.update({
            "v6_3_1_feature_path": new_event["feature_path"],
            "v6_3_1_feature_bytes": new_event["feature_bytes"],
            "v6_3_1_feature_sha256": new_event["feature_sha256"],
            "v6_3_1_base_causal_state_sha256": new_event[
                "base_causal_state_sha256"
            ],
            "v6_3_1_outer_post_q": protocol["outer_post_q"],
            "native_failure_auxiliary_target": failure_rows[-1],
        })
        records.append(record)
    output_arrays = {
        **{key: parent_arrays[key] for key in EMBEDDING_KEYS},
        "scalars": np.asarray(scalar_rows, dtype=np.float32),
        "target": parent_arrays["target"],
        "native_failure": np.asarray(failure_rows, dtype=np.float32),
    }
    if ARRAYS.exists() or MANIFEST.exists():
        raise RuntimeError("refusing to overwrite V6.3.1 assembled dataset")
    atomic_npz(ARRAYS, output_arrays)
    value = {
        "schema_version": "revealnav-rxr-v6.3.1-paired-dataset/1",
        "status": "RXR_V6_3_1_PAIRED_DATASET_READY",
        "cohort": "full_v6_3_1",
        "records": records,
        "rejections": copy.deepcopy(source_manifest["rejections"]),
        "metadata": {
            **source_manifest["metadata"],
            "feature_replayed_episodes": 120,
            "feature_replayed_events": 340,
            "labels_regenerated": False,
            "macros_rerun": False,
            "outer_fold_feature_shape": [5, 20],
            "native_failure_rows": int(sum(failure_rows)),
            "base_trajectory_exact": True,
            "development_only_due_to_prior_method_inspection": True,
        },
        "arrays": {
            "path": str(ARRAYS.relative_to(ROOT)),
            "bytes": ARRAYS.stat().st_size,
            "sha256": sha256_file(ARRAYS),
        },
        "parent": {
            "manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "arrays_path": str(SOURCE_ARRAYS.relative_to(ROOT)),
            "arrays_sha256": sha256_file(SOURCE_ARRAYS),
        },
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": sha256_file(PROTOCOL),
        },
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    atomic_json(MANIFEST, value)
    atomic_progress("assembled", 120, {}, [])
    return value


def parse_gpus(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item not in range(8) for item in result):
        raise ValueError("invalid GPU list")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "replay", "assemble", "all"))
    parser.add_argument("--gpus", default="0,0,0,0,1,1,1,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.command in ("seal", "all"):
        seal()
    if args.command in ("replay", "all"):
        replay(parse_gpus(args.gpus))
    if args.command in ("assemble", "all"):
        value = assemble()
        print(json.dumps(value["metadata"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
