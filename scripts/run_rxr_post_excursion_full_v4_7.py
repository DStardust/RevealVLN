#!/usr/bin/env python3
"""Generate and gate all train-only reached-branch BACKTRACK examples."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cr5_queue50_tx_worker as core  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V4 = BASE / "branch_excursion_v4"
SOURCE = V4 / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V4.json"
OUT = BASE / "post_excursion_v4_7"
RUNS = OUT / "runs"
PROTOCOL = OUT / "RXR_POST_EXCURSION_FULL_PROTOCOL_V4_7.json"
PROGRESS = OUT / "RXR_POST_EXCURSION_FULL_PROGRESS_V4_7.json"
MANIFEST = OUT / "RXR_POST_EXCURSION_FULL_MANIFEST_V4_7.json"
RESULT = OUT / "RXR_POST_EXCURSION_FULL_RESULT_V4_7.json"
WORKER = ROOT / "scripts/rxr_post_excursion_feature_worker_v4_7.py"
BASE_WORKER = ROOT / "scripts/rxr_post_excursion_feature_worker_v4_6.py"
PILOT_LOCK = ROOT / "locks/POST_EXCURSION_PILOT_V4_6_1.json"
PILOT_RESULT = BASE / (
    "post_excursion_v4_6_1/pilot/RXR_POST_EXCURSION_PILOT_RESULT_V4_6_1.json"
)
DESIGN = ROOT / "artifacts/design/MF2_POST_EXCURSION_BACKTRACK_V4_6.md"


def partition(scene_id: str) -> str:
    return (
        "development" if int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 6 == 1
        else "train"
    )


def source_profile() -> tuple[list[str], dict]:
    manifest = json.loads(SOURCE.read_text())
    counts = Counter()
    event_ids = []
    for record in manifest["records"]:
        event_ids.append(record["event_id"])
        counts["events"] += 1
        counts[f"source:{record['label_source']}"] += 1
        split = partition(record["scene_id"])
        counts[f"{split}_events"] += 1
        label = json.loads((V4 / record["path"]).read_text())
        for branch in label["labels"]:
            counts["branches"] += 1
            counts[f"{split}_branches"] += 1
            reached = bool(branch["commit_route"].get("success"))
            counts["expected_reachable"] += int(reached)
            counts["expected_unreachable"] += int(not reached)
            counts[f"{split}_reachable"] += int(reached)
    expected = {
        "events": 424, "branches": 880,
        "expected_reachable": 655, "expected_unreachable": 225,
        "source:primary_human_audited": 280,
        "source:automatic_secondary_pseudolabel": 144,
        "train_events": 341, "development_events": 83,
        "train_branches": 709, "development_branches": 171,
        "train_reachable": 537, "development_reachable": 118,
    }
    if dict(counts) != expected or len(event_ids) != len(set(event_ids)):
        raise RuntimeError(f"full source profile drift: {dict(counts)}")
    return event_ids, expected


def protocol_value() -> dict:
    pilot = json.loads(PILOT_RESULT.read_text())
    lock = json.loads(PILOT_LOCK.read_text())
    event_ids, counts = source_profile()
    if not (
        pilot.get("status") == "POST_EXCURSION_PILOT_GATE_PASS"
        and pilot.get("full_generation_authorized") is True
        and lock.get("status") == "LOCKED_BEFORE_FULL_TRAIN_ONLY_GENERATION"
    ):
        raise RuntimeError("full post-excursion precondition failed")
    sources = (SOURCE, PILOT_LOCK, PILOT_RESULT, DESIGN, BASE_WORKER, WORKER)
    return {
        "schema_version": "revealnav-mf2-post-excursion-full-protocol/4.7",
        "status": "SEALED_BEFORE_FULL_TRAIN_ONLY_POST_EXCURSION_GENERATION",
        "event_ids": event_ids,
        "expected_counts": counts,
        "partition": "sha256(scene_id) mod 6 == 1 internal development; otherwise train",
        "generation": (
            "accepted V4.6.1 causal reached-state feature and bounded-cost contract; "
            "failed outbound routes retained without synthetic features"
        ),
        "success_gates": {
            "all_424_events_and_880_branches_accounted": True,
            "all_655_reached_inputs_finite_nonzero": True,
            "all_225_outbound_failures_retained_nontrainable": True,
            "both_action_preferences_and_controller_failure_ties_present": True,
            "scene_disjoint_internal_partitions_match_frozen_counts": True,
            "no_target_truth_field_in_input_npz": True,
            "no_future_gold_or_evaluation_payload": True,
        },
        "sources": {
            str(path.relative_to(ROOT)): core.sha256_file(path) for path in sources
        },
        "training_authorized": False,
        "gold_payload_read": False, "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed full protocol drift")
    if not PROTOCOL.exists():
        core.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "events": len(value["event_ids"]),
        "expected_counts": value["expected_counts"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": core.sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def valid_existing(event_id: str, protocol_sha: str) -> bool:
    label_path = RUNS / f"{event_id}.json"
    feature_path = RUNS / f"{event_id}.npz"
    try:
        label = json.loads(label_path.read_text())
        declared = label["feature"]
        unsigned = dict(label)
        expected_event_sha = unsigned.pop("event_sha256")
        return (
            label["status"] == "POST_EXCURSION_EVENT_COMPLETE"
            and label["event_id"] == event_id
            and label["protocol_sha256"] == protocol_sha
            and core.stable_sha(unsigned) == expected_event_sha
            and Path(declared["path"]).name == feature_path.name
            and feature_path.is_file() and not feature_path.is_symlink()
            and feature_path.stat().st_size == declared["bytes"]
            and core.sha256_file(feature_path) == declared["sha256"]
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def run_lane(name: str, gpu: int, event_ids: list[str], protocol_sha: str):
    event_list = OUT / f"{name}_events.json"
    lane_result = OUT / f"{name}.json"
    core.atomic_json(event_list, event_ids)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "POST_EXCURSION_PROTOCOL_SHA256": protocol_sha}
    process = subprocess.run([
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--event-list", str(event_list), "--output-dir", str(RUNS),
        "--lane-result", str(lane_result), "--physical-gpu", str(gpu),
    ], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (OUT / f"{name}.log").write_text(process.stdout)
    return name, process.returncode, process.stdout[-4000:]


def write_progress(total: int, complete: int, started: float, status: str) -> None:
    elapsed = time.monotonic() - started
    rate = complete / elapsed if elapsed else 0.0
    core.atomic_json(PROGRESS, {
        "schema_version": "revealnav-mf2-post-excursion-progress/4.7",
        "status": status, "total": total, "completed": complete,
        "remaining": total - complete,
        "elapsed_s": core.qfloat(elapsed), "events_per_s": core.qfloat(rate),
        "eta_s": (
            core.qfloat((total - complete) / rate) if rate and complete < total else 0.0
        ),
    })


def run(gpus: tuple[int, ...]) -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("full protocol must be sealed")
    protocol_sha = core.sha256_file(PROTOCOL)
    event_ids = protocol_value()["event_ids"]
    RUNS.mkdir(parents=True, exist_ok=True)
    existing = [event_id for event_id in event_ids if valid_existing(event_id, protocol_sha)]
    missing = [event_id for event_id in event_ids if event_id not in set(existing)]
    started = time.monotonic()
    write_progress(len(event_ids), len(existing), started, "RUNNING")
    lanes = [[] for _ in gpus]
    for index, event_id in enumerate(missing):
        lanes[index % len(gpus)].append(event_id)
    jobs = [(f"full_gpu{gpu}", gpu, events) for gpu, events in zip(gpus, lanes) if events]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs) or 1) as pool:
        results = [future.result() for future in [
            pool.submit(run_lane, name, gpu, events, protocol_sha)
            for name, gpu, events in jobs
        ]]
    failures = [(name, code, tail) for name, code, tail in results if code]
    complete = sum(valid_existing(event_id, protocol_sha) for event_id in event_ids)
    write_progress(
        len(event_ids), complete, started,
        "FAILED_RESUMABLE" if failures else "COMPLETE",
    )
    if failures:
        raise RuntimeError(f"full lane failures (outputs remain resumable): {failures}")
    return aggregate(core.qfloat(time.monotonic() - started))


def aggregate(wall_clock_s: float | None = None) -> int:
    protocol = protocol_value()
    protocol_sha = core.sha256_file(PROTOCOL)
    records = []
    counts = Counter()
    all_finite = True
    all_nonzero = True
    for source_record in json.loads(SOURCE.read_text())["records"]:
        event_id = source_record["event_id"]
        if not valid_existing(event_id, protocol_sha):
            raise RuntimeError(f"missing or invalid full event: {event_id}")
        label_path = RUNS / f"{event_id}.json"
        feature_path = RUNS / f"{event_id}.npz"
        label = json.loads(label_path.read_text())
        split = partition(label["scene_id"])
        with np.load(feature_path, allow_pickle=False) as shard:
            if any("target" in key.lower() for key in shard.files):
                raise RuntimeError("target truth field present in model input")
            reachable = shard["reachable_mask"]
            arrays = [shard[key] for key in (
                "instruction_embedding", "pre_history_embeddings",
                "checkpoint_embedding", "selected_branch_embeddings",
                "post_history_embeddings", "post_candidate_embeddings",
                "normalized_excursion_elapsed",
            )]
            all_finite &= all(np.isfinite(array).all() for array in arrays)
            all_nonzero &= all(
                np.linalg.norm(shard["post_history_embeddings"][index]) > 0
                for index in np.where(reachable)[0]
            )
        counts["events"] += 1
        counts[f"{split}_events"] += 1
        counts[f"source:{label['label_source']}"] += 1
        for branch in label["branches"]:
            counts["branches"] += 1
            counts[f"{split}_branches"] += 1
            counts["trainable"] += int(branch["trainable"])
            counts["unreachable"] += int(not branch["trainable"])
            if branch["trainable"]:
                counts[f"{split}_trainable"] += 1
                counts[f"preferred_{branch['preferred_action']}"] += 1
                counts["return_failure"] += int(
                    not branch["return_route"].get("success", False)
                )
            counts["outbound_match"] += int(branch["outbound_matches_v4"])
        records.append({
            "event_id": event_id, "scene_id": label["scene_id"],
            "split": split, "label_source": label["label_source"],
            "candidate_count": len(label["branches"]),
            "trainable_examples": sum(row["trainable"] for row in label["branches"]),
            "feature_path": str(feature_path.relative_to(ROOT)),
            "feature_bytes": feature_path.stat().st_size,
            "feature_sha256": core.sha256_file(feature_path),
            "label_path": str(label_path.relative_to(ROOT)),
            "label_bytes": label_path.stat().st_size,
            "label_sha256": core.sha256_file(label_path),
        })
    manifest = {
        "schema_version": "revealnav-mf2-post-excursion-full-manifest/4.7",
        "records": records,
        "metadata": {
            "protocol_sha256": protocol_sha,
            "events": counts["events"], "branches": counts["branches"],
            "trainable_examples": counts["trainable"],
            "train_examples": counts["train_trainable"],
            "internal_development_examples": counts["development_trainable"],
            "causal_input": True, "future_frames_used": 0,
            "training_authorized": True,
            "gold_payload_read": False, "paper_result": False,
        },
    }
    core.atomic_json(MANIFEST, manifest)
    expected = protocol["expected_counts"]
    gates = {
        "all_424_events_and_880_branches_accounted": (
            counts["events"] == expected["events"]
            and counts["branches"] == expected["branches"]
            and counts["outbound_match"] == expected["branches"]
        ),
        "all_655_reached_inputs_finite_nonzero": (
            counts["trainable"] == expected["expected_reachable"]
            and all_finite and all_nonzero
        ),
        "all_225_outbound_failures_retained_nontrainable": (
            counts["unreachable"] == expected["expected_unreachable"]
        ),
        "both_action_preferences_and_controller_failure_ties_present": (
            counts["preferred_CONTINUE"] > 0
            and counts["preferred_BACKTRACK"] > 0
            and counts["preferred_TIE"] > 0
            and counts["return_failure"] > 0
        ),
        "scene_disjoint_internal_partitions_match_frozen_counts": (
            counts["train_events"] == expected["train_events"]
            and counts["development_events"] == expected["development_events"]
            and counts["train_trainable"] == expected["train_reachable"]
            and counts["development_trainable"] == expected["development_reachable"]
        ),
        "both_label_sources_complete": (
            counts["source:primary_human_audited"]
            == expected["source:primary_human_audited"]
            and counts["source:automatic_secondary_pseudolabel"]
            == expected["source:automatic_secondary_pseudolabel"]
        ),
        "no_target_truth_field_in_input_npz": True,
        "no_future_gold_or_evaluation_payload": True,
        "no_part_files": not list(OUT.rglob("*.part")),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-post-excursion-full-result/4.7",
        "status": (
            "POST_EXCURSION_FULL_GATE_PASS" if passed
            else "POST_EXCURSION_FULL_GATE_FAIL"
        ),
        "counts": dict(counts), "gates": gates,
        "wall_clock_s": wall_clock_s,
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "bytes": MANIFEST.stat().st_size,
            "sha256": core.sha256_file(MANIFEST),
        },
        "protocol_sha256": protocol_sha,
        "training_authorized": passed,
        "gold_payload_read": False, "paper_result": False,
        "next_gate": "three-seed BACKTRACK cost-head training" if passed else "repair full generation",
    }
    core.atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "counts": value["counts"],
        "gates": gates, "wall_clock_s": wall_clock_s,
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.seal:
        return seal()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("invalid GPU list")
    return run(gpus) if args.run else aggregate()


if __name__ == "__main__":
    raise SystemExit(main())
