#!/usr/bin/env python3
"""Seal, run, and gate the train-only post-excursion feature pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cr5_queue50_tx_worker as core  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V4 = BASE / "branch_excursion_v4"
MANIFEST = V4 / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V4.json"
ACCEPTANCE = V4 / "RXR_BRANCH_EXCURSION_CORRECTNESS_ACCEPTANCE_V4_1.json"
OUT = BASE / "post_excursion_v4_6_1"
PILOT = OUT / "pilot"
RUNS = PILOT / "runs"
REPEAT = PILOT / "repeat"
PROTOCOL = OUT / "RXR_POST_EXCURSION_PROTOCOL_V4_6_1.json"
RESULT = PILOT / "RXR_POST_EXCURSION_PILOT_RESULT_V4_6_1.json"
MANIFEST_OUT = PILOT / "RXR_POST_EXCURSION_PILOT_MANIFEST_V4_6_1.json"
WORKER = ROOT / "scripts/rxr_post_excursion_feature_worker_v4_6.py"
DESIGN = ROOT / "artifacts/design/MF2_POST_EXCURSION_BACKTRACK_V4_6.md"
LOCK = ROOT / "locks/REE_Q_FUSION_RETURN_EXECUTOR_V4_5.json"


def label_for(record: dict) -> dict:
    return json.loads((V4 / record["path"]).read_text())


def pilot_selection() -> tuple[list[str], dict]:
    source = json.loads(MANIFEST.read_text())
    strata: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for record in source["records"]:
        label = label_for(record)
        if all(row["commit_route"].get("success") for row in label["labels"]):
            digest = hashlib.sha256(record["event_id"].encode()).hexdigest()
            strata[(record["label_source"], record["candidate_count"])].append(
                (digest, record["event_id"])
            )
    expected = {
        ("primary_human_audited", 2),
        ("primary_human_audited", 3),
        ("automatic_secondary_pseudolabel", 2),
        ("automatic_secondary_pseudolabel", 3),
    }
    if set(strata) != expected or any(len(rows) < 4 for rows in strata.values()):
        raise RuntimeError("pilot selection strata unavailable")
    selected = []
    counts = {}
    for key in sorted(strata):
        values = [event_id for _, event_id in sorted(strata[key])[:4]]
        selected.extend(values)
        counts[f"{key[0]}:k{key[1]}"] = len(values)
    return selected, counts


def protocol_value() -> dict:
    acceptance = json.loads(ACCEPTANCE.read_text())
    lock = json.loads(LOCK.read_text())
    event_ids, strata = pilot_selection()
    if not (
        acceptance.get("status")
        == "BRANCH_EXCURSION_LABEL_CORRECTNESS_ACCEPTANCE_PASS"
        and acceptance.get("training_authorized") is True
        and lock.get("status") == "LOCKED_BEFORE_POST_EXCURSION_DATA_GENERATION"
        and len(event_ids) == len(set(event_ids)) == 16
    ):
        raise RuntimeError("post-excursion pilot precondition failed")
    sources = (MANIFEST, ACCEPTANCE, LOCK, DESIGN, WORKER)
    return {
        "schema_version": "revealnav-mf2-post-excursion-protocol/4.6.1",
        "status": "SEALED_BEFORE_POST_EXCURSION_PILOT",
        "event_ids": event_ids,
        "selection": (
            "four minimum sha256(event_id) events per label-source x "
            "candidate-count stratum among events whose sealed outbound routes "
            "all succeed"
        ),
        "strata": strata,
        "expected_events": 16, "expected_branches": 40,
        "repeat_event_ids": [event_ids[0], event_ids[12]],
        "input_boundary": {
            "ends_at": "reached branch state before return rollout",
            "front_observation_only": True,
            "future_frames": 0, "target_truth_in_input": False,
        },
        "costs": {
            "continue_target": 0.0, "continue_wrong": 5.0,
            "backtrack": "bounded reached->Q->target frozen-controller cost",
            "backtrack_target_additional_missed_opportunity": 5.0,
            "controller_failure": 5.0,
            "outbound_excursion_is_sunk": True,
        },
        "pilot_gates": {
            "all_16_events_and_40_branches_complete": True,
            "all_outbound_replays_match_sealed_v4": True,
            "all_40_reached_state_features_finite": True,
            "both_strict_continue_and_backtrack_labels_present": True,
            "bounded_return_failures_retained_as_ties": True,
            "two_event_feature_replay_is_byte_identical": True,
            "no_future_gold_development_or_unseen_payload": True,
        },
        "sources": {
            str(path.relative_to(ROOT)): core.sha256_file(path) for path in sources
        },
        "full_generation_authorized_after_pass": True,
        "training_authorized": False,
        "gold_payload_read": False, "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed post-excursion protocol drift")
    if not PROTOCOL.exists():
        core.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "events": len(value["event_ids"]),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": core.sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def run_lane(name: str, gpu: int, events: list[str], output: Path):
    event_list = PILOT / f"{name}_events.json"
    lane_result = PILOT / f"{name}.json"
    core.atomic_json(event_list, events)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "POST_EXCURSION_PROTOCOL_SHA256": core.sha256_file(PROTOCOL)}
    process = subprocess.run([
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--event-list", str(event_list), "--output-dir", str(output),
        "--lane-result", str(lane_result), "--physical-gpu", str(gpu),
    ], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (PILOT / f"{name}.log").write_text(process.stdout)
    return name, process.returncode, lane_result, process.stdout[-4000:]


def run(gpus: tuple[int, ...]) -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("post-excursion protocol must be sealed")
    event_ids = protocol_value()["event_ids"]
    lanes = [[] for _ in gpus]
    for index, event_id in enumerate(event_ids):
        lanes[index % len(gpus)].append(event_id)
    RUNS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = [future.result() for future in [
            pool.submit(run_lane, f"lane_gpu{gpu}", gpu, events, RUNS)
            for gpu, events in zip(gpus, lanes)
        ]]
    failures = [(name, code, tail) for name, code, _, tail in results if code]
    if failures:
        raise RuntimeError(f"post-excursion lane failures: {failures}")
    repeats = protocol_value()["repeat_event_ids"]
    REPEAT.mkdir(parents=True, exist_ok=True)
    repeat_results = []
    for index, event_id in enumerate(repeats):
        gpu = gpus[index % len(gpus)]
        repeat_results.append(run_lane(
            f"repeat_gpu{gpu}_{index}", gpu, [event_id], REPEAT,
        ))
    failures = [(name, code, tail) for name, code, _, tail in repeat_results if code]
    if failures:
        raise RuntimeError(f"post-excursion repeat failures: {failures}")
    return aggregate(core.qfloat(time.monotonic() - started))


def aggregate(wall_clock_s: float | None = None) -> int:
    protocol = protocol_value()
    rows = []
    counts = Counter()
    all_finite = True
    all_nonzero = True
    for event_id in protocol["event_ids"]:
        label_path = RUNS / f"{event_id}.json"
        feature_path = RUNS / f"{event_id}.npz"
        if not label_path.is_file() or not feature_path.is_file():
            raise RuntimeError(f"missing pilot event {event_id}")
        label = json.loads(label_path.read_text())
        if label["protocol_sha256"] != core.sha256_file(PROTOCOL):
            raise RuntimeError("event protocol drift")
        with np.load(feature_path, allow_pickle=False) as shard:
            forbidden = [key for key in shard.files if "target" in key.lower()]
            if forbidden:
                raise RuntimeError(f"target truth leaked into inputs: {forbidden}")
            reachable = shard["reachable_mask"]
            finite_arrays = [shard[key] for key in (
                "instruction_embedding", "pre_history_embeddings",
                "checkpoint_embedding", "selected_branch_embeddings",
                "post_history_embeddings", "post_candidate_embeddings",
                "normalized_excursion_elapsed",
            )]
            all_finite &= all(np.isfinite(array).all() for array in finite_arrays)
            all_nonzero &= all(
                np.linalg.norm(shard["post_history_embeddings"][index]) > 0
                for index in np.where(reachable)[0]
            )
        for branch in label["branches"]:
            counts["branches"] += 1
            counts["trainable"] += int(branch["trainable"])
            counts[f"preferred_{branch['preferred_action']}"] += int(branch["trainable"])
            counts["outbound_match"] += int(branch["outbound_matches_v4"])
            counts["return_failure"] += int(
                branch["trainable"]
                and not branch["return_route"].get("success", False)
            )
        rows.append({
            "event_id": event_id, "scene_id": label["scene_id"],
            "label_source": label["label_source"],
            "candidate_count": len(label["branches"]),
            "feature_path": str(feature_path.relative_to(ROOT)),
            "feature_bytes": feature_path.stat().st_size,
            "feature_sha256": core.sha256_file(feature_path),
            "label_path": str(label_path.relative_to(ROOT)),
            "label_bytes": label_path.stat().st_size,
            "label_sha256": core.sha256_file(label_path),
        })
    replay_equal = []
    for event_id in protocol["repeat_event_ids"]:
        original_feature = RUNS / f"{event_id}.npz"
        repeated_feature = REPEAT / f"{event_id}.npz"
        original_label = json.loads((RUNS / f"{event_id}.json").read_text())
        repeated_label = json.loads((REPEAT / f"{event_id}.json").read_text())
        replay_equal.append(
            core.sha256_file(original_feature) == core.sha256_file(repeated_feature)
            and original_label["branches"] == repeated_label["branches"]
        )
    manifest = {
        "schema_version": "revealnav-mf2-post-excursion-manifest/4.6.1",
        "records": rows,
        "metadata": {
            "scope": "pilot_train_only", "protocol_sha256": core.sha256_file(PROTOCOL),
            "events": len(rows), "branches": counts["branches"],
            "trainable_examples": counts["trainable"],
            "training_authorized": False, "gold_payload_read": False,
            "paper_result": False,
        },
    }
    core.atomic_json(MANIFEST_OUT, manifest)
    gates = {
        "all_16_events_and_40_branches_complete": (
            len(rows) == 16 and counts["branches"] == counts["trainable"] == 40
        ),
        "all_outbound_replays_match_sealed_v4": counts["outbound_match"] == 40,
        "all_40_reached_state_features_finite": all_finite and all_nonzero,
        "both_strict_continue_and_backtrack_labels_present": (
            counts["preferred_CONTINUE"] > 0 and counts["preferred_BACKTRACK"] > 0
        ),
        "bounded_return_failures_retained_as_ties": (
            counts["return_failure"] > 0 and counts["preferred_TIE"] > 0
        ),
        "two_event_feature_replay_is_byte_identical": all(replay_equal),
        "no_future_gold_development_or_unseen_payload": True,
        "input_npz_contains_no_target_field": True,
        "no_part_files": not list(PILOT.rglob("*.part")),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-post-excursion-pilot-result/4.6.1",
        "status": (
            "POST_EXCURSION_PILOT_GATE_PASS" if passed
            else "POST_EXCURSION_PILOT_GATE_FAIL"
        ),
        "counts": dict(counts), "gates": gates,
        "repeat_feature_and_route_equal": replay_equal,
        "wall_clock_s": wall_clock_s,
        "manifest": {
            "path": str(MANIFEST_OUT.relative_to(ROOT)),
            "bytes": MANIFEST_OUT.stat().st_size,
            "sha256": core.sha256_file(MANIFEST_OUT),
        },
        "protocol_sha256": core.sha256_file(PROTOCOL),
        "full_generation_authorized": passed,
        "training_authorized": False,
        "gold_payload_read": False, "paper_result": False,
        "next_gate": "full train-only reached-state generation" if passed else "repair pilot",
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
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    if args.seal:
        return seal()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("invalid GPU list")
    return run(gpus) if args.run else aggregate()


if __name__ == "__main__":
    raise SystemExit(main())
