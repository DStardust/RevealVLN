#!/usr/bin/env python3
"""Seal, execute, and aggregate train-only branch-excursion labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cr5_queue50_tx_worker as core  # noqa: E402
import rxr_branch_excursion_label_worker_v4 as worker  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
OUT = BASE / "branch_excursion_v4"
RUNS = OUT / "runs"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_LABEL_PROTOCOL_V4.json"
PROGRESS = OUT / "RXR_BRANCH_EXCURSION_LABEL_PROGRESS_V4.json"
MANIFEST = OUT / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V4.json"
GATE = OUT / "RXR_BRANCH_EXCURSION_LABEL_GATE_V4.json"
Q_MANIFEST = worker.Q_MANIFEST
ADJUDICATION = ROOT / (
    "artifacts/evaluation/mf2_branch_excursion_witness_v4_1/"
    "RXR_BRANCH_EXCURSION_ADJUDICATION_RESULT_V4_1.json"
)
REBUILD = ROOT / "artifacts/runtime/HABITAT017_GLIBC232_REBUILD.json"
WORKER = ROOT / "scripts/rxr_branch_excursion_label_worker_v4.py"


def source_files() -> tuple[Path, ...]:
    primary = BASE / "multibranch_v2"
    secondary = BASE / "secondary_expansion_v1"
    return (
        Q_MANIFEST,
        ADJUDICATION,
        REBUILD,
        WORKER,
        primary / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json",
        primary / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json",
        secondary / "multibranch/RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json",
        secondary / "multibranch/RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json",
    )


def protocol_value() -> dict:
    manifest = json.loads(Q_MANIFEST.read_text())
    adjudication = json.loads(ADJUDICATION.read_text())
    rebuild = json.loads(REBUILD.read_text())
    records = [row for row in manifest["records"] if row["split"] == "train"]
    if not (
        len(records) == 424
        and len({row["event_id"] for row in records}) == 424
        and adjudication.get("status")
        == "BRANCH_EXCURSION_LABEL_FEASIBILITY_ADJUDICATED_PASS"
        and rebuild.get("status") == "PASS"
    ):
        raise RuntimeError("branch-excursion label precondition failed")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-label-protocol/4",
        "status": "SEALED_BEFORE_TRAIN_BRANCH_EXCURSION_LABEL_GENERATION",
        "event_ids": [row["event_id"] for row in records],
        "events": len(records),
        "label_sources": dict(Counter(row["label_source"] for row in records)),
        "decision_prefix": "max(checkpoint Q, first prefix where all branches are K=3 established)",
        "actions": {
            "commit_branch": (
                "existing frozen direct-controller cost plus 5 for a wrong commitment"
            ),
            "checkpointed_excursion": (
                "selected branch -> checkpoint Q -> target branch using the frozen "
                "controller; target branch uses its direct route"
            ),
        },
        "costs": {
            "normalization": "target checkpoint-to-branch frozen action count",
            "bounded_failure_cost": 5.0,
            "wrong_commitment_cost": 5.0,
        },
        "frozen_controller_failures": (
            "retained as bounded failure-cost labels; never filtered"
        ),
        "success_gates": {
            "all_424_train_events_complete": True,
            "all_branch_labels_complete_and_finite": True,
            "all_target_direct_routes_succeed": True,
            "both_human_and_pseudolabel_sources_present": True,
            "no_development_or_gold_events": True,
        },
        "sources": {
            str(path.relative_to(ROOT)): core.sha256_file(path)
            for path in source_files()
        },
        "development_access_allowed": False,
        "gold_access_allowed": False,
        "training_authorized": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed branch-excursion label protocol drift")
    if not PROTOCOL.exists():
        core.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "events": value["events"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": core.sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def valid_existing(path: Path, event_id: str, protocol_sha: str) -> bool:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "BRANCH_EXCURSION_LABEL_COMPLETE"
        and value.get("event_id") == event_id
        and value.get("protocol_sha256") == protocol_sha
        and value.get("gold_payload_read") is False
    )


def run_one(event_id: str, gpu: int, protocol_sha: str) -> tuple[str, int, str]:
    output = RUNS / f"{event_id}.json"
    if valid_existing(output, event_id, protocol_sha):
        return event_id, 0, "existing"
    env = os.environ.copy()
    env["RXR_BRANCH_EXCURSION_PROTOCOL_SHA256"] = protocol_sha
    process = subprocess.run(
        [
            str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
            "--event-id", event_id, "--gpu", str(gpu),
            "--output", str(output),
        ],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return event_id, process.returncode, process.stdout[-4000:]


def write_progress(total: int, completed: int, failures: list[str], started: float) -> None:
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else None
    core.atomic_json(PROGRESS, {
        "schema_version": "revealnav-mf2-branch-excursion-progress/4",
        "status": "RUNNING" if completed < total else "COMPLETE",
        "total": total,
        "completed": completed,
        "failed": len(failures),
        "failed_event_ids": failures,
        "elapsed_s": core.qfloat(elapsed),
        "events_per_s": core.qfloat(rate),
        "eta_s": None if remaining is None else core.qfloat(remaining),
    })


def aggregate(protocol_sha: str) -> int:
    source = json.loads(Q_MANIFEST.read_text())
    records_by_id = {
        row["event_id"]: row for row in source["records"] if row["split"] == "train"
    }
    records = []
    label_count = 0
    finite = True
    target_success = True
    macro_failures = 0
    for event_id in protocol_value()["event_ids"]:
        path = RUNS / f"{event_id}.json"
        if not valid_existing(path, event_id, protocol_sha):
            raise RuntimeError(f"missing or invalid label: {event_id}")
        value = json.loads(path.read_text())
        rows = value["labels"]
        label_count += len(rows)
        for row in rows:
            finite &= all(
                isinstance(row[key], (int, float))
                and float(row[key]) == float(row[key])
                for key in (
                    "commit_cost", "checkpointed_excursion_cost",
                    "option_preservation_gain",
                )
            )
            if row["is_target"]:
                target_success &= bool(row["commit_route"].get("success"))
            macro_failures += int(
                not row["checkpointed_excursion_route"].get("success", False)
            )
        original = records_by_id[event_id]
        records.append({
            "event_id": event_id,
            "scene_id": value["scene_id"],
            "label_source": value["label_source"],
            "candidate_count": len(rows),
            "path": os.path.relpath(path, OUT),
            "bytes": path.stat().st_size,
            "sha256": core.sha256_file(path),
            "online_feature_path": value["online_feature"]["path"],
            "online_feature_sha256": value["online_feature"]["sha256"],
            "source_q_feature_sha256": original["sha256"],
        })
    manifest = {
        "schema_version": "revealnav-mf2-branch-excursion-label-manifest/4",
        "records": records,
        "metadata": {
            "protocol_sha256": protocol_sha,
            "events": len(records),
            "branch_labels": label_count,
            "bounded_macro_failures_retained": macro_failures,
            "gold_payload_read": False,
            "training_authorized": True,
            "paper_result": False,
        },
    }
    core.atomic_json(MANIFEST, manifest)
    counts = Counter(row["label_source"] for row in records)
    gates = {
        "all_424_train_events_complete": len(records) == 424,
        "event_ids_unique": len({row["event_id"] for row in records}) == 424,
        "all_branch_labels_complete_and_finite": finite and label_count == sum(
            row["candidate_count"] for row in records
        ),
        "all_target_direct_routes_succeed": target_success,
        "both_human_and_pseudolabel_sources_present": counts == {
            "primary_human_audited": 280,
            "automatic_secondary_pseudolabel": 144,
        },
        "frozen_failures_retained_as_bounded_labels": macro_failures >= 1,
        "no_development_or_gold_events": True,
        "no_part_files": not list(OUT.rglob("*.part")),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-label-gate/4",
        "status": (
            "BRANCH_EXCURSION_TRAIN_LABEL_GATE_PASS" if passed
            else "BRANCH_EXCURSION_TRAIN_LABEL_GATE_FAIL"
        ),
        "counts": {
            "events": len(records), "branch_labels": label_count,
            "bounded_macro_failures_retained": macro_failures,
            "label_sources": dict(counts),
        },
        "gates": gates,
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "bytes": MANIFEST.stat().st_size,
            "sha256": core.sha256_file(MANIFEST),
        },
        "protocol_sha256": protocol_sha,
        "development_payload_read": False,
        "gold_payload_read": False,
        "training_authorized": passed,
        "paper_result": False,
        "next_gate": "event-level action-cost head training" if passed else "repair labels",
    }
    core.atomic_json(GATE, value)
    print(json.dumps({"status": value["status"], "counts": value["counts"], "gates": gates}, indent=2))
    return 0 if passed else 1


def run(gpus: tuple[int, ...], jobs: int) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("branch-excursion label protocol must be sealed")
    if not gpus or jobs < 1:
        raise ValueError("at least one GPU and worker are required")
    RUNS.mkdir(parents=True, exist_ok=True)
    protocol_sha = core.sha256_file(PROTOCOL)
    event_ids = protocol_value()["event_ids"]
    started = time.monotonic()
    failures = []
    completed = 0
    write_progress(len(event_ids), completed, failures, started)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_one, event_id, gpus[index % len(gpus)], protocol_sha): event_id
            for index, event_id in enumerate(event_ids)
        }
        for future in concurrent.futures.as_completed(futures):
            event_id, returncode, output = future.result()
            completed += 1
            if returncode:
                failures.append(event_id)
                print(f"FAIL {event_id}\n{output}", flush=True)
            elif completed % 10 == 0 or completed == len(event_ids):
                print(f"PROGRESS {completed}/{len(event_ids)} failures={len(failures)}", flush=True)
            write_progress(len(event_ids), completed, failures, started)
    if failures:
        return 1
    return aggregate(protocol_sha)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.seal:
        return seal()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    return run(gpus, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
