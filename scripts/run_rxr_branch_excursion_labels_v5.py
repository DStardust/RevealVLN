#!/usr/bin/env python3
"""Seal, smoke-test, generate and aggregate expanded V5 Q supervision."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cr5_queue50_tx_worker as core  # noqa: E402
import rxr_branch_excursion_label_worker_v5 as worker  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SOURCE = BASE / "RXR_SCALE_AUTOMATIC_TRAINING_MANIFEST.json"
OUT = BASE / "branch_excursion_v5_1"
RUNS = OUT / "runs"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_LABEL_PROTOCOL_V5_1.json"
SMOKE = OUT / "RXR_BRANCH_EXCURSION_LABEL_SMOKE_V5_1.json"
PROGRESS = OUT / "RXR_BRANCH_EXCURSION_LABEL_PROGRESS_V5_1.json"
MANIFEST = OUT / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V5_1.json"
GATE = OUT / "RXR_BRANCH_EXCURSION_LABEL_GATE_V5_1.json"
V4_ROOT = BASE / "branch_excursion_v4"
V4_MANIFEST = V4_ROOT / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V4.json"
V4_ACCEPTANCE = V4_ROOT / "RXR_BRANCH_EXCURSION_CORRECTNESS_ACCEPTANCE_V4_1.json"
REBUILD = ROOT / "artifacts/runtime/HABITAT017_GLIBC232_REBUILD.json"
REVISION = ROOT / "artifacts/design/MF2_BRANCH_EXCURSION_Q_DATA_REVISION_V5.md"
WORKER = ROOT / "scripts/rxr_branch_excursion_label_worker_v5.py"
EXPECTED_SOURCES = {
    "primary_human_audited": 280,
    "automatic_secondary_pseudolabel": 144,
    "automatic_scale_pseudolabel": 314,
    "automatic_scale_v2_pseudolabel": 1092,
}


def train_records() -> list[dict]:
    value = json.loads(SOURCE.read_text())
    if value.get("metadata", {}).get("training_authorized") is not True:
        raise RuntimeError("expanded training manifest is not authorized")
    records = [row for row in value["records"] if row["split"] == "train"]
    counts = Counter(row["label_source"] for row in records)
    if (
        len(records) != 1830
        or len({row["event_id"] for row in records}) != 1830
        or counts != EXPECTED_SOURCES
    ):
        raise RuntimeError(f"expanded train population drift: {len(records)}, {counts}")
    return records


@lru_cache(maxsize=1)
def legacy_records() -> dict[str, dict]:
    acceptance = json.loads(V4_ACCEPTANCE.read_text())
    manifest = json.loads(V4_MANIFEST.read_text())
    if not (
        acceptance.get("status")
        == "BRANCH_EXCURSION_LABEL_CORRECTNESS_ACCEPTANCE_PASS"
        and acceptance.get("training_authorized") is True
        and acceptance.get("manifest_sha256") == core.sha256_file(V4_MANIFEST)
        and len(manifest["records"]) == 424
    ):
        raise RuntimeError("accepted V4 label reuse precondition failed")
    return {row["event_id"]: row for row in manifest["records"]}


def source_files() -> tuple[Path, ...]:
    files = [SOURCE, V4_MANIFEST, V4_ACCEPTANCE, REBUILD, REVISION, WORKER]
    for wave in ("scale_v1", "scale_v2"):
        multi = BASE / wave / "automatic/multibranch"
        files.extend((
            multi / "RXR_SCALE_FEATURE_GATE.json",
            multi / "RXR_SCALE_TX_GATE.json",
            multi / "RXR_SCALE_MULTIBRANCH_GEOMETRY.json",
            multi / "RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json",
        ))
    return tuple(files)


def protocol_value() -> dict:
    records = train_records()
    legacy = legacy_records()
    rebuild = json.loads(REBUILD.read_text())
    if rebuild.get("status") != "PASS":
        raise RuntimeError("Habitat runtime precondition failed")
    if not set(legacy).issubset({row["event_id"] for row in records}):
        raise RuntimeError("legacy V4 labels are outside expanded train population")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-label-protocol/5.1",
        "status": "SEALED_BEFORE_EXPANDED_TRAIN_LABEL_GENERATION",
        "event_ids": [row["event_id"] for row in records],
        "events": len(records),
        "label_sources": dict(Counter(row["label_source"] for row in records)),
        "reuse": {
            "accepted_v4_events": len(legacy),
            "new_scale_events": len(records) - len(legacy),
            "contract": "identical bounded branch-excursion target semantics",
        },
        "decision_prefix": (
            "max(checkpoint Q, first confirmation prefix for every persistent branch)"
        ),
        "actions": {
            "commit_branch": "frozen direct cost plus 5 for wrong commitment",
            "checkpointed_excursion": (
                "selected branch -> checkpoint Q -> target branch using frozen controller"
            ),
        },
        "costs": {
            "normalization": "target checkpoint-to-branch frozen action count",
            "bounded_failure_cost": 5.0,
            "wrong_commitment_cost": 5.0,
        },
        "frozen_controller_failures": "retained as bounded labels",
        "success_gates": {
            "all_1830_train_events_complete": True,
            "all_branch_labels_complete_and_finite": True,
            "failed_target_routes_have_exact_bounded_cost": True,
            "all_four_authorized_sources_present": True,
            "no_development_or_gold_events": True,
        },
        "sources": {
            str(path.relative_to(ROOT)): core.sha256_file(path)
            for path in source_files()
        },
        "development_access_allowed": False,
        "gold_access_allowed": False,
        "future_information_used_for_online_input": 0,
        "training_authorized": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.1 branch-excursion protocol drift")
    if not PROTOCOL.exists():
        core.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "events": value["events"],
        "reuse": value["reuse"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": core.sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def valid_new(path: Path, event_id: str, protocol_sha: str) -> bool:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("schema_version") == "revealnav-mf2-branch-excursion-label/5.1"
        and value.get("status") == "BRANCH_EXCURSION_LABEL_COMPLETE"
        and value.get("event_id") == event_id
        and value.get("protocol_sha256") == protocol_sha
        and value.get("network_attempts") == 0
        and value.get("gold_payload_read") is False
    )


def run_one(event_id: str, gpu: int, protocol_sha: str) -> tuple[str, int, str]:
    if event_id in legacy_records():
        return event_id, 0, "accepted_v4_reuse"
    output = RUNS / f"{event_id}.json"
    if valid_new(output, event_id, protocol_sha):
        return event_id, 0, "existing_v5"
    environment = os.environ.copy()
    environment["RXR_BRANCH_EXCURSION_V5_1_PROTOCOL_SHA256"] = protocol_sha
    process = subprocess.run(
        [
            str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
            "--event-id", event_id, "--gpu", str(gpu), "--output", str(output),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return event_id, process.returncode, process.stdout[-4000:]


def smoke(gpus: tuple[int, ...]) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("V5.1 protocol must be sealed before smoke")
    selected = []
    records = train_records()
    for source in ("automatic_scale_pseudolabel", "automatic_scale_v2_pseudolabel"):
        selected.append(next(row["event_id"] for row in records
                             if row["label_source"] == source))
    protocol_sha = core.sha256_file(PROTOCOL)
    outcomes = [
        run_one(event_id, gpus[index % len(gpus)], protocol_sha)
        for index, event_id in enumerate(selected)
    ]
    gates = {
        "one_scale_v1_event_complete": outcomes[0][1] == 0,
        "one_scale_v2_event_complete": outcomes[1][1] == 0,
        "outputs_validate": all(
            valid_new(RUNS / f"{event_id}.json", event_id, protocol_sha)
            for event_id in selected
        ),
    }
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-label-smoke/5.1",
        "status": "BRANCH_EXCURSION_V5_1_SMOKE_PASS" if all(gates.values())
                  else "BRANCH_EXCURSION_V5_1_SMOKE_FAIL",
        "event_ids": selected,
        "gates": gates,
        "outcomes": [
            {"event_id": event_id, "returncode": code, "tail": tail}
            for event_id, code, tail in outcomes
        ],
        "protocol_sha256": protocol_sha,
        "gold_payload_read": False,
        "paper_result": False,
    }
    core.atomic_json(SMOKE, value)
    print(json.dumps(value, indent=2))
    return 0 if all(gates.values()) else 1


def write_progress(total: int, completed: int, failures: list[str], started: float) -> None:
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    core.atomic_json(PROGRESS, {
        "schema_version": "revealnav-mf2-branch-excursion-progress/5.1",
        "status": "RUNNING" if completed < total else (
            "COMPLETE" if not failures else "FAILED"
        ),
        "total": total,
        "completed": completed,
        "failed": len(failures),
        "failed_event_ids": failures,
        "elapsed_s": core.qfloat(elapsed),
        "events_per_s": core.qfloat(rate),
        "eta_s": None if rate == 0 else core.qfloat((total - completed) / rate),
    })


def label_source(event_id: str, protocol_sha: str) -> tuple[Path, str]:
    legacy = legacy_records()
    if event_id in legacy:
        row = legacy[event_id]
        path = (V4_ROOT / row["path"]).resolve()
        if (
            V4_ROOT not in path.parents
            or path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or core.sha256_file(path) != row["sha256"]
        ):
            raise RuntimeError(f"legacy label provenance drift: {event_id}")
        return path, "accepted_v4_reuse"
    path = RUNS / f"{event_id}.json"
    if not valid_new(path, event_id, protocol_sha):
        raise RuntimeError(f"missing or invalid V5 label: {event_id}")
    return path, "new_v5_scale_label"


def aggregate(protocol_sha: str) -> int:
    records = train_records()
    manifest_rows = []
    labels_total = 0
    macro_failures = 0
    failed_target_exact = True
    all_finite = True
    reuse = Counter()
    for record in records:
        event_id = record["event_id"]
        label_path, reuse_kind = label_source(event_id, protocol_sha)
        label = json.loads(label_path.read_text())
        rows = sorted(label["labels"], key=lambda row: row["branch_index"])
        if not (
            label["event_id"] == event_id
            and label["scene_id"] == record["scene_id"]
            and label["label_source"] == record["label_source"]
            and len(rows) == record["candidate_count"]
            and [row["branch_index"] for row in rows] == list(range(len(rows)))
        ):
            raise RuntimeError(f"label identity drift: {event_id}")
        feature = (BASE / record["path"]).resolve()
        if (
            BASE not in feature.parents
            or feature.is_symlink()
            or feature.stat().st_size != record["bytes"]
            or core.sha256_file(feature) != record["sha256"]
        ):
            raise RuntimeError(f"online feature provenance drift: {event_id}")
        step = int(label["online_feature_relative_step"])
        with np.load(feature, allow_pickle=False) as shard:
            mask = shard["candidate_mask"]
            if not (
                0 <= step < mask.shape[0]
                and mask.shape[1] == len(rows)
                and bool(mask[step].all())
            ):
                raise RuntimeError(f"online feature alignment drift: {event_id}")
        for row in rows:
            values = (
                row["commit_cost"], row["checkpointed_excursion_cost"],
                row["option_preservation_gain"],
            )
            all_finite &= all(math.isfinite(float(value)) for value in values)
            macro_failures += int(
                not row["checkpointed_excursion_route"].get("success", False)
            )
            if row["is_target"] and not row["commit_route"].get("success", False):
                failed_target_exact &= abs(float(row["commit_cost"]) - 5.0) <= 1e-6
        labels_total += len(rows)
        reuse[reuse_kind] += 1
        manifest_rows.append({
            "event_id": event_id,
            "scene_id": record["scene_id"],
            "label_source": record["label_source"],
            "source_wave": record["source_wave"],
            "candidate_count": len(rows),
            "path": os.path.relpath(label_path, OUT),
            "bytes": label_path.stat().st_size,
            "sha256": core.sha256_file(label_path),
            "online_feature_path": str(feature.relative_to(ROOT)),
            "online_feature_sha256": record["sha256"],
            "label_provenance": reuse_kind,
        })

    counts = Counter(row["label_source"] for row in manifest_rows)
    manifest = {
        "schema_version": "revealnav-mf2-branch-excursion-label-manifest/5.1",
        "records": manifest_rows,
        "metadata": {
            "protocol_sha256": protocol_sha,
            "events": len(manifest_rows),
            "branch_labels": labels_total,
            "label_provenance": dict(reuse),
            "bounded_macro_failures_retained": macro_failures,
            "future_information_used_for_online_input": 0,
            "gold_payload_read": False,
            "training_authorized": True,
            "paper_result": False,
        },
    }
    core.atomic_json(MANIFEST, manifest)
    gates = {
        "all_1830_train_events_complete": len(manifest_rows) == 1830,
        "event_ids_unique": len({row["event_id"] for row in manifest_rows}) == 1830,
        "all_branch_labels_complete_and_finite": all_finite and labels_total >= 3660,
        "failed_target_routes_have_exact_bounded_cost": failed_target_exact,
        "all_four_authorized_sources_present": counts == EXPECTED_SOURCES,
        "accepted_v4_reuse_exact": reuse["accepted_v4_reuse"] == 424,
        "new_scale_labels_exact": reuse["new_v5_scale_label"] == 1406,
        "frozen_failures_retained_as_bounded_labels": macro_failures >= 1,
        "no_development_or_gold_events": True,
        "no_part_files": not list(OUT.rglob("*.part")),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-label-gate/5.1",
        "status": "BRANCH_EXCURSION_EXPANDED_LABEL_GATE_PASS" if passed
                  else "BRANCH_EXCURSION_EXPANDED_LABEL_GATE_FAIL",
        "counts": {
            "events": len(manifest_rows),
            "branch_labels": labels_total,
            "bounded_macro_failures_retained": macro_failures,
            "label_sources": dict(counts),
            "label_provenance": dict(reuse),
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
        "next_gate": "scene-disjoint regret training" if passed else "repair labels",
    }
    core.atomic_json(GATE, value)
    print(json.dumps({
        "status": value["status"], "counts": value["counts"], "gates": gates,
    }, indent=2))
    return 0 if passed else 1


def run(gpus: tuple[int, ...], jobs: int) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("V5.1 protocol must be sealed without drift")
    smoke_value = json.loads(SMOKE.read_text()) if SMOKE.exists() else {}
    protocol_sha = core.sha256_file(PROTOCOL)
    if not (
        smoke_value.get("status") == "BRANCH_EXCURSION_V5_1_SMOKE_PASS"
        and smoke_value.get("protocol_sha256") == protocol_sha
    ):
        raise RuntimeError("V5.1 cross-wave smoke must pass before full generation")
    if not gpus or jobs < 1:
        raise ValueError("at least one GPU and worker are required")
    RUNS.mkdir(parents=True, exist_ok=True)
    event_ids = [row["event_id"] for row in train_records()]
    started = time.monotonic()
    failures: list[str] = []
    completed = 0
    write_progress(len(event_ids), completed, failures, started)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                run_one, event_id, gpus[index % len(gpus)], protocol_sha
            ): event_id
            for index, event_id in enumerate(event_ids)
        }
        for future in concurrent.futures.as_completed(futures):
            event_id, returncode, output = future.result()
            completed += 1
            if returncode:
                failures.append(event_id)
                print(f"FAIL {event_id}\n{output}", flush=True)
            elif completed % 25 == 0 or completed == len(event_ids):
                print(
                    f"PROGRESS {completed}/{len(event_ids)} failures={len(failures)}",
                    flush=True,
                )
            write_progress(len(event_ids), completed, failures, started)
    if failures:
        return 1
    return aggregate(protocol_sha)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if len(gpus) != len(set(gpus)):
        raise ValueError("duplicate GPU index")
    if args.seal:
        return seal()
    if args.smoke:
        return smoke(gpus)
    if args.aggregate:
        return aggregate(core.sha256_file(PROTOCOL))
    return run(gpus, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
