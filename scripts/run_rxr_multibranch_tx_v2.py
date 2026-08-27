#!/usr/bin/env python3
"""Seal, execute twice, and aggregate MF2-CR6 per-branch T_X labels."""

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
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
PLAN = V2 / "RXR_MULTIBRANCH_TX_V2_PLAN.json"
OUT = V2 / "RXR_MULTIBRANCH_TX_V2_GATE.json"
RUNS = V2 / "tx_runs"
WORKER = ROOT / "scripts/rxr_multibranch_tx_v2_worker.py"
INDEX = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
GEOMETRY = V2 / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json"
CONTROLLER = V2 / "RXR_MULTIBRANCH_CONTROLLER_EXECUTION_V2.json"
CAUSAL = V2 / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json"
LANGUAGE = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_GATE_V2.json"
FOLLOWER = ROOT / "third_party/ETP-R1/habitat_extensions/shortest_path_follower.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def build_plan():
    index = json.loads(INDEX.read_text())
    if index.get("status") != "FEATURE_AND_TX_GENERATION_REQUIRED" \
            or index.get("resource_label_generation_authorized") is not True:
        raise RuntimeError("training index does not authorize resource labels")
    events = [row["event_id"] for row in index["records"]]
    if not events or len(events) != len(set(events)):
        raise RuntimeError("invalid T_X event population")
    sources = (GEOMETRY, CONTROLLER, CAUSAL, LANGUAGE, FOLLOWER)
    return {
        "schema_version": "revealnav-mf2-multibranch-tx-plan/2",
        "status": "SEALED_BEFORE_RESOURCE_EXECUTION",
        "eligible_event_ids": events,
        "normalized_budgets": [1.5, 2.0, 3.0, 4.0],
        "controllers": ["oracle_greedy", "frozen_shortest_path_compat"],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in sources
        },
        "training_index": {
            "path": str(INDEX.relative_to(ROOT)), "sha256": sha256_file(INDEX)
        },
        "online_future_information_used": 0,
        "training_authorized": False,
    }


def valid_run(path: Path, event_id: str):
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = json.loads(path.read_text())
        return (
            value["schema_version"] ==
            "revealnav-mf2-multibranch-tx-worker-run/2"
            and value["event_id"] == event_id
            and value["event_evidence_sha256"] == hashlib.sha256(
                json.dumps(value["evidence"], ensure_ascii=True,
                           sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
    except Exception:
        return False


def run_lane(round_name, gpu, events, plan_sha):
    environment = {**os.environ,
                   "RXR_MULTIBRANCH_TX_PLAN_SHA256": plan_sha}
    results = []
    for event_id in events:
        path = RUNS / round_name / (event_id + ".json")
        if valid_run(path, event_id):
            results.append((event_id, 0, "REUSED"))
            print(round_name, event_id, "REUSED", flush=True)
            continue
        completed = subprocess.run([
            sys.executable, str(WORKER), "--event-id", event_id,
            "--gpu", str(gpu), "--output", str(path),
        ], cwd=ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".stdout").write_text(completed.stdout)
        path.with_suffix(".stderr").write_text(completed.stderr)
        results.append((event_id, completed.returncode, "EXECUTED"))
        print(round_name, event_id, "rc=" + str(completed.returncode), flush=True)
    return results


def execute_round(round_name, events, gpu_slots, plan_sha):
    lanes = [[] for _ in gpu_slots]
    for index, event_id in enumerate(events):
        lanes[index % len(gpu_slots)].append(event_id)
    results = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpu_slots)) as pool:
        futures = [
            pool.submit(run_lane, round_name, gpu, lane, plan_sha)
            for gpu, lane in zip(gpu_slots, lanes)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--gpu-slots")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("GPU list is invalid")
    gpu_slots = (
        [int(value) for value in args.gpu_slots.split(",")]
        if args.gpu_slots else gpus
    )
    if not gpu_slots or not set(gpu_slots) <= set(gpus):
        raise SystemExit("GPU slot list is invalid")
    plan = build_plan()
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != plan:
            raise SystemExit("sealed multi-branch T_X plan drift")
    else:
        atomic_json(PLAN, plan)
    plan_sha = sha256_file(PLAN)
    if args.plan_only:
        print(json.dumps({"status": "PLAN_PASS", "events": len(
            plan["eligible_event_ids"]), "sha256": plan_sha}, indent=2))
        return 0
    events = plan["eligible_event_ids"]
    first = execute_round("round1", events, gpu_slots, plan_sha)
    second = execute_round("round2", events, gpu_slots, plan_sha)
    failures = [row for row in first + second if row[1] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} T_X worker executions failed")
    summaries = []
    for event_id in events:
        left = json.loads((RUNS / "round1" / (event_id + ".json")).read_text())
        right = json.loads((RUNS / "round2" / (event_id + ".json")).read_text())
        evidence = left["evidence"]
        branch_count = len(evidence["candidate_branch_ids"])
        complete = all(
            len(branch["controllers"]) == 2 and all(
                controller["complete_prefix_evidence"]
                for controller in branch["controllers"].values()
            ) for branch in evidence["branches"].values()
        )
        summaries.append({
            "event_id": event_id,
            "candidate_branch_count": branch_count,
            "exact_independent_reproduction": (
                left["event_evidence_sha256"] == right["event_evidence_sha256"]
            ),
            "complete_per_branch_evidence": complete,
            "round1_path": str((RUNS / "round1" / (event_id + ".json"))
                               .relative_to(ROOT)),
            "round1_sha256": sha256_file(
                RUNS / "round1" / (event_id + ".json")
            ),
            "round2_path": str((RUNS / "round2" / (event_id + ".json"))
                               .relative_to(ROOT)),
            "round2_sha256": sha256_file(
                RUNS / "round2" / (event_id + ".json")
            ),
        })
    gates = {
        "all_events_complete": all(row["complete_per_branch_evidence"]
                                    for row in summaries),
        "all_events_exactly_reproduced": all(
            row["exact_independent_reproduction"] for row in summaries
        ),
        "no_part_files": not list(V2.rglob("*.part")),
    }
    output = {
        "schema_version": "revealnav-mf2-multibranch-tx-gate/2",
        "status": "MULTIBRANCH_TX_PASS" if all(gates.values()) else
                  "MULTIBRANCH_TX_FAIL",
        "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": plan_sha},
        "counts": {
            "events": len(summaries),
            "branches": sum(row["candidate_branch_count"] for row in summaries),
            "two_branch_events": sum(row["candidate_branch_count"] == 2
                                     for row in summaries),
            "three_or_four_branch_events": sum(row["candidate_branch_count"] >= 3
                                               for row in summaries),
        },
        "events": summaries,
        "gates": gates,
        "future_information_used_for_online_input": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({"status": output["status"], "counts": output["counts"],
                      "gates": gates, "output": str(OUT.relative_to(ROOT))},
                     indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
