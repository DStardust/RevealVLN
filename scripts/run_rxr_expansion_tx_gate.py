#!/usr/bin/env python3
"""Expand the sealed resource-conditioned T_X factory to all 525 events."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_queue50_tx_gate as sealed  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
TX = BASE / "tx_gate"
RUNS = TX / "runs"
PLAN = TX / "RXR_EXPANSION_TX_PLAN.json"
OUT = TX / "RXR_EXPANSION_TX_GATE.json"
WORKER = ROOT / "scripts/rxr_expansion_tx_worker.py"
PYTHON = Path(sys.executable)
SELECTION = BASE / "human_pilot_300/RXR_HUMAN_PILOT_300_SELECTION.json"
AUTOMATIC = BASE / "RXR_EXPANSION_AUTOMATIC_FILTER_ACCEPTANCE.json"
GEOMETRY = BASE / "geometry/RXR_EXPANSION_DIRECTED_GEOMETRY.json"
CONTROLLER = BASE / "geometry/RXR_EXPANSION_CONTROLLER_EXECUTION.json"
ANALYSIS = BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json"
LANGUAGE = BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_PREFIX_LANGUAGE_GATE.json"
FOLLOWER = ROOT / "third_party/ETP-R1/habitat_extensions/shortest_path_follower.py"
FROZEN = ROOT / "FROZEN_SPEC.md"
PROTOCOL = ROOT / "PHASE0_PROTOCOL.md"
EXPECTED_SPEC = {
    FROZEN: "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    PROTOCOL: "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
}
EXPECTED_FOLLOWER = "d5e5890ad35c1bc73525505da875df8fe314f8d727f48681345bdde16702b7fb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(value)
    os.replace(part, path)


def command(arguments, environment=None):
    result = subprocess.run(
        arguments, cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result


def build_plan():
    automatic = json.loads(AUTOMATIC.read_text())
    selection = json.loads(SELECTION.read_text())
    event_ids = list(automatic["eligible_event_ids"])
    audited = [row["event_id"] for row in selection["items"]]
    if (automatic["status"] != "PASS_READY_FOR_300_HUMAN_PILOT"
            or len(event_ids) != len(set(event_ids))
            or len(event_ids) != 525
            or len(audited) != len(set(audited))
            or len(audited) != 300
            or not set(audited) <= set(event_ids)):
        raise RuntimeError("T_X plan input closure failure")
    source_paths = [AUTOMATIC, SELECTION, GEOMETRY, CONTROLLER, ANALYSIS,
                    LANGUAGE, FOLLOWER, FROZEN, PROTOCOL]
    source_sha = {str(path.relative_to(ROOT)): sha256_file(path)
                  for path in source_paths}
    for path, expected in EXPECTED_SPEC.items():
        if source_sha[str(path.relative_to(ROOT))] != expected:
            raise RuntimeError("frozen document drift")
    if source_sha[str(FOLLOWER.relative_to(ROOT))] != EXPECTED_FOLLOWER:
        raise RuntimeError("frozen controller drift")
    return {
        "manifest": "RevealNav RxR expansion resource-conditioned T_X plan",
        "revision": "rxr-expansion-tx-plan/1",
        "status": "SEALED_BEFORE_HUMAN_LABEL_JOIN",
        "definition": "T_X(B)=max{t:C*_t<=B and witnessed sequence is safe}",
        "normalized_budgets": list(sealed.NORMALIZED_BUDGETS),
        "controllers": list(sealed.CONTROLLERS),
        "minimum_unique_budget_count": 2,
        "eligible_event_ids": event_ids,
        "human_audit_subset_event_ids": audited,
        "source_sha256": source_sha,
        "future_suffix_use": "offline last-passage label only",
        "human_labels_created": 0,
        "training_authorized": False,
    }


def existing_run_valid(path: Path, event_id: str, plan_sha: str) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = json.loads(path.read_text())
        evidence = value["evidence"]
        sources = {row["path"]: row["sha256"]
                   for row in evidence["source_manifest"]}
        return (
            value["revision"] == "rxr-expansion-tx-worker-run/1"
            and value["event_id"] == event_id
            and sealed.stable_sha(evidence) == value["event_evidence_sha256"]
            and sources[str(PLAN.relative_to(ROOT))] == plan_sha
        )
    except Exception:
        return False


def run_gpu_batch(round_name: str, gpu: int, event_ids, plan_sha: str):
    rows = []
    environment = {**os.environ, "RXR_TX_PLAN_SHA256": plan_sha}
    for event_id in event_ids:
        output = RUNS / round_name / (event_id + ".json")
        if existing_run_valid(output, event_id, plan_sha):
            rows.append({"event_id": event_id, "gpu": gpu,
                         "path": str(output.relative_to(ROOT)),
                         "status": "REUSED", "returncode": 0})
            print(round_name, event_id, "REUSED", flush=True)
            continue
        result = command([
            str(PYTHON), str(WORKER), "--event-id", event_id,
            "--gpu", str(gpu), "--output", str(output),
        ], environment)
        atomic_text(output.with_suffix(".stdout"), result.stdout)
        atomic_text(output.with_suffix(".stderr"), result.stderr)
        rows.append({"event_id": event_id, "gpu": gpu,
                     "path": str(output.relative_to(ROOT)),
                     "status": "EXECUTED", "returncode": result.returncode})
        print(round_name, event_id, "rc=%d" % result.returncode, flush=True)
    return rows


def execute_round(round_name: str, event_ids, gpus, plan_sha: str):
    assignments = {gpu: [] for gpu in gpus}
    for index, event_id in enumerate(event_ids):
        assignments[gpus[index % len(gpus)]].append(event_id)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_gpu_batch, round_name, gpu, values,
                               plan_sha)
                   for gpu, values in assignments.items()]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    return sorted(rows, key=lambda row: row["event_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or not gpus:
        raise SystemExit("GPU list must be nonempty and unique")
    plan = build_plan()
    TX.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        existing = json.loads(PLAN.read_text())
        if existing != plan:
            raise SystemExit("existing sealed T_X plan drift")
    else:
        atomic_json(PLAN, plan)
    plan_sha = sha256_file(PLAN)
    environment = {**os.environ, "RXR_TX_PLAN_SHA256": plan_sha}
    self_test = command([str(PYTHON), str(WORKER), "--self-test"], environment)
    if self_test.returncode != 0:
        raise SystemExit("T_X worker self-test failed")
    if args.plan_only:
        print(json.dumps({
            "status": "TX_PLAN_PASS",
            "events": len(plan["eligible_event_ids"]),
            "human_audit_subset": len(plan["human_audit_subset_event_ids"]),
            "plan": str(PLAN.relative_to(ROOT)),
            "sha256": plan_sha,
        }, indent=2))
        return 0
    event_ids = plan["eligible_event_ids"]
    round1 = execute_round("round1", event_ids, gpus, plan_sha)
    round2 = execute_round("round2", event_ids, gpus, plan_sha)
    if any(row["returncode"] for row in round1 + round2):
        raise SystemExit("one or more T_X workers failed")

    summaries = []
    for event_id in event_ids:
        summaries.append(sealed.summarize_event(
            RUNS / "round1" / (event_id + ".json"),
            RUNS / "round2" / (event_id + ".json")))
    exact = sum(row["independent_process_exact_reproduction"]
                for row in summaries)
    complete = sum(row["complete_hashed_cost_evidence"]
                   for row in summaries)
    admitted = [row for row in summaries
                if row["passes_frozen_two_budget_gate"]]
    nontrivial = sum(any(row["nontrivial"].values()) for row in summaries)
    frontier_counts = {
        controller: {str(budget): Counter()
                     for budget in sealed.NORMALIZED_BUDGETS}
        for controller in sealed.CONTROLLERS}
    for row in summaries:
        for controller in sealed.CONTROLLERS:
            for budget, status in row["frontier_status"][controller].items():
                frontier_counts[controller][budget][status] += 1
    source_unchanged = all(
        sha256_file(ROOT / path) == expected
        for path, expected in plan["source_sha256"].items())
    parts = [str(path.relative_to(ROOT)) for path in TX.rglob("*.part")]
    gates = {
        "all_525_complete": complete == len(event_ids) == 525,
        "all_525_exactly_reproduced": exact == len(event_ids),
        "frozen_two_budget_fraction_at_least_60pct":
            len(admitted) / len(event_ids) >= 0.60,
        "all_events_nontrivial": nontrivial == len(event_ids),
        "sealed_sources_unchanged": source_unchanged,
        "no_part_files": not parts,
    }
    output = {
        "manifest": "RevealNav RxR expansion resource-conditioned T_X gate",
        "revision": "rxr-expansion-tx-gate/1",
        "status": ("TX_EXPANSION_PASS_HUMAN_JOIN_REQUIRED"
                   if all(gates.values()) else "TX_EXPANSION_FAIL"),
        "plan": {"path": str(PLAN.relative_to(ROOT)),
                 "sha256": plan_sha},
        "counts": {
            "input_events": len(event_ids),
            "human_audit_subset_events": len(
                plan["human_audit_subset_event_ids"]),
            "complete_events": complete,
            "exactly_reproduced_events": exact,
            "tx_admitted_events": len(admitted),
            "tx_admitted_fraction": len(admitted) / len(event_ids),
            "tx_admitted_scenes": len({row["scene_id"] for row in admitted}),
            "nontrivial_events": nontrivial,
            "frontier_status_by_controller_budget": {
                controller: {budget: dict(sorted(values.items()))
                             for budget, values in budgets.items()}
                for controller, budgets in frontier_counts.items()},
        },
        "tx_admitted_event_ids": [row["event_id"] for row in admitted],
        "events": summaries,
        "execution": {"gpus": gpus, "round1": round1, "round2": round2},
        "gates": gates,
        "failures": [name for name, passed in gates.items() if not passed],
        "future_suffix_used_only_for_offline_label": True,
        "online_future_information_used": 0,
        "human_labels_created": 0,
        "training_authorized": False,
        "forbidden_split_accessed": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"], "counts": output["counts"],
        "gates": gates, "failures": output["failures"],
        "output": str(OUT.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
