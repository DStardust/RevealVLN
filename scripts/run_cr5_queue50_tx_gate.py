#!/usr/bin/env python3
"""Run and accept the queue50 CR5 resource-conditioned T_X gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
CAUSAL = BASE / "causal_gate"
TX = BASE / "tx_gate"
RUNS = TX / "runs"
OUT = TX / "CR5_QUEUE50_TX_GATE.json"
LOG = TX / "CR5_QUEUE50_TX_GATE.log"
WORKER = ROOT / "scripts/cr5_queue50_tx_worker.py"
PYTHON = ROOT / ".envs/etpr1/bin/python"
UV = ROOT / ".tools/uv/uv"
ACCEPTANCE = CAUSAL / "CR5_QUEUE50_HUMAN50_ACCEPTANCE.json"
EXPECTED_ACCEPTANCE_SHA256 = (
    "fa0e126be303d53767b367ab90673ec4914282c589583cfa6178ccf4f7e3e681"
)
EXPECTED_SPEC_SHA256 = {
    ROOT / "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    ROOT / "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
}
NORMALIZED_BUDGETS = (1.5, 2.0, 3.0, 4.0)
CONTROLLERS = ("oracle_greedy", "frozen_shortest_path_compat")
MIN_TWO_BUDGET_FRACTION = 0.60
MIN_NONTRIVIAL_FRACTION = 0.25


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(value, encoding="utf-8")
    os.replace(part, path)


def command(arguments, env=None):
    result = subprocess.run(
        arguments, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": [str(value) for value in arguments],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "pass": result.returncode == 0,
    }


def run_gpu_batch(round_name: str, gpu: int, event_ids: list[str]):
    results = []
    for event_id in event_ids:
        output = RUNS / round_name / (event_id + ".json")
        result = command([
            str(PYTHON), str(WORKER), "--event-id", event_id,
            "--gpu", str(gpu), "--output", str(output),
        ])
        stdout_path = output.with_suffix(".stdout")
        stderr_path = output.with_suffix(".stderr")
        atomic_text(stdout_path, result["stdout"])
        atomic_text(stderr_path, result["stderr"])
        results.append({
            "event_id": event_id,
            "gpu": gpu,
            "output": str(output.relative_to(ROOT)),
            "stdout": str(stdout_path.relative_to(ROOT)),
            "stderr": str(stderr_path.relative_to(ROOT)),
            "returncode": result["returncode"],
            "pass": result["pass"],
        })
        print("%s %s gpu=%d rc=%d" % (
            round_name, event_id, gpu, result["returncode"]), flush=True)
    return results


def execute_round(round_name: str, event_ids: list[str], gpus: list[int]):
    assignments = {gpu: [] for gpu in gpus}
    for index, event_id in enumerate(event_ids):
        assignments[gpus[index % len(gpus)]].append(event_id)
    results = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(run_gpu_batch, round_name, gpu, assigned)
            for gpu, assigned in assignments.items() if assigned
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    return sorted(results, key=lambda row: row["event_id"])


def complete_evidence(evidence) -> bool:
    def replay_valid(replay) -> bool:
        if "replay_sha256" not in replay:
            return replay == {
                "status": "NOT_STRICTLY_REVEALED", "success": False}
        core = dict(replay)
        observed = core.pop("replay_sha256")
        return stable_sha(core) == observed

    start, end = evidence["observed_prefix_horizon"]
    count = end - start + 1
    for controller in CONTROLLERS:
        payload = evidence["controllers"][controller]
        if (
            not payload["complete_prefix_evidence"]
            or len(payload["prefix_costs"]) != count
            or set(payload["frontiers"]) != {
                str(value) for value in NORMALIZED_BUDGETS}
        ):
            return False
        if not replay_valid(payload["checkpoint_to_target_normalization"]):
            return False
        previous = None
        for offset, row in enumerate(payload["prefix_costs"]):
            if row["prefix_index"] != start + offset:
                return False
            if row["parent_cost_prefix_sha256"] != previous:
                return False
            core = dict(row)
            observed = core.pop("cost_prefix_sha256")
            if stable_sha(core) != observed:
                return False
            if (
                not replay_valid(row["direct"])
                or not replay_valid(row["saved_via_checkpoint"])
            ):
                return False
            previous = observed
        for frontier in payload["frontiers"].values():
            is_unique = frontier["status"].startswith("UNIQUE_LAST_SAFE")
            if is_unique:
                witness = frontier["safe_witness"]
                certificate = frontier["post_expiry_no_safe_certificate"]
                if witness is None or certificate is None:
                    return False
                witness_core = dict(witness)
                witness_sha = witness_core.pop("witness_sha256")
                certificate_core = dict(certificate)
                certificate_sha = certificate_core.pop("search_sha256")
                if (
                    stable_sha(witness_core) != witness_sha
                    or stable_sha(certificate_core) != certificate_sha
                    or certificate["feasible"] is not False
                    or certificate["prefix_index"]
                    != frontier["last_safe_prefix"] + 1
                ):
                    return False
            elif (
                frontier["safe_witness"] is not None
                or frontier["post_expiry_no_safe_certificate"] is not None
            ):
                return False
    return (
        evidence["network_attempts"] == 0
        and evidence["images_or_observation_tensors_written"] == 0
        and evidence["future_information_used_for_online_input"] == 0
    )


def summarize_event(first_path: Path, second_path: Path):
    first = load_json(first_path)
    second = load_json(second_path)
    evidence = first["evidence"]
    exact = (
        first["event_evidence_sha256"] == second["event_evidence_sha256"]
        and first["evidence"] == second["evidence"]
        and stable_sha(first["evidence"]) == first["event_evidence_sha256"]
    )
    complete = complete_evidence(evidence)
    frozen = evidence["controllers"]["frozen_shortest_path_compat"]
    oracle = evidence["controllers"]["oracle_greedy"]
    frozen_unique = frozen["unique_last_safe_budget_count"]
    oracle_unique = oracle["unique_last_safe_budget_count"]
    statuses = {
        controller: {
            budget: evidence["controllers"][controller]
                ["frontiers"][budget]["status"]
            for budget in sorted(
                evidence["controllers"][controller]["frontiers"],
                key=float)
        }
        for controller in CONTROLLERS
    }
    return {
        "event_id": evidence["event_id"],
        "episode_id": evidence["episode_id"],
        "scene_id": evidence["scene_id"],
        "strict_reveal_interval": evidence["strict_reveal_interval"],
        "observed_prefix_horizon": evidence["observed_prefix_horizon"],
        "round1": {
            "path": str(first_path.relative_to(ROOT)),
            "file_sha256": sha256_file(first_path),
        },
        "round2": {
            "path": str(second_path.relative_to(ROOT)),
            "file_sha256": sha256_file(second_path),
        },
        "event_evidence_sha256": first["event_evidence_sha256"],
        "independent_process_exact_reproduction": exact,
        "complete_hashed_cost_evidence": complete,
        "frontier_status": statuses,
        "frozen_unique_last_safe_budget_count": frozen_unique,
        "oracle_unique_last_safe_budget_count": oracle_unique,
        "passes_frozen_two_budget_gate": (
            exact and complete and frozen_unique >= 2),
        "nontrivial": evidence["nontrivial"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,3,4,5,7")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain unique integer ids")
    if sha256_file(ACCEPTANCE) != EXPECTED_ACCEPTANCE_SHA256:
        raise SystemExit("human50 acceptance SHA drift")
    for path, expected in EXPECTED_SPEC_SHA256.items():
        if sha256_file(path) != expected:
            raise SystemExit("frozen document SHA drift: " + str(path))
    acceptance = load_json(ACCEPTANCE)
    event_ids = list(acceptance["eligible_event_ids"])
    if len(event_ids) != 16 or len(set(event_ids)) != 16:
        raise SystemExit("expected 16 sealed strict T_R events")
    pre_source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in list(EXPECTED_SPEC_SHA256) + [ACCEPTANCE]
    }

    TX.mkdir(parents=True, exist_ok=True)
    negative_controls = {
        "frontier_classifier": command([
            str(PYTHON), str(WORKER), "--self-test",
        ]),
        "reject_non_allowlisted_event": command([
            str(PYTHON), str(WORKER), "--event-id", "forbidden_event",
            "--gpu", str(gpus[0]), "--output",
            str(TX / "negative_forbidden_event.json"),
        ]),
    }
    negative_controls["reject_non_allowlisted_event"]["pass"] = (
        negative_controls["reject_non_allowlisted_event"]["returncode"] != 0
        and not (TX / "negative_forbidden_event.json").exists()
    )

    round1 = execute_round("round1", event_ids, gpus)
    round2 = execute_round("round2", event_ids, gpus)
    worker_runs_pass = all(row["pass"] for row in round1 + round2)
    event_summaries = []
    if worker_runs_pass:
        for event_id in event_ids:
            event_summaries.append(summarize_event(
                RUNS / "round1" / (event_id + ".json"),
                RUNS / "round2" / (event_id + ".json"),
            ))

    exact_count = sum(
        row["independent_process_exact_reproduction"]
        for row in event_summaries)
    complete_count = sum(
        row["complete_hashed_cost_evidence"] for row in event_summaries)
    frozen_two_budget_count = sum(
        row["passes_frozen_two_budget_gate"] for row in event_summaries)
    frozen_two_budget_fraction = (
        frozen_two_budget_count / len(event_ids))
    nontrivial_count = sum(
        any(row["nontrivial"].values()) for row in event_summaries)
    nontrivial_fraction = nontrivial_count / len(event_ids)
    frontier_counts = {
        controller: {str(budget): Counter() for budget in NORMALIZED_BUDGETS}
        for controller in CONTROLLERS
    }
    for row in event_summaries:
        for controller in CONTROLLERS:
            for budget, status in row["frontier_status"][controller].items():
                frontier_counts[controller][budget][status] += 1

    regression = {
        "scripts_compile": command([
            str(PYTHON), "-m", "py_compile", str(WORKER), str(Path(__file__)),
        ]),
        "toporeveal_24": command([
            str(PYTHON), str(ROOT / "tests/test_toporeveal.py"), "-v",
        ], {**os.environ, "PYTHONPATH": str(ROOT)}),
        "uv_pip_check": command([
            str(UV), "pip", "check", "--python", str(PYTHON),
        ]),
    }
    post_source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in list(EXPECTED_SPEC_SHA256) + [ACCEPTANCE]
    }
    source_immutable = pre_source_hashes == post_source_hashes
    reserve_files = sorted(
        (ROOT / ".disk_reserve").glob("reserve_10G_*.bin"))
    reserve_pass = len(reserve_files) == 19 and all(
        path.is_file() and not path.is_symlink()
        and path.stat().st_size == 10_737_418_240
        for path in reserve_files)
    part_files = [
        str(path.relative_to(ROOT)) for path in BASE.rglob("*.part")]
    stat = os.statvfs(ROOT)
    free_bytes = stat.f_bavail * stat.f_frsize

    gates = {
        "sealed_strict_T_R_input_16": len(event_ids) == 16,
        "worker_processes_32_pass": worker_runs_pass,
        "negative_controls_pass": all(
            row["pass"] for row in negative_controls.values()),
        "gate3_complete_two_controller_cost_evidence": (
            complete_count == len(event_ids)),
        "independent_process_exact_reproduction_16": (
            exact_count == len(event_ids)),
        "gate4_frozen_unique_at_least_two_budgets_fraction": (
            frozen_two_budget_fraction >= MIN_TWO_BUDGET_FRACTION),
        "gate5_nontrivial_timing_fraction": (
            nontrivial_fraction >= MIN_NONTRIVIAL_FRACTION),
        "regression_pass": all(row["pass"] for row in regression.values()),
        "frozen_sources_immutable": source_immutable,
        "reserve_contract_pass": reserve_pass,
        "no_part_files": not part_files,
        "free_space_at_least_8_gib": free_bytes >= 8 * 1024 ** 3,
    }
    overall = all(gates.values())
    accepted_tx_ids = [
        row["event_id"] for row in event_summaries
        if row["passes_frozen_two_budget_gate"]
    ]
    accepted_scene_count = len({
        row["scene_id"] for row in event_summaries
        if row["passes_frozen_two_budget_gate"]
    })
    output = {
        "revision": "cr5-queue50-resource-conditioned-tx-gate/1",
        "verdict": (
            "TX_FEASIBILITY_PASS_AUTOMATED_EXPANSION_REQUIRED"
            if overall else "TX_FEASIBILITY_NO_GO"
        ),
        "scope": (
            "16 sealed RxR-CE-en train-only strict T_R engineering events; "
            "resource-conditioned T_X(B), not benchmark or training results"
        ),
        "definition": {
            "cost": "C*_t=min(C_direct_t,C_save_t)",
            "expiry": "T_X(B)=max{t:C*_t<=B and controller sequence is safe}",
            "fixed_normalized_budgets": list(NORMALIZED_BUDGETS),
            "primary_controller": "frozen_shortest_path_compat",
            "secondary_controller": "oracle_greedy",
            "right_censoring": (
                "a feasible final observed prefix has no observed T_X(B)"),
            "reentry": (
                "safe-unsafe-safe retains the unique last passage and is "
                "reported separately"),
            "online_boundary": (
                "offline last-passage labels use future suffixes; the online "
                "model receives only causal current-feasibility/cost inputs"),
        },
        "source": {
            "human50_acceptance_path": str(ACCEPTANCE.relative_to(ROOT)),
            "human50_acceptance_sha256": EXPECTED_ACCEPTANCE_SHA256,
            "source_hashes_before": pre_source_hashes,
            "source_hashes_after": post_source_hashes,
        },
        "execution": {
            "physical_gpus": gpus,
            "round1": round1,
            "round2": round2,
            "independent_processes": len(round1) + len(round2),
        },
        "counts": {
            "strict_T_R_input_events": len(event_ids),
            "input_scenes": len({row["scene_id"] for row in event_summaries}),
            "complete_two_controller_cost_events": complete_count,
            "exactly_reproduced_events": exact_count,
            "frozen_unique_at_least_two_budget_events":
                frozen_two_budget_count,
            "frozen_unique_at_least_two_budget_fraction":
                frozen_two_budget_fraction,
            "required_fraction": MIN_TWO_BUDGET_FRACTION,
            "tx_admitted_scene_count": accepted_scene_count,
            "nontrivial_events": nontrivial_count,
            "nontrivial_fraction": nontrivial_fraction,
            "required_nontrivial_fraction": MIN_NONTRIVIAL_FRACTION,
            "frontier_status_by_controller_budget": {
                controller: {
                    budget: dict(sorted(counts.items()))
                    for budget, counts in by_budget.items()
                }
                for controller, by_budget in frontier_counts.items()
            },
        },
        "tx_admitted_event_ids": accepted_tx_ids,
        "events": event_summaries,
        "negative_controls": negative_controls,
        "regression": regression,
        "integrity": {
            "reserve_file_count": len(reserve_files),
            "reserve_contract_pass": reserve_pass,
            "part_files": part_files,
            "free_bytes": free_bytes,
            "free_space_floor_bytes": 8 * 1024 ** 3,
        },
        "gates": gates,
        "next_stage": (
            "Implement the generic target-route-authoritative re-grounding "
            "correction, then automatically expand the train-only event pool "
            "toward the frozen 300-event pilot."
            if overall else
            "Stop and adjudicate the failed T_X feasibility gate without "
            "changing budgets or thresholds post hoc."
        ),
        "automated_event_expansion_authorized": overall,
        "feature_generation_authorized": False,
        "training_authorized": False,
        "forbidden_split_accessed": False,
        "dependencies_installed": 0,
        "failures": [name for name, passed in gates.items() if not passed],
    }
    atomic_text(OUT, json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    atomic_text(LOG, "\n".join([
        "verdict=" + output["verdict"],
        "strict_T_R_input_events=%d" % len(event_ids),
        "complete_two_controller_cost_events=%d" % complete_count,
        "exactly_reproduced_events=%d" % exact_count,
        "frozen_two_budget_events=%d" % frozen_two_budget_count,
        "frozen_two_budget_fraction=%.6f" % frozen_two_budget_fraction,
        "nontrivial_fraction=%.6f" % nontrivial_fraction,
        "failures=" + json.dumps(output["failures"]),
        "acceptance_sha256=" + sha256_file(OUT),
    ]) + "\n")
    print(json.dumps({
        "verdict": output["verdict"],
        "counts": output["counts"],
        "gates": gates,
        "failures": output["failures"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
