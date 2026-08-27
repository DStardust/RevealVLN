#!/usr/bin/env python3
"""Independent structural validation and scientific adjudication of CR5 T_X."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
TX = BASE / "tx_gate"
GATE = TX / "CR5_QUEUE50_TX_GATE.json"
HUMAN = BASE / "causal_gate/CR5_QUEUE50_HUMAN50_ACCEPTANCE.json"
VALIDATION = TX / "CR5_QUEUE50_TX_INDEPENDENT_VALIDATION.json"
ADJUDICATION = TX / "CR5_QUEUE50_TX_SCIENTIFIC_ADJUDICATION.json"
EXPECTED_GATE_SHA256 = (
    "b9fbe7af6e3ca589c42933d27a8640b16e0b0ada72e81d7f8dd5b3f6e77dbc40"
)
EXPECTED_HUMAN_SHA256 = (
    "fa0e126be303d53767b367ab90673ec4914282c589583cfa6178ccf4f7e3e681"
)
BUDGETS = (1.5, 2.0, 3.0, 4.0)
CONTROLLERS = ("oracle_greedy", "frozen_shortest_path_compat")


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


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    os.replace(part, path)


def classify(feasible, indices):
    if not any(feasible):
        return "NEVER_FEASIBLE", None
    last = len(feasible) - 1 - list(reversed(feasible)).index(True)
    if last == len(feasible) - 1:
        return "RIGHT_CENSORED", None
    seen_safe = False
    unsafe_after_safe = False
    reentry = False
    for value in feasible[:last + 1]:
        if value:
            reentry |= unsafe_after_safe
            seen_safe = True
        elif seen_safe:
            unsafe_after_safe = True
    return (
        "UNIQUE_LAST_SAFE_WITH_REENTRY" if reentry
        else "UNIQUE_LAST_SAFE_MONOTONE",
        indices[last],
    )


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)) / denominator
    return [round(center - half, 6), round(center + half, 6)]


def replay_hash_valid(replay):
    if "replay_sha256" not in replay:
        return replay == {
            "status": "NOT_STRICTLY_REVEALED", "success": False}
    core = dict(replay)
    expected = core.pop("replay_sha256")
    return stable_sha(core) == expected


def validate_event(summary, failures):
    first_path = ROOT / summary["round1"]["path"]
    second_path = ROOT / summary["round2"]["path"]
    if any(
        not path.is_file() or path.is_symlink()
        or ROOT.resolve() not in path.resolve().parents
        for path in (first_path, second_path)
    ):
        failures.append(summary["event_id"] + ": unsafe run reference")
        return None
    first, second = load(first_path), load(second_path)
    evidence = first["evidence"]
    event_id = summary["event_id"]
    if (
        first["event_id"] != event_id or second["event_id"] != event_id
        or sha256_file(first_path) != summary["round1"]["file_sha256"]
        or sha256_file(second_path) != summary["round2"]["file_sha256"]
        or first["event_evidence_sha256"]
        != second["event_evidence_sha256"]
        or first["evidence"] != second["evidence"]
        or stable_sha(evidence) != first["event_evidence_sha256"]
    ):
        failures.append(event_id + ": independent reproduction mismatch")
        return None

    start, end = evidence["observed_prefix_horizon"]
    indices = list(range(start, end + 1))
    recomputed = {}
    for controller in CONTROLLERS:
        payload = evidence["controllers"][controller]
        if not replay_hash_valid(payload["checkpoint_to_target_normalization"]):
            failures.append(event_id + ": normalization replay hash")
        rows = payload["prefix_costs"]
        if len(rows) != len(indices):
            failures.append(event_id + ": prefix row count")
            continue
        parent = None
        for prefix_index, row in zip(indices, rows):
            core = dict(row)
            prefix_hash = core.pop("cost_prefix_sha256")
            if (
                row["prefix_index"] != prefix_index
                or row["parent_cost_prefix_sha256"] != parent
                or stable_sha(core) != prefix_hash
                or not replay_hash_valid(row["direct"])
                or not replay_hash_valid(row["saved_via_checkpoint"])
            ):
                failures.append(event_id + ": prefix evidence mismatch")
                break
            candidates = []
            if row["direct"]["success"]:
                candidates.append((row["direct"]["action_count"], "direct",
                                   row["direct"]["replay_sha256"]))
            if row["saved_via_checkpoint"]["success"]:
                candidates.append((
                    row["saved_via_checkpoint"]["action_count"], "saved",
                    row["saved_via_checkpoint"]["replay_sha256"]))
            candidates.sort(key=lambda item: (item[0],
                                              item[1] != "direct"))
            best = candidates[0] if candidates else (None, None, None)
            if (
                [row["cstar_action_count"], row["cstar_source"],
                 row["cstar_replay_sha256"]] != list(best)
            ):
                failures.append(event_id + ": Cstar mismatch")
                break
            parent = prefix_hash

        recomputed[controller] = {}
        normalization = payload["checkpoint_to_target_normalization"]
        for budget in BUDGETS:
            declared = payload["frontiers"][str(budget)]
            if not normalization["success"]:
                observed = ("CONTROLLER_NORMALIZATION_FAIL", None)
            else:
                absolute = budget * payload[
                    "normalization_denominator_actions"]
                feasible = [
                    row["cstar_action_count"] is not None
                    and row["cstar_action_count"] <= absolute
                    for row in rows
                ]
                observed = classify(feasible, indices)
            if (
                declared["status"] != observed[0]
                or declared["last_safe_prefix"] != observed[1]
            ):
                failures.append(event_id + ": frontier mismatch")
            recomputed[controller][str(budget)] = observed[0]
    return {
        "event_id": event_id,
        "evidence_sha256": first["event_evidence_sha256"],
        "frontiers": recomputed,
    }


def main() -> int:
    failures = []
    if sha256_file(GATE) != EXPECTED_GATE_SHA256:
        failures.append("gate SHA drift")
    if sha256_file(HUMAN) != EXPECTED_HUMAN_SHA256:
        failures.append("human acceptance SHA drift")
    gate = load(GATE)
    human = load(HUMAN)
    event_ids = [row["event_id"] for row in gate["events"]]
    if (
        len(event_ids) != 16 or len(set(event_ids)) != 16
        or set(event_ids) != set(human["eligible_event_ids"])
    ):
        failures.append("event set mismatch")
    validated = [validate_event(row, failures) for row in gate["events"]]
    validated = [row for row in validated if row is not None]

    admitted = [
        row for row in gate["events"]
        if row["passes_frozen_two_budget_gate"]]
    admitted_ids = [row["event_id"] for row in admitted]
    if admitted_ids != gate["tx_admitted_event_ids"]:
        failures.append("admitted event ordering mismatch")
    frozen_unique = len(admitted)
    exact = sum(row["independent_process_exact_reproduction"]
                for row in gate["events"])
    complete = sum(row["complete_hashed_cost_evidence"]
                   for row in gate["events"])
    if (
        frozen_unique != 11 or exact != 16 or complete != 16
        or gate["verdict"]
        != "TX_FEASIBILITY_PASS_AUTOMATED_EXPANSION_REQUIRED"
        or gate["training_authorized"] is not False
        or gate["forbidden_split_accessed"] is not False
    ):
        failures.append("aggregate count or boundary mismatch")

    validation = {
        "revision": "cr5-queue50-tx-independent-validation/1",
        "status": "PASS" if not failures else "FAIL",
        "source": {
            "path": str(GATE.relative_to(ROOT)),
            "sha256": sha256_file(GATE),
        },
        "checked_events": len(validated),
        "checks": {
            "source_hashes": not any("SHA drift" in item for item in failures),
            "sealed_event_set_16": len(event_ids) == len(set(event_ids)) == 16,
            "two_independent_runs_and_evidence_digests": exact == 16,
            "route_and_prefix_hashes_recomputed": len(validated) == 16,
            "Cstar_and_four_budget_frontiers_recomputed": len(validated) == 16,
            "aggregate_boundary_recomputed": frozen_unique == 11,
            "training_remains_false": gate["training_authorized"] is False,
        },
        "events": validated,
        "failures": failures,
    }
    atomic_json(VALIDATION, validation)

    checkpoint_change_count = sum(
        row["nontrivial"]["checkpoint_changes_frozen_feasible_set"]
        for row in gate["events"])
    tx_before_reveal_count = sum(
        row["nontrivial"]["tx_before_reveal_any_frozen_budget"]
        for row in gate["events"])
    reentry_event_count = sum(any(
        status == "UNIQUE_LAST_SAFE_WITH_REENTRY"
        for status in row["frontier_status"]
            ["frozen_shortest_path_compat"].values())
        for row in gate["events"])
    end_to_end_yield = frozen_unique / 50
    adjudication = {
        "revision": "cr5-queue50-tx-scientific-adjudication/1",
        "decision": (
            "ENGINEERING_FEASIBILITY_ACCEPTED_TRAINING_STILL_NO_GO"
            if not failures else "EVIDENCE_INVALID"
        ),
        "validated_gate": {
            "path": str(VALIDATION.relative_to(ROOT)),
            "sha256": sha256_file(VALIDATION),
            "status": validation["status"],
        },
        "positive_findings": {
            "strict_T_R_inputs": 16,
            "complete_exactly_reproduced_cost_events": complete,
            "tx_admitted_events": frozen_unique,
            "tx_admitted_scenes": len({row["scene_id"] for row in admitted}),
            "gate4_fraction": frozen_unique / 16,
            "gate4_required_fraction": 0.60,
            "gate4_pass": frozen_unique / 16 >= 0.60,
        },
        "risk_findings": {
            "end_to_end_queue50_tx_yield": end_to_end_yield,
            "end_to_end_queue50_wilson_95ci": wilson(frozen_unique, 50),
            "frozen_pilot_point_target": 0.25,
            "point_target_currently_met": end_to_end_yield >= 0.25,
            "right_censored_or_frozen_normalization_failed_events":
                16 - frozen_unique,
            "events_with_any_frozen_reentry": reentry_event_count,
            "events_where_checkpoint_changes_frozen_feasible_set":
                checkpoint_change_count,
            "checkpoint_specific_fraction": checkpoint_change_count / 16,
            "events_with_tx_before_reveal": tx_before_reveal_count,
            "official_gate5_pass_basis": (
                "16/16 strict reveal onsets occur after episode prefix 0"),
            "stronger_option_specific_25pct_stress_test_pass":
                checkpoint_change_count / 16 >= 0.25,
        },
        "interpretation": [
            "The pre-registered MF2-CR1 60% two-budget feasibility gate passes.",
            "The 22% end-to-end queue yield is below the frozen 25% point "
            "target; its 50-item interval remains too wide for a final GO/NO-GO.",
            "Frequent re-entry supports cost/current-feasibility supervision "
            "and forbids a monotone online countdown interpretation of T_X.",
            "Only three events show checkpoint-specific feasible-set value; "
            "automated expansion must measure this rate without favorable "
            "resampling or post-hoc budget changes.",
        ],
        "next_authorized_work": (
            "Apply the generic target-route-authoritative re-grounding "
            "correction, then expand an unbiased RxR-train candidate queue and "
            "repeat the sealed T_R/T_X factory toward the 300-event pilot."
        ),
        "automated_event_expansion_authorized": not failures,
        "feature_generation_authorized": False,
        "training_authorized": False,
        "failures": failures,
    }
    atomic_json(ADJUDICATION, adjudication)
    print(json.dumps({
        "validation": validation["status"],
        "decision": adjudication["decision"],
        "positive_findings": adjudication["positive_findings"],
        "risk_findings": adjudication["risk_findings"],
        "validation_path": str(VALIDATION.relative_to(ROOT)),
        "validation_sha256": sha256_file(VALIDATION),
        "adjudication_path": str(ADJUDICATION.relative_to(ROOT)),
        "adjudication_sha256": sha256_file(ADJUDICATION),
    }, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
