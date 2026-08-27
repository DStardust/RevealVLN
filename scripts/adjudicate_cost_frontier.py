#!/usr/bin/env python3
"""Read-only mathematical adjudication of the Phase-0C cost witness.

The original witness conservatively called any safe->unsafe->safe sequence
NON_MONOTONE and excluded it from the unique-T_X gate.  MF2-CR1, however,
defines T_X(B) as a maximum over feasible prefixes.  A finite observed set has
a unique maximum even when feasibility re-enters.  This script preserves the
raw FAIL artifact and emits a versioned classification with re-entry reported
as a separate scientific risk.
"""

import hashlib
import json
import os
from collections import Counter


ROOT = "/mnt/daiyang/vla"
RAW = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHASE0C_COST_FRONTIER_WITNESS.json")
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHASE0C_COST_FRONTIER_ADJUDICATION.json")
EXPECTED_RAW_SHA = \
    "9b59ea9b7b9995aeb604b00587dd79a3af863ea51b16ddfe805b3f719f1a16d1"
BUDGETS = (1.5, 2.0, 3.0, 4.0)
MIN_UNIQUE_BUDGETS = 2
MIN_FRACTION = 0.60


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def classify(feasible, offset):
    if not any(feasible):
        return "NEVER_FEASIBLE", None, 0
    last = len(feasible) - 1 - list(reversed(feasible)).index(True)
    if last == len(feasible) - 1:
        return "RIGHT_CENSORED", None, 0
    # Re-entry is specifically safe -> unsafe -> safe.  An initially
    # infeasible prefix followed by its first feasible prefix is ordinary
    # entry, not re-entry.
    seen_safe = False
    seen_unsafe_after_safe = False
    reentry = False
    for value in feasible[:last + 1]:
        if value:
            if seen_unsafe_after_safe:
                reentry = True
            seen_safe = True
        elif seen_safe:
            seen_unsafe_after_safe = True
    status = ("UNIQUE_LAST_SAFE_WITH_REENTRY" if reentry else
              "UNIQUE_LAST_SAFE_MONOTONE")
    transitions = sum(feasible[k] != feasible[k - 1]
                      for k in range(1, len(feasible)))
    return status, offset + last, transitions


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    actual = sha256_file(RAW)
    if actual != EXPECTED_RAW_SHA:
        raise SystemExit("raw cost witness SHA drift")
    with open(RAW) as fh:
        raw = json.load(fh)
    if raw.get("status") != "FAIL" or raw.get("gates", {}).get(
            "gate3_complete_cost_evidence") is not True:
        raise SystemExit("unexpected raw witness state")

    controllers = ("oracle_greedy", "frozen_shortest_path_compat")
    counts = {c: {str(b): Counter() for b in BUDGETS}
              for c in controllers}
    adjudicated_events = []
    correspondence_ok = True
    for event in raw["events"]:
        event_out = {"provisional_event_id": event["provisional_event_id"],
                     "episode_id": event["episode_id"],
                     "scene_id": event["scene_id"],
                     "controllers": {}}
        for controller in controllers:
            data = event["controllers"][controller]
            offset = int(event["checkpoint_prefix"])
            rows = data["prefix_costs"][offset:]
            denom = data["normalization_denominator_actions"]
            frontiers = {}
            unique = 0
            reentry_budgets = 0
            for budget in BUDGETS:
                feasible = [
                    row["cstar_action_count"] is not None and
                    row["cstar_action_count"] <= budget * denom
                    for row in rows]
                status, last, transitions = classify(feasible, offset)
                counts[controller][str(budget)][status] += 1
                unique += int(status.startswith("UNIQUE_LAST_SAFE"))
                reentry_budgets += int(status.endswith("WITH_REENTRY"))
                old = data["frontiers"][str(budget)]["status"]
                expected_old = {
                    "UNIQUE_LAST_SAFE_MONOTONE": "UNIQUE_OBSERVED",
                    "UNIQUE_LAST_SAFE_WITH_REENTRY": "NON_MONOTONE",
                    "RIGHT_CENSORED": "RIGHT_CENSORED",
                    "NEVER_FEASIBLE": "NEVER_FEASIBLE",
                }[status]
                correspondence_ok &= old == expected_old
                frontiers[str(budget)] = {
                    "status": status,
                    "last_safe_prefix": last,
                    "feasibility_transition_count": transitions,
                    "raw_witness_status": old,
                    "raw_status_correspondence_ok": old == expected_old,
                }
            event_out["controllers"][controller] = {
                "frontiers": frontiers,
                "unique_last_safe_budget_count": unique,
                "reentry_budget_count": reentry_budgets,
                "passes_two_budget_gate": unique >= MIN_UNIQUE_BUDGETS,
            }
        adjudicated_events.append(event_out)

    frozen_pass = sum(
        x["controllers"]["frozen_shortest_path_compat"]
        ["passes_two_budget_gate"] for x in adjudicated_events)
    oracle_pass = sum(
        x["controllers"]["oracle_greedy"]["passes_two_budget_gate"]
        for x in adjudicated_events)
    n = len(adjudicated_events)
    frozen_fraction = frozen_pass / n
    gate4 = frozen_fraction >= MIN_FRACTION and correspondence_ok
    raw_immutable = sha256_file(RAW) == EXPECTED_RAW_SHA
    output = {
        "gate": "mf2_cr1_cost_frontier_mathematical_adjudication",
        "revision": "cost-frontier-adjudication/1",
        "status": "PASS" if gate4 else "FAIL",
        "decision": "GATE4_PASS_WITH_REENTRY_RISK" if gate4 else
                    "GATE4_NO_GO",
        "raw_witness": {"path": os.path.relpath(RAW, ROOT),
                        "sha256": actual, "preserved_status": raw["status"],
                        "preserved_decision": raw["decision"],
                        "immutable_after_adjudication": raw_immutable},
        "mathematical_basis": {
            "definition": "T_X(B)=max{t: C*_t <= B_t and safe}",
            "correction": "safe->unsafe->safe is re-entry, not multiple "
                          "maxima; the last feasible prefix remains unique "
                          "when followed by an observed infeasible suffix",
            "right_censoring": "if the last observed prefix is feasible, "
                               "T_X is not observed",
            "important_risk": "last-safe is a future-dependent last-passage "
                              "time, not a first-passage stopping time. REE "
                              "must predict current feasibility separately "
                              "and report re-entry rate.",
        },
        "gates": {
            "raw_gate3_complete_cost_evidence": True,
            "raw_gate5_nontrivial_pass": raw["gates"]["gate5_pass"],
            "status_correspondence_104x2x4": correspondence_ok,
            "frozen_events_unique_at_least_two_budgets": frozen_pass,
            "frozen_fraction": frozen_fraction,
            "required_fraction": MIN_FRACTION,
            "gate4_pass": gate4,
            "oracle_events_unique_at_least_two_budgets": oracle_pass,
            "oracle_fraction": oracle_pass / n,
        },
        "counts": {
            "events": n,
            "frontier_status_by_controller_budget": {
                c: {b: dict(v) for b, v in by_budget.items()}
                for c, by_budget in counts.items()},
        },
        "events": adjudicated_events,
        "required_method_handling": [
            "retain C*_t/current-feasibility supervision in addition to the "
            "derived last-safe label",
            "report UNIQUE_LAST_SAFE_MONOTONE and WITH_REENTRY separately",
            "never convert right-censored horizons into observed T_X",
            "condition T_X on controller and budget; release all four budgets",
        ],
        "non_conclusions": {
            "semantic_branch_validity": False,
            "language_evidence_closure": False,
            "validated_reveal_event_count": 0,
            "automatic_frontend_gate_pass": False,
            "training_authorized": False,
            "human_review_authorized": False,
            "frozen_spec_modified": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "gates": output["gates"], "counts": output["counts"],
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if gate4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
