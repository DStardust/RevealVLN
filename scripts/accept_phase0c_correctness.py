#!/usr/bin/env python3
"""Main-agent acceptance record for MF2-CR1 Phase-0C evidence."""

import hashlib
import json
import os


ROOT = "/mnt/daiyang/vla"
BASE = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness")
OUT = os.path.join(BASE, "PHASE0C_MAIN_ACCEPTANCE.json")
INPUTS = {
    "revision_spec": ("METHOD_FREEZE_2_CORRECTNESS_REVISION_1.md",
        "7848e58cef1e490e77185c8375fb6051394236784c7691e3a4890554eeca0d8c"),
    "identity_audit":
        ("artifacts/runtime/phase0_correctness/"
         "CANDIDATE_IDENTITY_AUDIT.json",
         "b4a815d41c830b748db18f9f8cabfd7001240870e841506456ace45ee4e4b9fb"),
    "identity_v3":
        ("artifacts/runtime/phase0_correctness/"
         "IDENTITY_V3_RERUN_SUMMARY.json",
         "cf4e5d51b1052bf789ae9747bfaf8136a9438e526b2c0206fadb0ec0afe59109"),
    "tx_audit":
        ("artifacts/runtime/phase0_correctness/TX_FEASIBILITY_AUDIT.json",
         "b24926f11f78e8ec6ecf78f18b4a48f99cffd1e041c8c06ac295f86d393e7472"),
    "highlevel_probe":
        ("artifacts/runtime/phase0_correctness/"
         "PHASE0C_ORACLE_EGOFOV_PROBE.json",
         "97f0de47610bf4f388cdf2527d702b3c248e3fdb345a05fb6ed1b81d6e566f99"),
    "lowlevel_probe":
        ("artifacts/runtime/phase0_correctness/"
         "PHASE0C_ORACLE_LOWLEVEL_PROBE.json",
         "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac"),
    "raw_cost_witness":
        ("artifacts/runtime/phase0_correctness/"
         "PHASE0C_COST_FRONTIER_WITNESS.json",
         "9b59ea9b7b9995aeb604b00587dd79a3af863ea51b16ddfe805b3f719f1a16d1"),
    "cost_adjudication":
        ("artifacts/runtime/phase0_correctness/"
         "PHASE0C_COST_FRONTIER_ADJUDICATION.json",
         "43481d408358322a826f9769e269b38115ba0cacb794d2de377aaae4b6b12551"),
    "determinism_replay":
        ("artifacts/runtime/phase0_correctness/"
         "PHASE0C_COST_FRONTIER_REPLAY.json",
         "cfa53fd23b12505283265dbf0d0021d2415ca44667b3047ab15c078b1d41013d"),
    "regression":
        ("artifacts/runtime/phase0_correctness/PHASE0C_REGRESSION.json",
         "d4cfb856fd1a5d4afd3bc758c7d0c66e5ac61e9cc3a96afb5f8eacc4d9aae4f3"),
}


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load(relative):
    with open(os.path.join(ROOT, relative)) as fh:
        return json.load(fh)


def main():
    verified = {}
    for name, (relative, expected) in INPUTS.items():
        observed = sha256_file(os.path.join(ROOT, relative))
        if observed != expected:
            raise SystemExit("input SHA drift: " + name)
        verified[name] = {"path": relative, "sha256": observed}

    identity = load(INPUTS["identity_v3"][0])
    lowlevel = load(INPUTS["lowlevel_probe"][0])
    raw = load(INPUTS["raw_cost_witness"][0])
    adjudication = load(INPUTS["cost_adjudication"][0])
    replay = load(INPUTS["determinism_replay"][0])
    regression = load(INPUTS["regression"][0])

    gates = {
        "gate1_causal_sensor_hidden_view_perturbation": {
            "status": "NOT_RUN_AUTOMATIC_FRONTEND_REQUIRED",
            "pass": False,
        },
        "gate2_oracle_event_floor": {
            "status": "PASS",
            "pass": lowlevel.get("status") == "PASS" and
                    lowlevel["counts"]["provisional_k3_events"] >= 15 and
                    lowlevel["counts"]["scenes_with_event"] >= 10,
            "events": lowlevel["counts"]["provisional_k3_events"],
            "scenes": lowlevel["counts"]["scenes_with_event"],
            "clock": "counted 0.25m MOVE / 30deg TURN prefixes",
        },
        "gate3_complete_reproducible_cost_witness": {
            "status": "PASS",
            "pass": raw["gates"]["gate3_complete_cost_evidence"] is True and
                    replay.get("status") == "PASS",
            "events_complete": raw["counts"]["complete_events"],
            "exact_replay_events": len(replay["results"]),
        },
        "gate4_budget_frontier": {
            "status": "PASS_WITH_HIGH_REENTRY_RISK",
            "pass": adjudication["gates"]["gate4_pass"] is True,
            "frozen_controller_unique_two_budget_events":
                adjudication["gates"][
                    "frozen_events_unique_at_least_two_budgets"],
            "events": adjudication["counts"]["events"],
            "fraction": adjudication["gates"]["frozen_fraction"],
            "required_fraction": adjudication["gates"][
                "required_fraction"],
            "warning": "T_X is a future-dependent last-passage label; high "
                       "safe-unsafe-safe re-entry requires separate causal "
                       "current-feasibility prediction and reporting.",
        },
        "gate5_nontrivial_timing": {
            "status": "PASS_ORACLE_GEOMETRIC",
            "pass": raw["gates"]["gate5_pass"] is True,
            "fraction": raw["gates"]["gate5_nontrivial_fraction"],
        },
        "gate6_identity_and_semantic_branch": {
            "status": "NUMERIC_PASS_SEMANTIC_NOT_RUN",
            "pass": False,
            "numeric_identity_complete":
                identity.get("status") == "ENGINEERING_PASS",
            "numeric_traces": 50,
            "semantic_ambiguity_zero": False,
            "validated_semantic_events": 0,
        },
        "gate7_boundary_regression": {
            "status": "PASS",
            "pass": regression.get("status") == "PASS",
            "checks": "%d/%d" % (regression["checks_passed"],
                                    regression["checks_total"]),
        },
    }
    if not all(item["pass"] for key, item in gates.items()
               if key not in {
                   "gate1_causal_sensor_hidden_view_perturbation",
                   "gate6_identity_and_semantic_branch"}):
        raise SystemExit("an accepted engineering gate failed")

    output = {
        "manifest": "RevealNav MF2-CR1 Phase-0C main-agent acceptance",
        "revision": "PHASE0C-CORRECTNESS-ACCEPT-1",
        "status": "PARTIAL_ENGINEERING_ACCEPTANCE",
        "overall_phase0_decision": "NO_GO",
        "training_authorized": False,
        "canonical_freeze_replacement_authorized": False,
        "method_decision": "MF2-CR1_FEASIBLE_GEOMETRIC_CORE; "
                           "AUTOMATIC_CAUSAL_FRONTEND_AND_SEMANTIC_CLOSURE_"
                           "REQUIRED",
        "scope": "Accepts a corrected low-level, budget-conditioned, "
                 "geometric cost-frontier object on the frozen 50-item "
                 "RxR-train queue. It does not accept automatic candidate "
                 "causality, semantic Reveal Events, training, or a paper "
                 "benchmark claim.",
        "inputs": verified,
        "gates": gates,
        "accepted_corrections": [
            "Use counted low-level MOVE/TURN prefixes instead of sparse ETP "
            "high-level graph decisions.",
            "Define resource-conditioned C_direct/C_save/Cstar frontiers.",
            "Treat T_X as an offline last-passage maximum, not an online "
            "first-passage stopping time.",
            "Predict causal current feasibility separately and retain all "
            "re-entry/right-censored/never-feasible statuses.",
            "Preserve upstream nearest-ID behavior with complete v3 numeric "
            "evidence; do not equate it with semantic branch identity.",
        ],
        "blocking_next_gates": [
            "Implement one causal 63-degree view buffer shared by the frozen "
            "waypoint frontend and policy; hidden-view perturbation must have "
            "zero pre-acquisition effect.",
            "Build semantic branch tracks and adjudicate the fixed queue "
            "without fabricating review; admitted events require zero "
            "semantic ambiguity.",
            "Only after both pass, version and review a replacement canonical "
            "method freeze before feature generation or training.",
        ],
        "explicit_non_conclusions": {
            "validated_semantic_reveal_events": 0,
            "automatic_frontend_pass": False,
            "training_allowed": False,
            "benchmark_established": False,
            "cvpr_competitiveness_established": False,
            "val_unseen_or_test_used": False,
            "checkpoint_loaded_in_this_batch": False,
            "human_review_performed": False,
            "frozen_spec_modified": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"],
        "overall_phase0_decision": output["overall_phase0_decision"],
        "gates": {key: value["status"] for key, value in gates.items()},
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
