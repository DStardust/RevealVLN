#!/usr/bin/env python3
"""Versioned main-agent acceptance for MF2-CR2 Phase-0C.

The result is intentionally a machine-evidence acceptance with a human-review
blocker, not an overall Phase-0 GO record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/runtime/phase0_correctness"
OUT = BASE / "PHASE0C_MAIN_ACCEPTANCE_V2.json"
INPUTS = {
    "previous_acceptance_preserved": (
        "artifacts/runtime/phase0_correctness/PHASE0C_MAIN_ACCEPTANCE.json",
        "ca98f155b8d066a756c809fb8dc548542d1ce92b1347cde8fa28a00bf462b015"),
    "correctness_revision_1": (
        "METHOD_FREEZE_2_CORRECTNESS_REVISION_1.md",
        "de1cc32a890153d9962047841ff2dbc469c130f4cfb68de53c4ba5f9fb90262b"),
    "correctness_revision_2": (
        "METHOD_FREEZE_2_CORRECTNESS_REVISION_2.md",
        "3026e4696803ec6e7278831cb1f781a93a588f9fe09db73ddd67869e7c6e314b"),
    "identity_v3": (
        "artifacts/runtime/phase0_correctness/IDENTITY_V3_RERUN_SUMMARY.json",
        "cf4e5d51b1052bf789ae9747bfaf8136a9438e526b2c0206fadb0ec0afe59109"),
    "lowlevel_oracle": (
        "artifacts/runtime/phase0_correctness/PHASE0C_ORACLE_LOWLEVEL_PROBE.json",
        "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac"),
    "raw_cost_witness": (
        "artifacts/runtime/phase0_correctness/PHASE0C_COST_FRONTIER_WITNESS.json",
        "9b59ea9b7b9995aeb604b00587dd79a3af863ea51b16ddfe805b3f719f1a16d1"),
    "cost_adjudication": (
        "artifacts/runtime/phase0_correctness/PHASE0C_COST_FRONTIER_ADJUDICATION.json",
        "43481d408358322a826f9769e269b38115ba0cacb794d2de377aaae4b6b12551"),
    "cost_replay": (
        "artifacts/runtime/phase0_correctness/PHASE0C_COST_FRONTIER_REPLAY.json",
        "cfa53fd23b12505283265dbf0d0021d2415ca44667b3047ab15c078b1d41013d"),
    "causal_contract": (
        "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_CONTRACT_GATE.json",
        "5a64238c1cf66dcf5aedb01b2ab63575575164a4d2eccaea92407a1d2bbd75d8"),
    "causal_model": (
        "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_MODEL_INTEGRATION.json",
        "3b5cf7d50cc5fab8a1e241f6a1e6416144866f74d059471490677a3901492ab9"),
    "physical_inspect": (
        "artifacts/runtime/phase0_correctness/PHYSICAL_INSPECT_ACQUISITION_GATE.json",
        "d76362431b05a962b0569915f82d45db1fe05e014afe878c34d3f8c5e8f0d93a"),
    "causal_multiview": (
        "artifacts/runtime/phase0_correctness/CAUSAL_FRONTEND_MULTIVIEW_INTEGRATION.json",
        "33354897e7db3fe0b5e88e727fbb817a5c279c3db5cdaba8bc4939f90cfd394b"),
    "oracle_semantic": (
        "artifacts/runtime/phase0_correctness/ORACLE_SEMANTIC_BRANCH_TRACK_AUDIT.json",
        "e4b570dc9cdbe317d28b57507f1f74b9a16f92c8350810beb6b0f4dacd9df6a4"),
    "automatic_raw_preserved": (
        "artifacts/runtime/phase0_correctness/AUTOMATIC_SEMANTIC_CANDIDATE_GATE.json",
        "13797692e69847392b572f17f0559f36b685ec84b10051fc14c9f26c13ad2f7b"),
    "automatic_multiplicity_adjudication": (
        "artifacts/runtime/phase0_correctness/AUTOMATIC_SEMANTIC_MULTIPLICITY_ADJUDICATION.json",
        "e2dfba0b25f7df3cfcc4082567d95d897860595a1b6e0bf46bbe81846f696d3a"),
    "pending_language_packet": (
        "artifacts/phase0/phase0c_language_review_35/PHASE0C_LANGUAGE_REVIEW_35.json",
        "b97f546d454d09a57c21153adc55bc02c30a4c694b07cd925091fac0b07a6784"),
    "final_regression": (
        "artifacts/runtime/phase0_correctness/PHASE0C_FINAL_REGRESSION.json",
        "121e794631080face2fc22b249705bada3a493e1d917df16c8b5a726a22b88e1"),
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def load(name: str):
    return json.loads((ROOT / INPUTS[name][0]).read_text())


def all_true(mapping) -> bool:
    return isinstance(mapping, dict) and mapping and all(
        value is True for value in mapping.values())


def main() -> int:
    verified = {}
    for name, (relative, expected) in INPUTS.items():
        observed = sha256_file(ROOT / relative)
        if observed != expected:
            raise SystemExit("input SHA drift: %s: %s" % (name, observed))
        verified[name] = {"path": relative, "sha256": observed}

    identity = load("identity_v3")
    lowlevel = load("lowlevel_oracle")
    raw_cost = load("raw_cost_witness")
    cost = load("cost_adjudication")
    replay = load("cost_replay")
    causal_contract = load("causal_contract")
    causal_model = load("causal_model")
    physical = load("physical_inspect")
    multiview = load("causal_multiview")
    oracle_semantic = load("oracle_semantic")
    automatic_raw = load("automatic_raw_preserved")
    automatic = load("automatic_multiplicity_adjudication")
    packet = load("pending_language_packet")
    regression = load("final_regression")

    causal_pass = (
        causal_contract.get("status") == "PASS"
        and all_true(causal_contract.get("checks"))
        and causal_model.get("status") == "PASS"
        and all_true(causal_model.get("checks"))
        and physical.get("status") == "PASS"
        and all_true(physical.get("checks"))
        and multiview.get("status") == "PASS"
        and all_true(multiview.get("checks")))
    rows = packet.get("rows", [])
    human_fields = packet.get("human_fields", [])
    packet_pending = (
        packet.get("status") == "PASS_PENDING_HUMAN_REVIEW"
        and len(rows) == 35
        and len({row.get("scene_id") for row in rows}) == 22
        and all(row.get("reviewed") is False
                and all(row.get(field) is None for field in human_fields)
                for row in rows))

    gates = {
        "gate1_causal_sensor_and_counted_inspect": {
            "status": "PASS_ENGINEERING",
            "pass": causal_pass,
            "contract_checks": 6,
            "real_single_view_model_checks": 8,
            "physical_inspect_checks": 9,
            "real_post_inspect_multiview_checks": 8,
            "scope": "shared raw-view mask, waypoint/panorama/policy chain, "
                     "physical 30-degree turn and cache reset after move",
        },
        "gate2_oracle_event_floor": {
            "status": "PASS",
            "pass": lowlevel.get("status") == "PASS"
                    and lowlevel["counts"]["provisional_k3_events"] == 104
                    and lowlevel["counts"]["scenes_with_event"] == 32,
            "events": 104,
            "scenes": 32,
        },
        "gate3_complete_reproducible_cost_witness": {
            "status": "PASS_AFTER_STATUS_ADJUDICATION",
            "pass": raw_cost["gates"]["gate3_complete_cost_evidence"] is True
                    and raw_cost["counts"]["complete_events"] == 104
                    and replay.get("status") == "PASS",
            "complete_events": 104,
            "exact_fresh_process_replays": 3,
            "raw_output_status_preserved": raw_cost.get("status"),
        },
        "gate4_budget_frontier": {
            "status": "PASS_WITH_HIGH_REENTRY_RISK",
            "pass": cost["gates"]["gate4_pass"] is True,
            "frozen_unique_at_least_two_budgets":
                cost["gates"]["frozen_events_unique_at_least_two_budgets"],
            "events": cost["counts"]["events"],
            "fraction": cost["gates"]["frozen_fraction"],
            "required_fraction": cost["gates"]["required_fraction"],
            "warning": "safe-unsafe-safe re-entry is frequent; T_X is an "
                       "offline last-passage label and online policy must use "
                       "causal current-feasibility/cost prediction",
        },
        "gate5_nontrivial_timing": {
            "status": "PASS_ORACLE_GEOMETRIC",
            "pass": raw_cost["gates"]["gate5_pass"] is True,
            "fraction": raw_cost["gates"]["gate5_nontrivial_fraction"],
            "required_fraction": raw_cost["gates"]["gate5_required_fraction"],
        },
        "gate6_identity_semantic_and_language": {
            "status": "MACHINE_SUBGATES_PASS_HUMAN_LANGUAGE_PENDING",
            "pass": False,
            "numeric_identity_v3": identity.get("status") == "ENGINEERING_PASS",
            "numeric_traces": identity["counts"][
                "full_50_traces_after_reusing_16_v1_ok"],
            "oracle_machine_geometric_events": oracle_semantic["counts"][
                "machine_geometric_admitted"],
            "oracle_machine_geometric_scenes": oracle_semantic["counts"][
                "admitted_scenes"],
            "automatic_raw_failure_preserved":
                automatic_raw.get("status") == "FAIL",
            "automatic_tracks_after_pre_registered_ontology_correction":
                automatic["counts"]["tracked_k3"],
            "automatic_track_scenes": automatic["counts"]["tracked_scenes"],
            "fixed_review_intersection_events": len(rows),
            "fixed_review_intersection_scenes": len(
                {row.get("scene_id") for row in rows}),
            "pending_packet_integrity": packet_pending,
            "human_reviewed_events": 0,
            "human_review_performed": False,
            "full_gate6_pass": False,
        },
        "gate7_boundary_and_regression": {
            "status": "PASS",
            "pass": regression.get("status") == "PASS"
                    and regression.get("checks_passed") == 51
                    and regression.get("checks_total") == 51,
            "checks": "51/51",
        },
    }
    machine_gate_names = [name for name in gates
                          if name != "gate6_identity_semantic_and_language"]
    if not all(gates[name]["pass"] for name in machine_gate_names):
        raise SystemExit("one or more machine engineering gates failed")
    if not packet_pending:
        raise SystemExit("pending review packet integrity failed")

    output = {
        "manifest": "RevealNav MF2-CR2 Phase-0C main-agent acceptance v2",
        "revision": "PHASE0C-CORRECTNESS-ACCEPT-2",
        "status": "MACHINE_PHASE0C_ACCEPTANCE_PENDING_HUMAN_REVIEW",
        "overall_phase0_decision": "NO_GO_PENDING_HUMAN_LANGUAGE_REVIEW",
        "training_authorized": False,
        "feature_generation_authorized": False,
        "canonical_freeze_replacement_authorized": False,
        "paper_benchmark_claim_authorized": False,
        "method_decision": "CORRECTED_MACHINE_FEASIBILITY_SUPPORTED; "
                           "HUMAN_LANGUAGE_BRANCH_CLOSURE_REQUIRED",
        "inputs": verified,
        "gates": gates,
        "accepted_machine_findings": [
            "The shared causal 63-degree adapter prevents hidden raw views, "
            "candidate features, panorama tokens, logits and actions from "
            "affecting the model before physical acquisition.",
            "A counted 30-degree INSPECT turn adds a second physical view; a "
            "subsequent move resets the pose-local view buffer.",
            "The fixed Oracle queue contains 104 low-level provisional events "
            "across 32 scenes with complete controller-cost evidence.",
            "After correcting the future-dependent label to a last-passage "
            "object, 94/104 events pass the two-budget frontier gate, while "
            "high re-entry frequency remains a mandatory reported risk.",
            "The fixed automatic candidate frontend supports 38 K=3 semantic "
            "tracks across 23 scenes; 27 retain multiple numeric proposals "
            "inside one directed exit region rather than collapsing them.",
            "The deterministic cost-and-automatic intersection contains 35 "
            "private review candidates across 22 scenes.",
        ],
        "pre_registered_human_acceptance": {
            "packet_rows_all_must_be_reviewed": 35,
            "independent_blinded_reviewers_per_row": 2,
            "third_reviewer_for_disagreement_or_uncertainty": True,
            "minimum_accepted_unique_events": 15,
            "minimum_accepted_scenes": 10,
            "maximum_unresolved_semantic_ambiguity": 0,
            "retain_both_original_tables_and_adjudication": True,
            "report_inter_rater_agreement": True,
            "favorable_subset_only_review_forbidden": True,
        },
        "remaining_scientific_risks": [
            "Gate 6 has zero human-validated language-dependent Reveal Events.",
            "The automatic frontend retains 38/90 machine-geometric events "
            "before the stricter cost intersection, so coverage/selectivity "
            "must be reported and cannot be hidden.",
            "Re-entry is common at every fixed budget; a model must predict "
            "causal current feasibility/cost rather than claim online access "
            "to the future-dependent last-passage label.",
            "The INSPECT state machine is engineered but no learned INSPECT "
            "policy or training result exists yet.",
        ],
        "next_gate": {
            "name": "MF2-CR2 dual-review language/branch acceptance",
            "input_packet": verified["pending_language_packet"],
            "decision_before_completion": "NO_GO",
        },
        "explicit_non_conclusions": {
            "human_validated_reveal_events": 0,
            "full_phase0c_pass": False,
            "training_allowed": False,
            "feature_generation_allowed": False,
            "benchmark_established": False,
            "cvpr_competitiveness_established": False,
            "cvpr_acceptance_guaranteed": False,
            "val_unseen_or_test_used": False,
            "canonical_frozen_spec_modified": False,
            "reserve_released": False,
        },
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"],
        "overall_phase0_decision": output["overall_phase0_decision"],
        "gate_statuses": {name: value["status"]
                          for name, value in gates.items()},
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
