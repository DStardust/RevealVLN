"""Deterministically prepare, but never execute, the four-trial physical plan."""
from __future__ import annotations

from typing import Any

from common import digest_value


def _yaw(value: Any) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            return int(value.removeprefix("yaw_"))
        except ValueError:
            return None
    return None


def _trial(
    trial_id: str,
    stratum: str,
    proposal: dict[str, Any],
    yaw_row: dict[str, Any],
    house: dict[str, Any],
    proposal_object_sha256: str,
    history_row: dict[str, Any] | None,
) -> dict[str, Any]:
    indices = proposal["role_indices"]
    role_context = {
        role: {
            "index": index,
            "spec": house["roles"][index]["spec"],
            "eligible": house["roles"][index]["eligible"],
        }
        for role, index in zip(("anchor_A", "anchor_B", "terminal"), indices)
    }
    return {
        "trial_id": trial_id,
        "stratum": stratum,
        "source_run": "v16_repair_rqf_001",
        "proposal_id": proposal["id"],
        "base_proposal_id": proposal.get("base_proposal_id"),
        "hub": proposal["hub"],
        "start_position": proposal["start"],
        "yaw": yaw_row["yaw"],
        "role_indices": indices,
        "roles_and_eligible_sha256": digest_value(role_context),
        "targets_and_order_sha256": digest_value(proposal["targets"]),
        "source_proposal_object_sha256": proposal_object_sha256,
        "source_probe_object_sha256": yaw_row["input_object_sha256"],
        "source_history_object_sha256": history_row.get("input_object_sha256") if history_row else None,
        "selection_reason": (
            "complete frozen probe fails the corrected anchor-only neutral gate"
            if stratum == "A_INITIAL_NEUTRAL_REJECTION"
            else "complete frozen probe is neutral and saved history has another-anchor SEE2 evidence"
        ),
        "expected_stage_checks": (
            ["initial_probe", "first_anchor_trigger"]
            if stratum == "A_INITIAL_NEUTRAL_REJECTION"
            else ["initial_probe", "discovery_history", "first_other_anchor_trigger"]
        ),
        "planned_attempts": 1,
    }


def prepare_local_validation_plan(
    neutral: dict[str, Any],
    isolation: dict[str, Any],
    proposals: list[dict[str, Any]],
    house: dict[str, Any],
    proposal_object_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    proposal_by_id = {row["id"]: row for row in proposals}
    neutral_rows = neutral.get("yaw_records", [])
    candidates_a = sorted(
        (
            row
            for row in neutral_rows
            if row.get("neutral_gate_result") == "INVALID_ANCHOR_PRESENT"
            and row.get("probe_gate_reached") is True
        ),
        key=lambda row: (proposal_by_id[row["proposal_id"]]["hub"], row["proposal_id"], row["yaw"]),
    )
    selected_a: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates_a:
        if row["proposal_id"] in seen:
            continue
        selected_a.append(row)
        seen.add(row["proposal_id"])
        if len(selected_a) == 2:
            break

    valid_slots = {
        (row["proposal_id"], row["yaw"]): row
        for row in neutral_rows
        if row.get("neutral_gate_result") == "VALID_NEUTRAL_START" and row.get("probe_gate_reached") is True
    }
    candidates_b: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for history in isolation.get("rows", []):
        yaw = _yaw(history.get("yaw"))
        key = (history.get("proposal_id"), yaw)
        if (
            history.get("record_kind") == "HISTORY_DISCOVERY"
            and history.get("other_anchor_seen") is True
            and history.get("evidence_status") == "READ"
            and key in valid_slots
        ):
            candidates_b.append((valid_slots[key], history))
    candidates_b.sort(
        key=lambda pair: (
            proposal_by_id[pair[0]["proposal_id"]]["hub"],
            pair[0]["proposal_id"],
            pair[0]["yaw"],
            pair[1].get("history_id") or "",
        )
    )
    selected_b: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for yaw_row, history in candidates_b:
        proposal_id = yaw_row["proposal_id"]
        if proposal_id in seen:
            continue
        selected_b.append((yaw_row, history))
        seen.add(proposal_id)
        if len(selected_b) == 2:
            break

    trials: list[dict[str, Any]] = []
    for index, row in enumerate(selected_a, 1):
        trials.append(
            _trial(
                f"rqf_stage_trace_A{index}",
                "A_INITIAL_NEUTRAL_REJECTION",
                proposal_by_id[row["proposal_id"]],
                row,
                house,
                proposal_object_sha256,
                None,
            )
        )
    for index, (row, history) in enumerate(selected_b, 1):
        trials.append(
            _trial(
                f"rqf_stage_trace_B{index}",
                "B_HISTORY_ISOLATION_CONTAMINATION",
                proposal_by_id[row["proposal_id"]],
                row,
                house,
                proposal_object_sha256,
                history,
            )
        )
    complete = len(selected_a) == 2 and len(selected_b) == 2
    manifest = {
        "proposed_version": "v16_rqf_stage_trace_probe_v2_1",
        "proposed_run_id": "v16_repair_rqf_003_diagnostic_001",
        "authorization_status": "PENDING_EXPLICIT_APPROVAL",
        "status": "READY_FOR_REVIEW" if complete else "PLAN_INCOMPLETE_INSUFFICIENT_ELIGIBLE_CANDIDATES",
        "selection_denominators": {
            "stratum_A_eligible_rows": len(candidates_a),
            "stratum_A_distinct_proposals_selected": len(selected_a),
            "stratum_B_eligible_history_rows": len(candidates_b),
            "stratum_B_distinct_proposals_selected": len(selected_b),
        },
        "trials": trials,
        "trial_count": len(trials),
        "selection_order": "hub ascending, proposal_id lexicographic, yaw ascending; distinct proposals; no model scores",
    }
    plan = {
        "status": "PENDING_EXPLICIT_APPROVAL" if complete else "PLAN_INCOMPLETE_INSUFFICIENT_ELIGIBLE_CANDIDATES",
        "max_unique_proposals": 4,
        "max_trial_starts": 4,
        "yaws_per_proposal": 1,
        "attempts_per_trial": 1,
        "automatic_retries": 0,
        "max_initial_probe_actions_per_trial": 2,
        "max_discovery_histories_per_trial": 4,
        "max_decisions_per_discovery_history": 240,
        "max_episode_resets_upper_bound": 20,
        "max_action_calls_upper_bound": 3848,
        "per_trial_wall_time_limit_seconds": 900,
        "whole_job_wall_time_limit_seconds": 3600,
        "execute_continuation_suffixes": False,
        "publish_formal_family": False,
        "new_family_admissions": 0,
        "g1_allowed": False,
        "qwen_allowed": False,
        "training_allowed": False,
        "navigation_evaluation_allowed": False,
        "automatic_promotion_allowed": False,
        "only_change": "side-channel phase boundary logging without extra observe/reset/step calls",
        "trial_execution_contract": "run the original initial L,R probe; if it passes, run H_A,H_B,H_A_R,H_B_R in that order and stop the trial at the first physical rejection; do not run C0/C_A/C_B",
        "stage_logger_contract": {
            "required_per_action_fields": [
                "trial_id",
                "episode_id",
                "history_id",
                "action_index",
                "action",
                "phase_before_action",
                "observation_index_after_action",
                "phase_of_observation",
                "source_function",
                "role_being_sought",
                "witness_target_index",
                "collision_result",
                "trace_prefix_hash"
            ],
            "locked_phases": [
                "initial_probe",
                "outbound_navigation",
                "outbound_heading_adjustment",
                "witness_scan",
                "perturbation",
                "return_navigation",
                "return_heading_adjustment",
                "public_tail"
            ],
            "cross_phase_see2": "record both supporting frame phases, cross_phase_event=true, and assign first_completion_stage to the second frame",
            "side_effect_rule": "logger must not call observe, reset, step, follower, renderer, or alter random state"
        },
        "stop_conditions": [
            "any registered limit is reached",
            "program error, evidence corruption, or identity mismatch stops the whole job",
            "physical rejection ends only its registered trial",
            "no fifth trial, yaw substitution, house expansion, retry, suffix, or promotion",
        ],
    }
    patch_plan = {
        "version": "v16_rqf_offline_errata_v2_1",
        "status": "PLAN_ONLY_NO_PHYSICAL_EXECUTION",
        "shared_engineering_changes": [
            "anchor-only neutral predicate with evidence-completeness gate",
            "strict four-history/twelve-trace validation",
            "atomic no-replace publication and read-only recovery",
            "typed exception routing and bounded runner",
        ],
        "next_node_change": "record pre-registered action/observation phase boundaries without changing action selection",
        "future_entrypoint_required": "independent approved diagnostic entrypoint; current offline driver cannot execute it",
        "manifest_status": manifest["status"],
        "authorization_status": "PENDING_EXPLICIT_APPROVAL",
    }
    return manifest, plan, patch_plan
