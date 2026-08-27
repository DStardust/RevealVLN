#!/usr/bin/env python3
"""Aligned reversible-disagreement controller for final V5.12."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_continuous_controller_worker_v5_2 as v52  # noqa: E402
import r2r_native_control_opp_worker_v5_10 as v510  # noqa: E402
from revealnav_mf2r3 import OptionStatus  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    BranchMacroAction,
    StateConditionedReturnExecutor,
)


V54 = v510.v59.v58.v57.v56.v54


class AlignedNativeControlFullOPPActionController(
    v510.NativeControlFullOPPActionController
):
    """Test a disagreeing option reversibly while retaining ETP's action.

    Every final control receives its own equally available temporal history.
    A learned disagreement never becomes a direct irreversible commit.  It
    can only run as a one-step checkpointed excursion.  If the post-excursion
    heads reject that option and the online return succeeds, the retained ETP
    action is consumed at the restored checkpoint.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.global_rows = []
        self.temporal_prefixes_cached = 0
        self.aligned_decision_prefixes = 0
        self.reversible_alternative_trials = 0
        self.reversible_shadow_trials = 0
        self.direct_override_suppressions = 0
        self.retained_native_commits = 0
        self.retained_native_unavailable = 0
        self.final_step_suppressions = 0
        self.ree_closed_return_vetoes = 0
        self.trial_native_fallback: str | None = None
        self.trial_preservation_gain: float | None = None
        self.pending_native_fallback: dict | None = None
        self.checkpoint_graph_snapshot = None
        self.checkpoint_graph_signature: dict | None = None
        self.topology_snapshots = 0
        self.topology_restores = 0

    def _aligned_value(self, current, persistent, native_branch):
        """Evaluate 2-4 controls with identical causal availability masks."""
        if len(persistent) < 2:
            return None, ()
        if native_branch is None:
            self.stop_suppressions += 1
            self.record(
                "opp_base_action_safety_suppression",
                reason="native_ETP_STOP_must_not_be_overridden",
                persistent_branch_count=len(persistent),
            )
            return None, ()
        if native_branch not in self.global_current:
            self.native_outside_candidate_suppressions += 1
            self.record(
                "opp_base_action_safety_suppression",
                reason="native_ETP_action_is_not_an_unvisited_candidate",
                native_base_branch=native_branch,
                persistent_branch_count=len(persistent),
            )
            return None, ()
        controls = tuple(dict.fromkeys((*persistent, native_branch)))
        if len(controls) > 4:
            self.candidate_width_suppressions += 1
            self.record(
                "opp_base_action_safety_suppression",
                reason="candidate_plus_native_width_outside_training_support",
                native_base_branch=native_branch,
                persistent_branch_count=len(persistent),
                scored_candidate_count=len(controls),
            )
            return None, ()

        scoring = {
            branch_id: self.global_current[branch_id]
            for branch_id in controls
        }
        self.rows = [
            (
                history,
                {
                    branch_id: candidates[branch_id]
                    for branch_id in controls
                    if branch_id in candidates
                },
            )
            for history, candidates in self.global_rows[:-1]
        ]
        self.safe_decision_prefixes += 1
        self.native_control_comparisons += 1
        self.aligned_decision_prefixes += 1
        self._native_control_branch = native_branch
        try:
            value = self._evaluate(scoring, controls)
        finally:
            self._native_control_branch = None
        return value, controls

    @staticmethod
    def _backup_branch(value, controls, native_branch):
        macro = value["macro"]
        if macro.branch_id in controls and macro.branch_id != native_branch:
            return macro.branch_id, "macro_checkpoint_branch"
        ranked = sorted(
            (
                (float(value["probabilities"][index]), branch_id)
                for index, branch_id in enumerate(controls)
                if branch_id != native_branch
            ),
            key=lambda row: (-row[0], row[1]),
        )
        return (ranked[0][1], "highest_target_probability_alternative") if ranked else (None, None)

    def _record_initial(self, value, native_branch, selected) -> None:
        macro = value["macro"]
        self.record(
            "opp_initial_decision",
            opp_action=value["action"],
            opp_reason=value["reason"],
            selected_branch=selected,
            native_base_branch=native_branch,
            action_differs_from_base=(
                selected is not None and selected != native_branch
            ),
            macro_action=macro.action.value,
            macro_branch=macro.branch_id,
            preservation_gain=(
                None if macro.preservation_gain is None
                else round(float(macro.preservation_gain), 8)
            ),
            **{
                key: round(number, 8)
                for key, number in value["belief"].items()
            },
        )

    def _consume_pending_native(self):
        pending = self.pending_native_fallback
        if pending is None:
            return None, False
        checkpoint_id = pilot._CURRENT_IDS[0]
        native = pending["branch_id"]
        if pending.get("topology_restored") is not True:
            raise RuntimeError("native fallback reached before topology restoration")
        if self.step >= int(pilot._TRAINER.max_len) - 1:
            self.record(
                "retained_native_cancelled",
                checkpoint_id=pending["checkpoint_id"],
                observed_checkpoint_id=checkpoint_id,
                branch_id=native,
                reason="ETP_forces_STOP_at_episode_horizon",
                fail_closed_to_native=True,
            )
            self.pending_native_fallback = None
            self.checkpoint_graph_signature = None
            self.global_rows.clear()
            self._reset_search()
            return None, True
        # ETP-R1 assigns a fresh online node id after a successful graph
        # return, even when the physical checkpoint error is exactly zero.
        # The fallback is armed only after that metric-space check in
        # ``complete_pending_return``; candidate presence is therefore the
        # stable executable identity gate here, not the transient node id.
        available = native in self.global_current
        if not available:
            self.retained_native_unavailable += 1
            self.record(
                "retained_native_cancelled",
                checkpoint_id=pending["checkpoint_id"],
                observed_checkpoint_id=checkpoint_id,
                branch_id=native,
                reason="retained_native_unavailable_after_verified_return",
                fail_closed_to_native=True,
            )
            self.pending_native_fallback = None
            self.checkpoint_graph_signature = None
            self.global_rows.clear()
            self._reset_search()
            return None, True
        if not self.ledger.authorize_branch(
            checkpoint_id, native, pending["preservation_gain"]
        ):
            raise RuntimeError("retained native action is no longer ledger-authorized")
        self.ledger.resolve_continue(checkpoint_id, native)
        self.retained_native_commits += 1
        self.record(
            "retained_native_committed",
            checkpoint_id=checkpoint_id,
            branch_id=native,
            rejected_trial_branch=pending["trial_branch"],
            preservation_gain=round(pending["preservation_gain"], 8),
            checkpoint_native_action_restored=True,
            causal_precondition="successful_online_return_to_checkpoint",
        )
        self.pending_native_fallback = None
        self.checkpoint_graph_signature = None
        self.global_rows.clear()
        self._reset_search()
        return native, True

    def _start_alternative_trial(
        self, current, controls, native_branch, backup, backup_source, value,
    ):
        macro = value["macro"]
        preservation_gain = float(macro.preservation_gain)
        checkpoint_id = pilot._CURRENT_IDS[0]
        self.reversible_alternative_trials += 1
        if self.mode == "shadow":
            self.reversible_shadow_trials += 1
            self.record(
                "reversible_alternative_trial_shadow",
                checkpoint_id=checkpoint_id,
                trial_branch=backup,
                retained_native_branch=native_branch,
                trial_source=backup_source,
                preservation_gain=round(preservation_gain, 8),
                shadow_only_not_executed=True,
            )
            self.global_rows.clear()
            self._reset_search()
            return None

        self.ledger.register(checkpoint_id, controls)
        if not self.ledger.authorize_branch(checkpoint_id, backup, preservation_gain):
            self.direct_override_suppressions += 1
            self.record(
                "reversible_alternative_trial_suppressed",
                checkpoint_id=checkpoint_id,
                trial_branch=backup,
                retained_native_branch=native_branch,
                reason="alternative_branch_already_resolved_at_checkpoint",
                fail_closed_to_native=True,
            )
            self.global_rows.clear()
            self._reset_search()
            return None

        self.phase = "outbound_in_flight"
        self.pre_histories = [row[0].detach() for row in self.rows]
        self.selected_embedding = current[backup].detach()
        self.checkpoint_embedding = self.latest_history.detach()
        self.selected_branch = backup
        self.checkpoint_id = checkpoint_id
        graph = pilot._TRAINER.gmaps[0]
        self.checkpoint_graph_snapshot = copy.deepcopy(graph)
        self.checkpoint_graph_signature = self._graph_signature(graph)
        self.topology_snapshots += 1
        self.checkpoint_position = np.asarray(
            graph.node_pos[checkpoint_id], dtype=float
        ).copy()
        self.executor = StateConditionedReturnExecutor(
            checkpoint_id, "ETP-R1:frozen-control", controls
        )
        self.executor.start_excursion(backup)
        self.checkpoint_candidates = {
            branch_id: current[branch_id].detach() for branch_id in controls
        }
        self.trial_native_fallback = native_branch
        self.trial_preservation_gain = preservation_gain
        self.checkpointed_excursions += 1
        self.record(
            "reversible_alternative_trial_created",
            checkpoint_id=checkpoint_id,
            trial_branch=backup,
            retained_native_branch=native_branch,
            trial_source=backup_source,
            preservation_gain=round(preservation_gain, 8),
            direct_irreversible_commit=False,
        )
        self.global_rows.clear()
        return backup

    @staticmethod
    def _graph_signature(graph) -> dict:
        return {
            "nodes": tuple(sorted(graph.node_pos)),
            "ghosts": tuple(sorted(graph.ghost_pos)),
            "edges": int(graph.graph_nx.number_of_edges()),
            "ghost_counter": int(graph.ghost_cnt),
        }

    def restore_checkpoint_topology(self, trainer, cur_vp) -> None:
        """Restore the policy graph before its next navigation forward."""
        if self.pending_native_fallback is None:
            return
        if self.checkpoint_graph_snapshot is None:
            raise RuntimeError("verified return lacks checkpoint graph snapshot")
        if len(trainer.gmaps) != 1 or len(cur_vp) != 1:
            raise RuntimeError("V5.12 topology restoration requires one environment")
        trainer.gmaps[0] = self.checkpoint_graph_snapshot
        checkpoint_id = self.pending_native_fallback["checkpoint_id"]
        cur_vp[0] = checkpoint_id
        restored = self._graph_signature(trainer.gmaps[0])
        if restored != self.checkpoint_graph_signature:
            raise RuntimeError("restored checkpoint graph signature drift")
        self.checkpoint_graph_snapshot = None
        self.topology_restores += 1
        self.pending_native_fallback["topology_restored"] = True
        self.record(
            "checkpoint_topology_restored",
            checkpoint_id=checkpoint_id,
            graph_nodes=len(restored["nodes"]),
            graph_ghosts=len(restored["ghosts"]),
            graph_edges=restored["edges"],
            transient_current_id_rewritten=True,
        )

    def _initial_decision(self, current, persistent, native_branch):
        backup, consumed = self._consume_pending_native()
        if consumed:
            return backup

        self.global_rows.append((
            self.latest_history.detach(), dict(self.global_current)
        ))
        self.temporal_prefixes_cached += 1
        value, controls = self._aligned_value(
            current, persistent, native_branch
        )
        if value is None:
            return None

        action = value["action"]
        macro = value["macro"]
        selected = (
            value["commit_branch"] if action == "commit"
            else macro.branch_id if action == "explore" else None
        )
        self._record_initial(value, native_branch, selected)
        if action == "inspect":
            self.inspect_delegations += 1
            return None
        if action == "follow":
            self.follow_delegations += 1
            return None
        if action == "unresolved":
            self.unresolved_decisions += 1
            return None
        if action not in ("commit", "explore"):
            raise RuntimeError("unknown frozen OPP action")

        if action == "commit":
            self.commit_decisions += 1
        else:
            self.explore_decisions += 1
        backup, backup_source = self._backup_branch(
            value, controls, native_branch
        )
        valid_trial = (
            macro.action is BranchMacroAction.CHECKPOINTED_EXCURSION
            and macro.preservation_gain is not None
            and macro.preservation_gain > V54.FROZEN_CONFIG["opv_threshold"]
            and backup is not None
        )
        if valid_trial and self.step >= int(pilot._TRAINER.max_len) - 1:
            self.final_step_suppressions += 1
            self.record(
                "reversible_alternative_trial_suppressed",
                trial_branch=backup,
                retained_native_branch=native_branch,
                reason="ETP_forces_STOP_at_episode_horizon",
                fail_closed_to_native=True,
            )
            valid_trial = False
        if valid_trial:
            if action == "commit":
                # Keep the frozen event-gate count above, and separately make
                # the executed reversible macro visible to gate accounting.
                self.explore_decisions += 1
            return self._start_alternative_trial(
                {branch_id: self.global_current[branch_id] for branch_id in controls},
                controls, native_branch, backup, backup_source, value,
            )

        # Direct learned replacement was the V5.11 failure mode.  Outside a
        # valid reversible trial, always delegate to the unchanged base action.
        if selected != native_branch:
            self.direct_override_suppressions += 1
            self.record(
                "direct_override_suppressed",
                proposed_branch=selected,
                native_branch=native_branch,
                reason="no_valid_reversible_checkpoint_trial",
                fail_closed_to_native=True,
            )
        self.global_rows.clear()
        self._reset_search()
        return None

    def _post_decision(self, current) -> None:
        # The post head was trained on reached candidate excursions, not on
        # unchanged native actions.  Preserve that semantic contract and use
        # the full OPP closure gate before committing the trial branch.
        V54.FullOPPContinuousController._post_decision(self, current)
        if self.phase == "seeking_excursion":
            self.effective_commit_interventions += 1
            self.trial_native_fallback = None
            self.trial_preservation_gain = None
            self.checkpoint_graph_snapshot = None
            self.checkpoint_graph_signature = None

    def complete_pending_return(self) -> None:
        checkpoint_id = self.checkpoint_id
        trial_branch = self.selected_branch
        native = self.trial_native_fallback
        preservation_gain = self.trial_preservation_gain
        v52.ContinuousController.complete_pending_return(self)
        success = bool(self.return_intervention_success)
        if self.ledger.status(checkpoint_id, trial_branch) is OptionStatus.ACTIVE:
            if success:
                self.ledger.resolve_return(checkpoint_id, trial_branch)
            else:
                self.ledger.resolve_continue(checkpoint_id, trial_branch)
        if success and native is not None and preservation_gain is not None:
            self.pending_native_fallback = {
                "checkpoint_id": checkpoint_id,
                "trial_branch": trial_branch,
                "branch_id": native,
                "preservation_gain": preservation_gain,
            }
            self.record(
                "retained_native_armed",
                checkpoint_id=checkpoint_id,
                rejected_trial_branch=trial_branch,
                branch_id=native,
                return_verified=True,
            )
        else:
            self.pending_native_fallback = None
            self.checkpoint_graph_snapshot = None
            self.checkpoint_graph_signature = None
        self.trial_native_fallback = None
        self.trial_preservation_gain = None
        self.global_rows.clear()

    def finalize_episode(self) -> None:
        super().finalize_episode()
        if self.pending_native_fallback is not None:
            self.record(
                "terminal_retained_native_not_executed",
                branch_id=self.pending_native_fallback["branch_id"],
                fail_closed=True,
            )

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value.update({
            "temporal_prefixes_cached": self.temporal_prefixes_cached,
            "aligned_decision_prefixes": self.aligned_decision_prefixes,
            "reversible_alternative_trials": self.reversible_alternative_trials,
            "reversible_shadow_trials": self.reversible_shadow_trials,
            "direct_override_suppressions": self.direct_override_suppressions,
            "retained_native_commits": self.retained_native_commits,
            "retained_native_unavailable": self.retained_native_unavailable,
            "final_step_suppressions": self.final_step_suppressions,
            "ree_closed_return_vetoes": self.ree_closed_return_vetoes,
            "topology_snapshots": self.topology_snapshots,
            "topology_restores": self.topology_restores,
            "temporal_history_contract": (
                "cache global causal features, then replay only the final "
                "2-4 controls with equal availability masks"
            ),
            "intervention_contract": (
                "never directly commit a disagreement; execute it only as a "
                "one-step checkpointed trial; commit it after post-excursion "
                "closure or restore the retained ETP action after verified return"
            ),
        })
        return value


def _install_topology_restore_layer() -> None:
    """Run graph restoration after sensing/update and before navigation."""
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer

    original = RLTrainer._nav_gmap_variable

    def restore_then_navigate(self, cur_vp, cur_pos, cur_ori, task_type):
        state = v510.v59.v58.v57.v56.v55._CONTROLLER
        if state is not None:
            state.restore_checkpoint_topology(self, cur_vp)
        return original(self, cur_vp, cur_pos, cur_ori, task_type)

    RLTrainer._nav_gmap_variable = restore_then_navigate


def _install_v5_12_hooks() -> None:
    v55_module = v510.v59.v58.v57.v56.v55
    original_installer = v55_module.install_native_hooks

    def install() -> None:
        _install_topology_restore_layer()
        original_installer()

    v55_module.install_native_hooks = install


def _validate_executed_actions(state, base_trace: Path) -> dict:
    actions = [
        json.loads(line) for line in base_trace.read_text().splitlines() if line
    ]
    checks = []
    for event in state.events:
        if event["event"] == "reversible_alternative_trial_created":
            expected = event["trial_branch"]
            kind = "alternative_trial"
        elif event["event"] == "retained_native_committed":
            expected = event["branch_id"]
            kind = "retained_native"
        else:
            continue
        step = event["step"]
        observed = actions[step].get("ghost_vp") if step < len(actions) else None
        checks.append({
            "kind": kind,
            "step": step,
            "expected_branch": expected,
            "executed_ghost_vp": observed,
            "equal": observed == expected,
        })
    if not all(row["equal"] for row in checks):
        raise RuntimeError("V5.12 declared/executed action identity mismatch")
    return {
        "checks": len(checks),
        "all_equal": True,
        "alternative_trial_checks": sum(
            row["kind"] == "alternative_trial" for row in checks
        ),
        "retained_native_checks": sum(
            row["kind"] == "retained_native" for row in checks
        ),
        "rows": checks,
    }


def main() -> None:
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
            break
    if run_dir is None:
        raise SystemExit("--run-dir is required")
    v510.NativeControlFullOPPActionController = (
        AlignedNativeControlFullOPPActionController
    )
    _install_v5_12_hooks()
    v510.main()
    state = v510.v59.v58.v57.v56.v55._CONTROLLER
    if not isinstance(state, AlignedNativeControlFullOPPActionController):
        raise RuntimeError("V5.12 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-full-opp-worker/5.12"
    summary["correctness_revision"] = (
        "equal causal control histories plus reversible disagreement trials "
        "and return-conditioned restoration of the retained native action"
    )
    summary["safety_funnel"] = state.safety_funnel()
    if summary.get("mode") == "revealnav":
        summary["executed_action_validation"] = _validate_executed_actions(
            state, run_dir / "base_trace.jsonl"
        )
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
