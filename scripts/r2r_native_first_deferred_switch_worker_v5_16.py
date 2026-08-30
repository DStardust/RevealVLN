#!/usr/bin/env python3
"""Native-first, evidence-delayed reversible branch switching for R2R.

This revision never replaces ETP-R1's outbound action from pre-action scores.
It saves a stable alternative, executes the native action, and switches only
after a unanimous frozen post-Q rejection and a verified physical return.
"""

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
import r2r_aligned_native_control_opp_worker_v5_12 as v512  # noqa: E402
import r2r_continuous_controller_worker_v5_2 as v52  # noqa: E402
from revealnav_mf2r3 import OptionStatus  # noqa: E402
from revealnav_mf2r4 import (  # noqa: E402
    BranchMacroAction,
    PostExcursionAction,
    PostExcursionQHead,
    StateConditionedReturnExecutor,
)
from rxr_unseen_controller_worker import sha256_file  # noqa: E402


class NativeFirstDeferredSwitchController(
    v512.AlignedNativeControlFullOPPActionController
):
    """Preserve ETP first; switch only on unanimous post-observation evidence."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.post_models = self._load_post_ensemble()
        self.retained_alternative: str | None = None
        self.retained_alternative_source: str | None = None
        self.trial_preservation_gain: float | None = None
        self.pending_alternative: dict | None = None
        self.native_first_trials = 0
        self.native_first_shadow_trials = 0
        self.unanimous_return_decisions = 0
        self.ensemble_disagreement_vetoes = 0
        self.ree_closed_return_vetoes = 0
        self.alternative_commits = 0
        self.alternative_unavailable = 0
        self.return_schedule_failures = 0

    def _load_post_ensemble(self) -> tuple[PostExcursionQHead, ...]:
        lock = json.loads(pilot.POST_LOCK.read_text())
        rows = lock["post_excursion_checkpoints"]
        if tuple(sorted(row["seed"] for row in rows)) != (
            20260826, 20260827, 20260828,
        ):
            raise RuntimeError("frozen post-Q ensemble seed set drift")
        models = []
        for row in sorted(rows, key=lambda value: value["seed"]):
            path = ROOT / row["path"]
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != row["bytes"]
                or sha256_file(path) != row["sha256"]
            ):
                raise RuntimeError("frozen post-Q ensemble checkpoint drift")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = PostExcursionQHead(768, 96, 5.0)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to(self.device).eval()
            models.append(model)
        return tuple(models)

    @staticmethod
    def _ensemble_backtrack_votes(
        costs: list[tuple[float, float]],
    ) -> tuple[tuple[bool, ...], bool]:
        if len(costs) != 3:
            raise RuntimeError("native-first gate requires exactly three post-Q heads")
        votes = tuple(backtrack < keep for keep, backtrack in costs)
        return votes, all(votes)

    def _consume_pending_alternative(self):
        pending = self.pending_alternative
        if pending is None:
            return None, False
        checkpoint_id = pilot._CURRENT_IDS[0]
        alternative = pending["branch_id"]
        if pending.get("topology_restored") is not True:
            raise RuntimeError("alternative reached before topology restoration")
        if self.step >= int(pilot._TRAINER.max_len) - 1:
            self.record(
                "retained_alternative_cancelled",
                checkpoint_id=pending["checkpoint_id"],
                observed_checkpoint_id=checkpoint_id,
                branch_id=alternative,
                reason="ETP_forces_STOP_at_episode_horizon",
                fail_closed=True,
            )
            self._clear_pending_alternative()
            return None, True
        if alternative not in self.global_current:
            self.alternative_unavailable += 1
            self.record(
                "retained_alternative_cancelled",
                checkpoint_id=pending["checkpoint_id"],
                observed_checkpoint_id=checkpoint_id,
                branch_id=alternative,
                reason="retained_alternative_unavailable_after_verified_return",
                fail_closed=True,
            )
            self._clear_pending_alternative()
            return None, True
        if not self.ledger.authorize_branch(
            checkpoint_id, alternative, pending["preservation_gain"]
        ):
            raise RuntimeError("retained alternative is no longer ledger-authorized")
        self.ledger.resolve_continue(checkpoint_id, alternative)
        self.alternative_commits += 1
        self.record(
            "retained_alternative_committed",
            checkpoint_id=checkpoint_id,
            branch_id=alternative,
            rejected_native_branch=pending["native_branch"],
            alternative_source=pending["alternative_source"],
            preservation_gain=round(pending["preservation_gain"], 8),
            causal_precondition=(
                "unanimous_three_head_post_rejection_and_verified_return"
            ),
        )
        self._clear_pending_alternative()
        return alternative, True

    def _clear_pending_alternative(self) -> None:
        self.pending_alternative = None
        self.checkpoint_graph_snapshot = None
        self.checkpoint_graph_signature = None
        self.global_rows.clear()
        self._reset_search()

    def _start_native_trial(
        self, controls, native_branch, alternative, alternative_source, value,
    ):
        macro = value["macro"]
        preservation_gain = float(macro.preservation_gain)
        checkpoint_id = pilot._CURRENT_IDS[0]
        self.native_first_trials += 1
        if self.mode == "shadow":
            self.native_first_shadow_trials += 1
            self.record(
                "native_first_trial_shadow",
                checkpoint_id=checkpoint_id,
                native_branch=native_branch,
                retained_alternative=alternative,
                alternative_source=alternative_source,
                preservation_gain=round(preservation_gain, 8),
                shadow_only_not_executed=True,
            )
            self.global_rows.clear()
            self._reset_search()
            return None

        self.ledger.register(checkpoint_id, controls)
        if not self.ledger.authorize_branch(
            checkpoint_id, native_branch, preservation_gain
        ):
            self.direct_override_suppressions += 1
            self.record(
                "native_first_trial_suppressed",
                checkpoint_id=checkpoint_id,
                native_branch=native_branch,
                reason="native_branch_already_resolved_at_checkpoint",
                fail_closed_to_native=True,
            )
            self.global_rows.clear()
            self._reset_search()
            return None

        self.phase = "outbound_in_flight"
        self.pre_histories = [row[0].detach() for row in self.rows]
        self.selected_embedding = self.global_current[native_branch].detach()
        self.checkpoint_embedding = self.latest_history.detach()
        self.selected_branch = native_branch
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
        self.executor.start_excursion(native_branch)
        self.checkpoint_candidates = {
            branch_id: self.global_current[branch_id].detach()
            for branch_id in controls
        }
        self.retained_alternative = alternative
        self.retained_alternative_source = alternative_source
        self.trial_preservation_gain = preservation_gain
        self.checkpointed_excursions += 1
        self.record(
            "native_first_trial_created",
            checkpoint_id=checkpoint_id,
            native_branch=native_branch,
            retained_alternative=alternative,
            alternative_source=alternative_source,
            preservation_gain=round(preservation_gain, 8),
            base_action_overridden=False,
        )
        self.global_rows.clear()
        return native_branch

    def _initial_decision(self, current, persistent, native_branch):
        pending, consumed = self._consume_pending_alternative()
        if consumed:
            return pending
        self.global_rows.append((
            self.latest_history.detach(), dict(self.global_current)
        ))
        self.temporal_prefixes_cached += 1
        value, controls = self._aligned_value(current, persistent, native_branch)
        if value is None:
            return None
        action = value["action"]
        selected = (
            value["commit_branch"] if action == "commit"
            else value["macro"].branch_id if action == "explore" else None
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
        macro = value["macro"]
        alternative, source = self._backup_branch(
            value, controls, native_branch
        )
        valid = (
            macro.action is BranchMacroAction.CHECKPOINTED_EXCURSION
            and macro.preservation_gain is not None
            and macro.preservation_gain > v512.V54.FROZEN_CONFIG["opv_threshold"]
            and alternative is not None
            and self.step < int(pilot._TRAINER.max_len) - 1
        )
        if not valid:
            if selected is not None and selected != native_branch:
                self.direct_override_suppressions += 1
                self.record(
                    "pre_action_replacement_suppressed",
                    proposed_branch=selected,
                    native_branch=native_branch,
                    reason="no_valid_native_first_reversible_trial",
                    fail_closed_to_native=True,
                )
            self.global_rows.clear()
            self._reset_search()
            return None
        return self._start_native_trial(
            controls, native_branch, alternative, source, value
        )

    def _post_decision(self, current) -> None:
        history = torch.stack(
            [*self.pre_histories, self.latest_history.detach()]
        ).unsqueeze(0)
        local = (
            torch.stack(list(current.values())).mean(0)
            if current else torch.zeros(768, device=self.device)
        )
        inputs = (
            history,
            torch.tensor([history.shape[1]], device=self.device),
            self.instruction.unsqueeze(0),
            self.selected_embedding.unsqueeze(0),
            self.checkpoint_embedding.unsqueeze(0),
            local.unsqueeze(0),
            torch.tensor([1.0], device=self.device),
        )
        costs = []
        with torch.no_grad():
            for model in self.post_models:
                output = model(*inputs)
                costs.append((
                    float(output.continue_cost[0]),
                    float(output.backtrack_cost[0]),
                ))
        votes, unanimous_backtrack = self._ensemble_backtrack_votes(costs)
        if unanimous_backtrack:
            self.raw_post_backtracks += 1
        else:
            self.raw_post_continues += 1
            if any(votes):
                self.ensemble_disagreement_vetoes += 1
        belief = self._post_ree_belief(current)
        ree_closed, reason = self.event_gate.post_excursion_decision(
            belief["p_discriminable"], belief["evidence"],
            belief["selected_target_probability"],
        )
        execute_return = unanimous_backtrack and not ree_closed
        if unanimous_backtrack and ree_closed:
            self.ree_closed_return_vetoes += 1
        if execute_return:
            self.unanimous_return_decisions += 1
        action = (
            PostExcursionAction.BACKTRACK
            if execute_return else PostExcursionAction.CONTINUE
        )
        self.post_policy_action = action.value
        self.record(
            "post_decision",
            policy_action=action.value,
            post_q_votes=["backtrack" if vote else "continue" for vote in votes],
            unanimous_backtrack=unanimous_backtrack,
            ree_closed_selected_branch=ree_closed,
            ree_reason=reason,
            predicted_costs=[
                {"continue": round(keep, 8), "backtrack": round(backtrack, 8)}
                for keep, backtrack in costs
            ],
            executed_return=execute_return,
            forced_stress_return=False,
            **{key: round(value, 8) for key, value in belief.items()},
        )
        checkpoint_id = self.checkpoint_id
        native_branch = self.selected_branch
        if not execute_return:
            self.continue_decisions += 1
            self.executor.continue_excursion()
            self.ledger.resolve_continue(checkpoint_id, native_branch)
            self.retained_alternative = None
            self.retained_alternative_source = None
            self.trial_preservation_gain = None
            self.checkpoint_graph_snapshot = None
            self.checkpoint_graph_signature = None
            self._reset_search()
            return
        self.backtrack_decisions += 1
        if not self._schedule_return():
            self.return_schedule_failures += 1
            if self.ledger.status(checkpoint_id, native_branch) is OptionStatus.ACTIVE:
                self.ledger.resolve_continue(checkpoint_id, native_branch)
            self.retained_alternative = None
            self.retained_alternative_source = None
            self.trial_preservation_gain = None
            self.checkpoint_graph_snapshot = None
            self.checkpoint_graph_signature = None

    def complete_pending_return(self) -> None:
        checkpoint_id = self.checkpoint_id
        native_branch = self.selected_branch
        alternative = self.retained_alternative
        alternative_source = self.retained_alternative_source
        preservation_gain = self.trial_preservation_gain
        v52.ContinuousController.complete_pending_return(self)
        success = bool(self.return_intervention_success)
        if self.ledger.status(checkpoint_id, native_branch) is OptionStatus.ACTIVE:
            if success:
                self.ledger.resolve_return(checkpoint_id, native_branch)
            else:
                self.ledger.resolve_continue(checkpoint_id, native_branch)
        if (
            success
            and alternative is not None
            and alternative_source is not None
            and preservation_gain is not None
        ):
            self.pending_alternative = {
                "checkpoint_id": checkpoint_id,
                "native_branch": native_branch,
                "branch_id": alternative,
                "alternative_source": alternative_source,
                "preservation_gain": preservation_gain,
            }
            self.record(
                "retained_alternative_armed",
                checkpoint_id=checkpoint_id,
                rejected_native_branch=native_branch,
                branch_id=alternative,
                return_verified=True,
            )
        else:
            self.pending_alternative = None
            self.checkpoint_graph_snapshot = None
            self.checkpoint_graph_signature = None
        self.retained_alternative = None
        self.retained_alternative_source = None
        self.trial_preservation_gain = None
        self.global_rows.clear()

    def restore_checkpoint_topology(self, trainer, cur_vp) -> None:
        if self.pending_alternative is None:
            return
        if self.checkpoint_graph_snapshot is None:
            raise RuntimeError("verified return lacks checkpoint graph snapshot")
        if len(trainer.gmaps) != 1 or len(cur_vp) != 1:
            raise RuntimeError("V5.16 topology restoration requires one environment")
        trainer.gmaps[0] = self.checkpoint_graph_snapshot
        checkpoint_id = self.pending_alternative["checkpoint_id"]
        cur_vp[0] = checkpoint_id
        restored = self._graph_signature(trainer.gmaps[0])
        if restored != self.checkpoint_graph_signature:
            raise RuntimeError("restored checkpoint graph signature drift")
        self.checkpoint_graph_snapshot = None
        self.topology_restores += 1
        self.pending_alternative["topology_restored"] = True
        self.record(
            "checkpoint_topology_restored",
            checkpoint_id=checkpoint_id,
            graph_nodes=len(restored["nodes"]),
            graph_ghosts=len(restored["ghosts"]),
            graph_edges=restored["edges"],
            transient_current_id_rewritten=True,
        )

    def finalize_episode(self) -> None:
        super().finalize_episode()
        if self.pending_alternative is not None:
            self.record(
                "terminal_retained_alternative_not_executed",
                branch_id=self.pending_alternative["branch_id"],
                fail_closed=True,
            )

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value.update({
            "native_first_trials": self.native_first_trials,
            "native_first_shadow_trials": self.native_first_shadow_trials,
            "unanimous_return_decisions": self.unanimous_return_decisions,
            "ensemble_disagreement_vetoes": self.ensemble_disagreement_vetoes,
            "ree_closed_return_vetoes": self.ree_closed_return_vetoes,
            "alternative_commits": self.alternative_commits,
            "alternative_unavailable": self.alternative_unavailable,
            "return_schedule_failures": self.return_schedule_failures,
            "post_q_ensemble_seeds": [20260826, 20260827, 20260828],
            "intervention_contract": (
                "execute ETP native first; switch once only after unanimous "
                "three-head post-Q rejection, open REE evidence, verified "
                "physical return, exact topology restoration, and retained "
                "alternative identity validation"
            ),
        })
        return value


def _validate_executed_actions(state, base_trace: Path) -> dict:
    actions = [
        json.loads(line) for line in base_trace.read_text().splitlines() if line
    ]
    checks = []
    for event in state.events:
        if event["event"] == "native_first_trial_created":
            expected = event["native_branch"]
            kind = "native_outbound"
        elif event["event"] == "retained_alternative_committed":
            expected = event["branch_id"]
            kind = "retained_alternative"
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
        raise RuntimeError("V5.16 declared/executed action identity mismatch")
    return {"checks": len(checks), "all_equal": True, "rows": checks}


def main() -> None:
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
            break
    if run_dir is None:
        raise SystemExit("--run-dir is required")
    v512.AlignedNativeControlFullOPPActionController = (
        NativeFirstDeferredSwitchController
    )
    v512.main()
    state = v512.v510.v59.v58.v57.v56.v55._CONTROLLER
    if not isinstance(state, NativeFirstDeferredSwitchController):
        raise RuntimeError("V5.16 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-worker/5.16"
    summary["method_revision"] = (
        "native-first delayed commitment with unanimous post-Q return and "
        "return-conditioned alternative execution"
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
