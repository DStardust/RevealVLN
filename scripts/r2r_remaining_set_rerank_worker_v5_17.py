#!/usr/bin/env python3
"""Native-first option elimination with frozen-ETP remaining-set reranking."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_continuous_controller_worker_v5_2 as v52  # noqa: E402
import r2r_native_first_deferred_switch_worker_v5_16 as v516  # noqa: E402
from revealnav_mf2r3 import OptionStatus  # noqa: E402
from revealnav_mf2r4 import PostExcursionAction  # noqa: E402


V55 = v516.v512.v510.v59.v58.v57.v56.v55


class RemainingSetRerankController(v516.NativeFirstDeferredSwitchController):
    """Reject one evidenced-bad option, then let frozen ETP rank the rest."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint_control_ids: tuple[str, ...] = ()
        self.exhausted_option_ids: set[str] = set()
        self.search_histories: list[torch.Tensor] = []
        self.search_rows: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        self.latest_probe_row: tuple[
            torch.Tensor, dict[str, torch.Tensor]
        ] | None = None
        self.current_probe_is_reranked = False
        self.current_probe_navigation_step: int | None = None
        self.rerank_ready: dict | None = None
        self.switch_budget_suppressions = 0
        self.robust_median_returns = 0
        self.robust_median_disagreements = 0
        self.remaining_set_rerank_commits = 0
        self.remaining_set_stop_commits = 0
        self.remaining_set_rerank_cancellations = 0
        self.remaining_set_probe_count = 0

    @staticmethod
    def _has_switch_budget(step: int, max_len: int) -> bool:
        # Native excursion, physical return, and reranked branch must all occur
        # before ETP's max_len-1 forced-STOP boundary.
        return step < max_len - 3

    @staticmethod
    def _robust_post_decision(
        costs: list[tuple[float, float]],
    ) -> tuple[tuple[bool, ...], float, float, bool]:
        if len(costs) != 3:
            raise RuntimeError("robust post gate requires exactly three heads")
        votes = tuple(backtrack < keep for keep, backtrack in costs)
        median_continue = sorted(keep for keep, _ in costs)[1]
        median_backtrack = sorted(backtrack for _, backtrack in costs)[1]
        return (
            votes,
            median_continue,
            median_backtrack,
            median_backtrack < median_continue,
        )

    @staticmethod
    def _remaining_index(
        ids, logits: torch.Tensor, controls: tuple[str, ...],
        exhausted: set[str],
    ) -> int | None:
        remaining = logits.clone()
        allowed = set(controls) - exhausted
        for index, branch_id in enumerate(ids):
            if index == 0:
                continue
            if branch_id is None or str(branch_id) not in allowed:
                remaining[index] = -torch.inf
        index = int(torch.argmax(remaining))
        if not torch.isfinite(remaining[index]):
            return None
        return index

    @staticmethod
    def _eligible_score_evidence(
        ids, logits: torch.Tensor, controls: tuple[str, ...],
        exhausted: set[str],
    ) -> list[dict]:
        allowed = set(controls) - exhausted
        rows = []
        for index, branch_id in enumerate(ids):
            normalized = None if index == 0 or branch_id is None else str(branch_id)
            if index != 0 and normalized not in allowed:
                continue
            score = float(logits[index])
            rows.append({
                "index": index,
                "branch_id": normalized,
                "score": score if math.isfinite(score) else None,
                "finite": math.isfinite(score),
            })
        return rows

    def _start_native_trial(
        self, controls, native_branch, alternative, alternative_source, value,
    ):
        if not self._has_switch_budget(self.step, int(pilot._TRAINER.max_len)):
            self.switch_budget_suppressions += 1
            self.record(
                "native_first_trial_suppressed",
                native_branch=native_branch,
                reason="insufficient_budget_for_excursion_return_and_rerank",
                current_step=self.step,
                max_len=int(pilot._TRAINER.max_len),
                fail_closed_to_native=True,
            )
            self.global_rows.clear()
            self._reset_search()
            return None
        selected = super()._start_native_trial(
            controls, native_branch, alternative, alternative_source, value
        )
        if selected is not None:
            self.checkpoint_control_ids = tuple(controls)
            self.exhausted_option_ids = set()
            self.search_histories = [value.detach() for value in self.pre_histories]
            self.search_rows = [
                (
                    history.detach(),
                    {key: embedding.detach() for key, embedding in candidates.items()},
                )
                for history, candidates in self.rows
            ]
            if len(self.search_rows) != len(self.search_histories):
                raise RuntimeError("native trial temporal candidate rows are misaligned")
            self.latest_probe_row = None
            self.current_probe_is_reranked = False
            self.current_probe_navigation_step = self.step
        return selected

    def _clear_trial_state(self) -> None:
        self.retained_alternative = None
        self.retained_alternative_source = None
        self.trial_preservation_gain = None
        self.checkpoint_control_ids = ()
        self.exhausted_option_ids.clear()
        self.search_histories.clear()
        self.search_rows.clear()
        self.latest_probe_row = None
        self.current_probe_is_reranked = False
        self.current_probe_navigation_step = None
        self.checkpoint_graph_snapshot = None
        self.checkpoint_graph_signature = None
        self._reset_search()

    def _post_decision(self, current) -> None:
        post_candidates = dict(self.checkpoint_candidates)
        post_candidates.update(current)
        post_candidates[self.selected_branch] = self.selected_embedding
        self.latest_probe_row = (
            self.latest_history.detach(),
            {
                key: embedding.detach()
                for key, embedding in post_candidates.items()
            },
        )
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
        votes, median_continue, median_backtrack, robust_backtrack = (
            self._robust_post_decision(costs)
        )
        if len(set(votes)) > 1:
            self.robust_median_disagreements += 1
        if robust_backtrack:
            self.raw_post_backtracks += 1
        else:
            self.raw_post_continues += 1
        belief = self._post_ree_belief(current)
        ree_closed, reason = self.event_gate.post_excursion_decision(
            belief["p_discriminable"], belief["evidence"],
            belief["selected_target_probability"],
        )
        execute_return = robust_backtrack and not ree_closed
        if robust_backtrack and ree_closed:
            self.ree_closed_return_vetoes += 1
        if execute_return:
            self.robust_median_returns += 1
        action = (
            PostExcursionAction.BACKTRACK
            if execute_return else PostExcursionAction.CONTINUE
        )
        self.post_policy_action = action.value
        self.record(
            "post_decision",
            policy_action=action.value,
            post_q_votes=["backtrack" if vote else "continue" for vote in votes],
            robust_estimator="three_head_coordinatewise_median",
            median_continue_cost=round(median_continue, 8),
            median_backtrack_cost=round(median_backtrack, 8),
            robust_median_backtrack=robust_backtrack,
            ree_closed_selected_branch=ree_closed,
            ree_reason=reason,
            predicted_costs=[
                {"continue": round(keep, 8), "backtrack": round(backtrack, 8)}
                for keep, backtrack in costs
            ],
            executed_return=execute_return,
            forced_stress_return=False,
            temporal_history_steps=history.shape[1],
            historical_candidate_row_steps=len(self.rows),
            final_candidate_count=len(post_candidates),
            **{key: round(value, 8) for key, value in belief.items()},
        )
        checkpoint_id = self.checkpoint_id
        native_branch = self.selected_branch
        if not execute_return:
            self.continue_decisions += 1
            self.executor.continue_excursion()
            self.ledger.resolve_continue(checkpoint_id, native_branch)
            if self.current_probe_is_reranked:
                self.alternative_commits += 1
                self.remaining_set_rerank_commits += 1
                self.record(
                    "remaining_set_probe_accepted",
                    checkpoint_id=checkpoint_id,
                    branch_id=native_branch,
                    navigation_step=self.current_probe_navigation_step,
                    acceptance="robust_post_Q_continue_or_REE_closed",
                )
            self._clear_trial_state()
            return
        self.backtrack_decisions += 1
        if not self._schedule_return():
            self.return_schedule_failures += 1
            if self.ledger.status(checkpoint_id, native_branch) is OptionStatus.ACTIVE:
                self.ledger.resolve_continue(checkpoint_id, native_branch)
            self._clear_trial_state()

    def complete_pending_return(self) -> None:
        checkpoint_id = self.checkpoint_id
        native_branch = self.selected_branch
        controls = self.checkpoint_control_ids
        histories = [*self.search_histories, self.latest_history.detach()]
        if self.latest_probe_row is None:
            raise RuntimeError("post-excursion temporal candidate row is absent")
        search_rows = [*self.search_rows, self.latest_probe_row]
        if len(search_rows) != len(histories):
            raise RuntimeError("returned temporal candidate rows are misaligned")
        exhausted = {*self.exhausted_option_ids, native_branch}
        preservation_gain = self.trial_preservation_gain
        v52.ContinuousController.complete_pending_return(self)
        success = bool(self.return_intervention_success)
        if self.ledger.status(checkpoint_id, native_branch) is OptionStatus.ACTIVE:
            if success:
                self.ledger.resolve_return(checkpoint_id, native_branch)
            else:
                self.ledger.resolve_continue(checkpoint_id, native_branch)
        if success and controls and preservation_gain is not None:
            self.pending_alternative = {
                "checkpoint_id": checkpoint_id,
                "native_branch": native_branch,
                "candidate_ids": controls,
                "exhausted_option_ids": tuple(sorted(exhausted)),
                "search_histories": histories,
                "search_rows": search_rows,
                "preservation_gain": preservation_gain,
            }
            self.record(
                "remaining_set_rerank_armed",
                checkpoint_id=checkpoint_id,
                rejected_native_branch=native_branch,
                candidate_ids=list(controls),
                exhausted_option_ids=sorted(exhausted),
                return_verified=True,
            )
        else:
            self.pending_alternative = None
            self.checkpoint_graph_snapshot = None
            self.checkpoint_graph_signature = None
        self.retained_alternative = None
        self.retained_alternative_source = None
        self.trial_preservation_gain = preservation_gain if success else None
        self.checkpoint_control_ids = controls if success else ()
        self.exhausted_option_ids = exhausted if success else set()
        self.search_histories = histories if success else []
        self.search_rows = search_rows if success else []
        self.latest_probe_row = None
        self.current_probe_is_reranked = False
        self.current_probe_navigation_step = None
        self.global_rows.clear()

    def _initial_decision(self, current, persistent, native_branch):
        if self.pending_alternative is not None:
            if self.pending_alternative.get("topology_restored") is not True:
                raise RuntimeError("remaining-set rerank reached before restoration")
            restored_candidates = {
                branch_id: self.global_current[branch_id].detach()
                for branch_id in self.pending_alternative["candidate_ids"]
                if branch_id in self.global_current
            }
            if set(restored_candidates) != set(
                self.pending_alternative["candidate_ids"]
            ):
                raise RuntimeError(
                    "restored checkpoint is missing an aligned control embedding"
                )
            self.rerank_ready = {
                "navigation_step": self.step,
                "restored_history": self.latest_history.detach(),
                "restored_candidates": restored_candidates,
                **self.pending_alternative,
            }
            self.checkpoint_control_ids = tuple(
                self.pending_alternative["candidate_ids"]
            )
            self.exhausted_option_ids = set(
                self.pending_alternative["exhausted_option_ids"]
            )
            self.search_histories = list(
                self.pending_alternative["search_histories"]
            )
            self.search_rows = list(self.pending_alternative["search_rows"])
            return None
        return super()._initial_decision(current, persistent, native_branch)

    def apply_remaining_set_rerank(self, result: dict, gmap_vp_ids) -> dict:
        ready = self.rerank_ready
        if ready is None:
            return result
        ids = gmap_vp_ids[0]
        logits = result["global_logits"][0]
        index = self._remaining_index(
            ids, logits, tuple(ready["candidate_ids"]),
            set(ready["exhausted_option_ids"]),
        )
        if index is None:
            self.remaining_set_rerank_cancellations += 1
            self.record(
                "remaining_set_rerank_cancelled",
                navigation_step=ready["navigation_step"],
                checkpoint_id=ready["checkpoint_id"],
                rejected_native_branch=ready["native_branch"],
                reason="no_finite_restored_remaining_candidate",
                fail_closed_to_frozen_etp=True,
            )
            self._finish_rerank()
            raise RuntimeError(
                "no finite frozen-ETP action in STOP plus remaining option set"
            )
        eligible_scores = self._eligible_score_evidence(
            ids, logits, tuple(ready["candidate_ids"]),
            set(ready["exhausted_option_ids"]),
        )
        branch_id = None if index == 0 or ids[index] is None else str(ids[index])
        checkpoint_id = pilot._CURRENT_IDS[0]
        if checkpoint_id != ready["checkpoint_id"]:
            raise RuntimeError(
                "restored checkpoint identity drift before remaining-set rerank"
            )
        base_index = int(torch.argmax(logits))
        base_branch = ids[base_index] if base_index > 0 else None
        remaining_count = sum(
            str(value) in (
                set(ready["candidate_ids"])
                - set(ready["exhausted_option_ids"])
            ) for value in ids if value is not None
        )
        if branch_id is None or not self._has_switch_budget(
            ready["navigation_step"], int(pilot._TRAINER.max_len)
        ):
            if branch_id is not None:
                if not self.ledger.authorize_branch(
                    checkpoint_id, branch_id, ready["preservation_gain"]
                ):
                    raise RuntimeError("ETP-reranked branch is not ledger-authorized")
                self.ledger.resolve_continue(checkpoint_id, branch_id)
            self.remaining_set_rerank_commits += 1
            if branch_id is None:
                self.remaining_set_stop_commits += 1
            else:
                self.alternative_commits += 1
            self.record(
                "remaining_set_rerank_committed",
                navigation_step=ready["navigation_step"],
                checkpoint_id=checkpoint_id,
                rejected_native_branch=ready["native_branch"],
                frozen_etp_pre_mask_action=base_branch,
                branch_id=branch_id,
                candidate_ids=list(ready["candidate_ids"]),
                exhausted_option_ids=list(ready["exhausted_option_ids"]),
                remaining_candidate_count=remaining_count,
                eligible_frozen_etp_scores=eligible_scores,
                selected_global_index=index,
                selection_rule="frozen_ETP_argmax_over_STOP_and_unexhausted_options",
                commit_reason=(
                    "frozen_ETP_STOP" if branch_id is None
                    else "option_expiry_no_complete_probe_budget"
                ),
                causal_precondition=(
                    "robust_post_rejection_verified_return_and_topology_restore"
                ),
            )
            self._finish_rerank()
        else:
            if not self.ledger.authorize_branch(
                checkpoint_id, branch_id, ready["preservation_gain"]
            ):
                raise RuntimeError("ETP-reranked probe is not ledger-authorized")
            graph = pilot._TRAINER.gmaps[0]
            self.phase = "outbound_in_flight"
            self.pre_histories = [
                *ready["search_histories"], ready["restored_history"]
            ]
            self.rows = [
                *ready["search_rows"],
                (ready["restored_history"], ready["restored_candidates"]),
            ]
            if len(self.rows) != len(self.pre_histories):
                raise RuntimeError("reranked probe temporal candidate rows are misaligned")
            self.selected_embedding = self.global_current[branch_id].detach()
            self.checkpoint_embedding = self.latest_history.detach()
            self.selected_branch = branch_id
            self.checkpoint_id = checkpoint_id
            self.checkpoint_position = torch.as_tensor(
                graph.node_pos[checkpoint_id]
            ).cpu().numpy().copy()
            self.checkpoint_graph_snapshot = copy.deepcopy(graph)
            self.checkpoint_graph_signature = self._graph_signature(graph)
            self.topology_snapshots += 1
            self.executor = v516.StateConditionedReturnExecutor(
                checkpoint_id, "ETP-R1:frozen-control",
                tuple(ready["candidate_ids"]),
            )
            self.executor.start_excursion(branch_id)
            self.checkpoint_candidates = {
                value: self.global_current[value].detach()
                for value in ready["candidate_ids"]
                if value in self.global_current
            }
            self.trial_preservation_gain = ready["preservation_gain"]
            self.checkpoint_control_ids = tuple(ready["candidate_ids"])
            self.exhausted_option_ids = set(ready["exhausted_option_ids"])
            self.search_histories = list(ready["search_histories"])
            self.search_rows = list(ready["search_rows"])
            self.search_histories.append(ready["restored_history"])
            self.search_rows.append((
                ready["restored_history"], ready["restored_candidates"]
            ))
            self.latest_probe_row = None
            self.current_probe_is_reranked = True
            self.current_probe_navigation_step = ready["navigation_step"]
            self.remaining_set_probe_count += 1
            self.checkpointed_excursions += 1
            self.record(
                "remaining_set_probe_created",
                navigation_step=ready["navigation_step"],
                checkpoint_id=checkpoint_id,
                rejected_native_branch=ready["native_branch"],
                exhausted_option_ids=list(ready["exhausted_option_ids"]),
                frozen_etp_pre_mask_action=base_branch,
                branch_id=branch_id,
                candidate_ids=list(ready["candidate_ids"]),
                remaining_candidate_count=remaining_count,
                eligible_frozen_etp_scores=eligible_scores,
                selected_global_index=index,
                selection_rule="frozen_ETP_argmax_over_STOP_and_unexhausted_options",
                reversible=True,
                temporal_history_steps=len(self.pre_histories),
                temporal_candidate_row_steps=len(self.rows),
                restored_control_ids=sorted(ready["restored_candidates"]),
            )
            self.pending_alternative = None
            self.rerank_ready = None
        changed = dict(result)
        forced = result["global_logits"].clone()
        forced[0].fill_(-1e9)
        forced[0, index] = 1e9
        changed["global_logits"] = forced
        return changed

    def _finish_rerank(self) -> None:
        self.pending_alternative = None
        self.rerank_ready = None
        self.checkpoint_control_ids = ()
        self.exhausted_option_ids.clear()
        self.search_histories.clear()
        self.search_rows.clear()
        self.latest_probe_row = None
        self.current_probe_is_reranked = False
        self.current_probe_navigation_step = None
        self.trial_preservation_gain = None
        self.checkpoint_graph_snapshot = None
        self.checkpoint_graph_signature = None
        self.global_rows.clear()
        self._reset_search()

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value.update({
            "switch_budget_suppressions": self.switch_budget_suppressions,
            "robust_median_returns": self.robust_median_returns,
            "robust_median_disagreements": self.robust_median_disagreements,
            "remaining_set_rerank_commits": self.remaining_set_rerank_commits,
            "remaining_set_stop_commits": self.remaining_set_stop_commits,
            "remaining_set_rerank_cancellations": (
                self.remaining_set_rerank_cancellations
            ),
            "remaining_set_probe_count": self.remaining_set_probe_count,
            "post_gate": "coordinatewise_median_of_three_frozen_Q_heads",
            "switch_rule": (
                "mark each rejected option exhausted, use frozen ETP logits "
                "to rank STOP and the restored unexhausted option set, and "
                "probe each branch at most once while transaction budget remains"
            ),
            "intervention_contract": (
                "execute frozen ETP native action first; reject an observed "
                "option only when the coordinatewise median of three frozen "
                "post-Q heads prefers backtracking and REE remains open; after "
                "verified physical return and exact topology restoration, "
                "rerank only STOP plus unexhausted checkpoint options with "
                "unchanged frozen ETP logits"
            ),
        })
        return value

    def finalize_episode(self) -> None:
        # V5.16's terminal reporter expects its one-alternative pending schema.
        # Bypass that reporter while retaining the shared idempotent unresolved-
        # excursion accounting from the V5.4 controller.
        if self._episode_finalized:
            return
        pending = self.pending_alternative or self.rerank_ready
        v516.v512.V54.FullOPPContinuousController.finalize_episode(self)
        if pending is not None:
            self.record(
                "terminal_remaining_set_rerank_not_executed",
                checkpoint_id=pending.get("checkpoint_id"),
                candidate_ids=list(pending.get("candidate_ids", ())),
                exhausted_option_ids=list(
                    pending.get("exhausted_option_ids", ())
                ),
                fail_closed=True,
            )


def _install_remaining_set_hook() -> None:
    original_installer = V55.install_native_hooks

    def install() -> None:
        original_installer()
        from vlnce_baselines.models.R1Policy import ETP

        original_forward = ETP.forward

        def rerank_wrapped(self, *args, **kwargs):
            result = original_forward(self, *args, **kwargs)
            mode = kwargs.get("mode", args[0] if args else None)
            state = V55._CONTROLLER
            if (
                mode == "navigation"
                and isinstance(state, RemainingSetRerankController)
                and state.rerank_ready is not None
            ):
                return state.apply_remaining_set_rerank(
                    result, kwargs["gmap_vp_ids"]
                )
            return result

        ETP.forward = rerank_wrapped

    V55.install_native_hooks = install


def _validate_actions(state, base_trace: Path) -> dict:
    actions = [
        json.loads(line) for line in base_trace.read_text().splitlines() if line
    ]
    checks = []
    for event in state.events:
        if event["event"] == "native_first_trial_created":
            expected = event["native_branch"]
            step = event["step"]
            kind = "native_outbound"
        elif event["event"] in (
            "remaining_set_rerank_committed", "remaining_set_probe_created",
        ):
            expected = event["branch_id"]
            step = event["navigation_step"]
            kind = "remaining_set_rerank"
        else:
            continue
        in_range = 0 <= step < len(actions)
        action = actions[step] if in_range else {}
        observed = action.get("ghost_vp")
        act = action.get("act")
        equal = (
            in_range
            and (
                (expected is None and act == 0)
                or (expected is not None and act == 4 and observed == expected)
            )
        )
        checks.append({
            "kind": kind, "step": step, "expected_branch": expected,
            "executed_act": act, "executed_ghost_vp": observed,
            "in_range": in_range, "equal": equal,
        })
    if not all(row["equal"] for row in checks):
        raise RuntimeError("V5.17 declared/executed action identity mismatch")
    return {"checks": len(checks), "all_equal": True, "rows": checks}


def main() -> None:
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
            break
    if run_dir is None:
        raise SystemExit("--run-dir is required")
    v516.NativeFirstDeferredSwitchController = RemainingSetRerankController
    _install_remaining_set_hook()
    v516.main()
    state = V55._CONTROLLER
    if not isinstance(state, RemainingSetRerankController):
        raise RuntimeError("V5.17 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-worker/5.17"
    summary["method_revision"] = (
        "budget-feasible native-first option elimination with robust post-Q "
        "median and frozen-ETP remaining-set reranking"
    )
    summary["safety_funnel"] = state.safety_funnel()
    if summary.get("mode") == "revealnav":
        summary["executed_action_validation"] = _validate_actions(
            state, run_dir / "base_trace.jsonl"
        )
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
