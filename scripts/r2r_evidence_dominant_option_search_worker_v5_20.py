#!/usr/bin/env python3
"""Alternative-first search with threshold-free evidence-dominance acceptance."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_alternative_first_option_search_worker_v5_19 as v519  # noqa: E402
import r2r_remaining_set_rerank_worker_v5_17 as v517  # noqa: E402
from revealnav_mf2r3 import OptionStatus  # noqa: E402
from revealnav_mf2r4 import PostExcursionAction  # noqa: E402


class EvidenceDominantOptionSearchController(
    v519.AlternativeFirstOptionSearchController
):
    """Commit an initial probe only when new REE evidence supports that option."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initial_probe_pre_belief: dict | None = None
        self.evidence_dominance_accepts = 0
        self.evidence_dominance_returns = 0

    def _start_native_trial(
        self, controls, native_branch, alternative, alternative_source, value,
    ):
        selected = super()._start_native_trial(
            controls, native_branch, alternative, alternative_source, value
        )
        if selected == alternative and self.current_trial_is_alternative_first:
            self.initial_probe_pre_belief = dict(value["belief"])
        return selected

    def _post_decision(self, current) -> None:
        if not self.current_trial_is_alternative_first:
            super()._post_decision(current)
            return
        if self.initial_probe_pre_belief is None:
            raise RuntimeError("alternative-first probe lacks pre-action belief")
        history = torch.stack(
            [*self.pre_histories, self.latest_history.detach()]
        ).unsqueeze(0)
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
        selected_is_top = (
            belief["selected_target_probability"]
            >= belief["maximum_target_probability"] - 1e-7
        )
        discriminability_nondecreasing = (
            belief["p_discriminable"]
            >= float(self.initial_probe_pre_belief["p_discriminable"])
        )
        evidence_accept = selected_is_top and discriminability_nondecreasing
        execute_return = (
            (robust_backtrack and not ree_closed) or not evidence_accept
        )
        if robust_backtrack and ree_closed and evidence_accept:
            self.ree_closed_return_vetoes += 1
        if execute_return:
            self.robust_median_returns += 1
            if not evidence_accept:
                self.evidence_dominance_returns += 1
        else:
            self.evidence_dominance_accepts += 1
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
            selected_is_post_target_argmax=selected_is_top,
            pre_p_discriminable=round(
                float(self.initial_probe_pre_belief["p_discriminable"]), 8
            ),
            discriminability_nondecreasing=discriminability_nondecreasing,
            evidence_dominance_accept=evidence_accept,
            acceptance_rule=(
                "selected target remains argmax AND REE discriminability "
                "does not decrease"
            ),
            executed_return=execute_return,
            forced_stress_return=False,
            temporal_history_steps=history.shape[1],
            historical_candidate_row_steps=len(self.rows),
            final_candidate_count=len(post_candidates),
            **{key: round(value, 8) for key, value in belief.items()},
        )
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        navigation_step = self.current_trial_navigation_step
        if not execute_return:
            self.continue_decisions += 1
            self.executor.continue_excursion()
            self.ledger.resolve_continue(checkpoint_id, branch_id)
            self.alternative_first_accepts += 1
            self.alternative_commits += 1
            self.record(
                "alternative_first_probe_accepted",
                checkpoint_id=checkpoint_id,
                branch_id=branch_id,
                navigation_step=navigation_step,
                acceptance="post_Q_and_evidence_dominance",
            )
            self.current_trial_is_alternative_first = False
            self.current_trial_navigation_step = None
            self.initial_probe_pre_belief = None
            self._clear_trial_state()
            return
        self.backtrack_decisions += 1
        if not self._schedule_return():
            self.return_schedule_failures += 1
            if self.ledger.status(checkpoint_id, branch_id) is OptionStatus.ACTIVE:
                self.ledger.resolve_continue(checkpoint_id, branch_id)
            self.current_trial_is_alternative_first = False
            self.current_trial_navigation_step = None
            self.initial_probe_pre_belief = None
            self._clear_trial_state()

    def complete_pending_return(self) -> None:
        super().complete_pending_return()
        self.initial_probe_pre_belief = None

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value.update({
            "evidence_dominance_accepts": self.evidence_dominance_accepts,
            "evidence_dominance_returns": self.evidence_dominance_returns,
            "initial_probe_acceptance": (
                "three-head median locally prefers continue AND the probed "
                "option remains REE target argmax AND REE discriminability "
                "is nondecreasing; the last two comparisons are threshold-free"
            ),
            "intervention_contract": (
                "probe the OPP/OPV alternative reversibly; post-Q certifies "
                "local continuation cost, while threshold-free REE evidence "
                "dominance certifies task relevance; any rejection triggers "
                "verified return, exact topology restoration, option exhaustion, "
                "and frozen-ETP remaining-set reranking"
            ),
        })
        return value


def main() -> None:
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
            break
    if run_dir is None:
        raise SystemExit("--run-dir is required")
    v519.AlternativeFirstOptionSearchController = (
        EvidenceDominantOptionSearchController
    )
    v519.main()
    state = v519.v517.V55._CONTROLLER
    if not isinstance(state, EvidenceDominantOptionSearchController):
        raise RuntimeError("V5.20 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-worker/5.20"
    summary["method_revision"] = (
        "threshold-free evidence-dominant reversible option search"
    )
    summary["safety_funnel"] = state.safety_funnel()
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
