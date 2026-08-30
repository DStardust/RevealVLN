#!/usr/bin/env python3
"""Separate exploration from learned action replacement at OPP checkpoints."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_evidence_dominant_option_search_worker_v5_20 as v520  # noqa: E402
import r2r_remaining_set_rerank_worker_v5_17 as v517  # noqa: E402

NativeFirstRemainingSetController = v517.RemainingSetRerankController
RobustAlternativeFirstController = (
    v520.v519.AlternativeFirstOptionSearchController
)


class ConsensusExplorationOptionSearchController(
    v520.EvidenceDominantOptionSearchController
):
    """Probe an information branch only under OPP/native action consensus."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.consensus_exploration_trials = 0
        self.replacement_conflict_suppressions = 0
        self.evidence_credits_granted = 0
        self.evidence_conditioned_updates = 0
        self.evidence_credit_available = False
        self.current_trial_origin: str | None = None

    @staticmethod
    def _opp_selected_branch(value):
        action = value["action"]
        if action == "commit":
            return value["commit_branch"]
        if action == "explore":
            return value["macro"].branch_id
        return None

    @staticmethod
    def _gate_origin(selected, native_branch, evidence_credit: bool) -> str:
        if selected == native_branch:
            return "consensus_information_probe"
        if evidence_credit:
            return "evidence_conditioned_action_update"
        return "native_first_no_direct_replacement"

    def _start_native_trial(
        self, controls, native_branch, alternative, alternative_source, value,
    ):
        selected = self._opp_selected_branch(value)
        consensus = selected == native_branch
        origin = self._gate_origin(
            selected, native_branch, self.evidence_credit_available
        )
        self.record(
            "consensus_exploration_gate",
            checkpoint_id=v517.pilot._CURRENT_IDS[0],
            native_branch=native_branch,
            opp_selected_branch=selected,
            macro_branch=value["macro"].branch_id,
            alternative_branch=alternative,
            action_consensus=consensus,
            evidence_credit_available=self.evidence_credit_available,
            decision=origin,
            causal_inputs_only=True,
        )
        if origin == "native_first_no_direct_replacement":
            self.replacement_conflict_suppressions += 1
            return NativeFirstRemainingSetController._start_native_trial(
                self, controls, native_branch, alternative,
                alternative_source, value,
            )
        selected_branch = super()._start_native_trial(
            controls, native_branch, alternative, alternative_source, value,
        )
        if (
            selected_branch == alternative
            and self.current_trial_is_alternative_first
        ):
            self.current_trial_origin = origin
            if consensus:
                self.consensus_exploration_trials += 1
            else:
                self.evidence_credit_available = False
                self.evidence_conditioned_updates += 1
        return selected_branch

    def _post_decision(self, current) -> None:
        origin = self.current_trial_origin
        before = len(self.events)
        if origin == "evidence_conditioned_action_update":
            RobustAlternativeFirstController._post_decision(self, current)
        else:
            super()._post_decision(current)
        if origin is None:
            return
        post = [
            event for event in self.events[before:]
            if event.get("event") == "post_decision"
        ]
        if len(post) != 1:
            raise RuntimeError("V5.21 trial lacks one post decision")
        accepted = post[0].get("executed_return") is False
        if origin == "consensus_information_probe" and accepted:
            self.evidence_credit_available = True
            self.evidence_credits_granted += 1
            self.record(
                "evidence_credit_granted",
                checkpoint_id=self.checkpoint_id,
                branch_id=self.selected_branch,
                scope="next_OPP_ETP_action_conflict_only",
                causal_precondition=(
                    "accepted_consensus_information_probe"
                ),
            )
        if accepted or self.pending_return_action is None:
            self.current_trial_origin = None

    def complete_pending_return(self) -> None:
        super().complete_pending_return()
        self.current_trial_origin = None

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value.update({
            "consensus_exploration_trials": self.consensus_exploration_trials,
            "replacement_conflict_suppressions": (
                self.replacement_conflict_suppressions
            ),
            "evidence_credits_granted": self.evidence_credits_granted,
            "evidence_conditioned_updates": self.evidence_conditioned_updates,
            "evidence_credit_available_at_terminal": (
                self.evidence_credit_available
            ),
            "pre_action_gate": (
                "probe the distinct OPP macro information branch only when "
                "the OPP task action agrees with frozen ETP; when they "
                "disagree, keep the frozen-ETP native-first reversible path "
                "unless an accepted consensus probe grants one causal update"
            ),
            "intervention_contract": (
                "decouple task-action selection from information gathering: "
                "OPP/native consensus permits a reversible macro exploration "
                "probe; accepting it grants exactly one evidence-conditioned "
                "action update, while an ungrounded conflict cannot directly "
                "replace frozen ETP; evidence dominance accepts or returns the probe, "
                "and rejected options use verified topology restoration plus "
                "frozen-ETP remaining-set reranking"
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
    v520.EvidenceDominantOptionSearchController = (
        ConsensusExplorationOptionSearchController
    )
    v520.main()
    state = v520.v519.v517.V55._CONTROLLER
    if not isinstance(state, ConsensusExplorationOptionSearchController):
        raise RuntimeError("V5.21 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-worker/5.21"
    summary["method_revision"] = (
        "consensus-gated evidence-dominant reversible exploration"
    )
    summary["safety_funnel"] = state.safety_funnel()
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
