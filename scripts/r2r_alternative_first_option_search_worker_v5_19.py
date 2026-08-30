#!/usr/bin/env python3
"""Alternative-first reversible probes with V5.17 remaining-set search."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_remaining_set_rerank_worker_v5_17 as v517  # noqa: E402


class AlternativeFirstOptionSearchController(v517.RemainingSetRerankController):
    """Probe OPP's alternative first, then reject it into C minus E if needed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alternative_first_trials = 0
        self.alternative_first_accepts = 0
        self.current_trial_is_alternative_first = False
        self.current_trial_navigation_step: int | None = None

    def _start_native_trial(
        self, controls, native_branch, alternative, alternative_source, value,
    ):
        if not self._has_switch_budget(self.step, int(pilot._TRAINER.max_len)):
            return super()._start_native_trial(
                controls, native_branch, alternative, alternative_source, value
            )
        preservation_gain = float(value["macro"].preservation_gain)
        checkpoint_id = pilot._CURRENT_IDS[0]
        self.ledger.register(checkpoint_id, controls)
        if not self.ledger.authorize_branch(
            checkpoint_id, alternative, preservation_gain
        ):
            raise RuntimeError("alternative-first branch is not ledger-authorized")
        self.phase = "outbound_in_flight"
        self.pre_histories = [row[0].detach() for row in self.rows]
        self.selected_embedding = self.global_current[alternative].detach()
        self.checkpoint_embedding = self.latest_history.detach()
        self.selected_branch = alternative
        self.checkpoint_id = checkpoint_id
        graph = pilot._TRAINER.gmaps[0]
        self.checkpoint_graph_snapshot = copy.deepcopy(graph)
        self.checkpoint_graph_signature = self._graph_signature(graph)
        self.topology_snapshots += 1
        self.checkpoint_position = np.asarray(
            graph.node_pos[checkpoint_id], dtype=float
        ).copy()
        self.executor = v517.v516.StateConditionedReturnExecutor(
            checkpoint_id, "ETP-R1:frozen-control", tuple(controls)
        )
        self.executor.start_excursion(alternative)
        self.checkpoint_candidates = {
            branch: self.global_current[branch].detach() for branch in controls
        }
        self.trial_preservation_gain = preservation_gain
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
            raise RuntimeError("alternative-first temporal rows are misaligned")
        self.latest_probe_row = None
        self.current_probe_is_reranked = False
        self.current_probe_navigation_step = None
        self.current_trial_is_alternative_first = True
        self.current_trial_navigation_step = self.step
        self.alternative_first_trials += 1
        self.checkpointed_excursions += 1
        self.record(
            "alternative_first_trial_created",
            checkpoint_id=checkpoint_id,
            trial_branch=alternative,
            retained_native_branch=native_branch,
            candidate_ids=list(controls),
            alternative_source=alternative_source,
            preservation_gain=round(preservation_gain, 8),
            reversible=True,
            direct_irreversible_commit=False,
        )
        self.global_rows.clear()
        return alternative

    def _post_decision(self, current) -> None:
        alternative_trial = self.current_trial_is_alternative_first
        checkpoint_id = self.checkpoint_id
        branch_id = self.selected_branch
        navigation_step = self.current_trial_navigation_step
        before = len(self.events)
        super()._post_decision(current)
        if not alternative_trial:
            return
        post = [
            event for event in self.events[before:]
            if event.get("event") == "post_decision"
        ]
        if len(post) != 1:
            raise RuntimeError("alternative-first probe lacks one post decision")
        if post[0].get("executed_return") is False:
            self.alternative_first_accepts += 1
            self.alternative_commits += 1
            self.record(
                "alternative_first_probe_accepted",
                checkpoint_id=checkpoint_id,
                branch_id=branch_id,
                navigation_step=navigation_step,
                acceptance="robust_post_Q_continue_or_REE_closed",
            )
            self.current_trial_is_alternative_first = False
            self.current_trial_navigation_step = None
        elif self.pending_return_action is None:
            self.current_trial_is_alternative_first = False
            self.current_trial_navigation_step = None

    def complete_pending_return(self) -> None:
        super().complete_pending_return()
        self.current_trial_is_alternative_first = False
        self.current_trial_navigation_step = None

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value.update({
            "alternative_first_trials": self.alternative_first_trials,
            "alternative_first_accepts": self.alternative_first_accepts,
            "intervention_contract": (
                "at every valid OPP/OPV checkpoint, probe the learned "
                "alternative for one causal observation; continue only when "
                "the robust post-Q median or closed REE evidence accepts it, "
                "otherwise physically return, restore topology, exhaust that "
                "option, and let frozen ETP rerank the remaining set"
            ),
        })
        return value


def _validate_actions(state, base_trace: Path) -> dict:
    actions = [
        json.loads(line) for line in base_trace.read_text().splitlines() if line
    ]
    checks = []
    for event in state.events:
        kind = event.get("event")
        if kind == "alternative_first_trial_created":
            expected, step = event["trial_branch"], event["step"]
        elif kind == "native_first_trial_created":
            expected, step = event["native_branch"], event["step"]
        elif kind in (
            "remaining_set_rerank_committed", "remaining_set_probe_created",
        ):
            expected, step = event["branch_id"], event["navigation_step"]
        else:
            continue
        in_range = 0 <= step < len(actions)
        action = actions[step] if in_range else {}
        observed = action.get("ghost_vp")
        act = action.get("act")
        equal = in_range and (
            (expected is None and act == 0)
            or (expected is not None and act == 4 and observed == expected)
        )
        checks.append({
            "event": kind, "step": step, "expected_branch": expected,
            "executed_act": act, "executed_ghost_vp": observed,
            "in_range": in_range, "equal": equal,
        })
    if not all(row["equal"] for row in checks):
        raise RuntimeError("V5.19 declared/executed action identity mismatch")
    return {"checks": len(checks), "all_equal": True, "rows": checks}


def main() -> None:
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
            break
    if run_dir is None:
        raise SystemExit("--run-dir is required")
    v517.RemainingSetRerankController = AlternativeFirstOptionSearchController
    v517.main()
    state = v517.V55._CONTROLLER
    if not isinstance(state, AlternativeFirstOptionSearchController):
        raise RuntimeError("V5.19 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-worker/5.19"
    summary["method_revision"] = (
        "alternative-first robust reversible probe with frozen-ETP "
        "remaining-set option elimination"
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
