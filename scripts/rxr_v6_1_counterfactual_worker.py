#!/usr/bin/env python3
"""Broad persistent-branch candidate collector for RxR V6.1."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import rxr_v6_counterfactual_worker as base  # noqa: E402
from revealnav_mf2r4 import StateConditionedReturnExecutor  # noqa: E402


class BroadPersistentCandidateController(base.V6CounterfactualController):
    """Create native-first trials independently of the old absolute-Q gate."""

    @staticmethod
    def ranked_alternative(value: dict, controls, native: str):
        rows = sorted(
            (
                (float(value["probabilities"][index]), branch_id)
                for index, branch_id in enumerate(controls)
                if branch_id != native
            ),
            key=lambda row: (-row[0], row[1]),
        )
        return rows[0][1] if rows else None

    def _initial_decision(self, current, persistent, native_branch):
        pending, consumed = self._consume_pending_alternative()
        if consumed:
            return pending
        self.global_rows.append((
            self.latest_history.detach(), dict(self.global_current)
        ))
        self.temporal_prefixes_cached += 1
        value, controls = self._aligned_value(
            current, persistent, native_branch
        )
        if value is None:
            return None
        alternative = self.ranked_alternative(value, controls, native_branch)
        if (
            alternative is None
            or self.step >= int(pilot._TRAINER.max_len) - 3
        ):
            return None
        checkpoint_id = pilot._CURRENT_IDS[0]
        # This ticket authorizes a train-only counterfactual transaction; it is
        # not a learned benefit and never enters the V6 input or target.
        transaction_ticket = self.ledger.opv_threshold + 1.0
        self.ledger.register(checkpoint_id, controls)
        if not self.ledger.authorize_branch(
            checkpoint_id, native_branch, transaction_ticket
        ):
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
        self.retained_alternative_source = "causal_REE_highest_non_native"
        self.trial_preservation_gain = transaction_ticket
        self.native_first_trials += 1
        self.checkpointed_excursions += 1
        self.record(
            "v6_1_broad_native_trial_created",
            checkpoint_id=checkpoint_id,
            native_branch=native_branch,
            retained_alternative=alternative,
            proposal_rule="highest_causal_REE_probability_non_native",
            persistent_branch_count=len(controls),
            base_action_overridden=False,
            transaction_ticket_not_model_input=True,
        )
        self.global_rows.clear()
        return native_branch


def main() -> int:
    base.V6CounterfactualController = BroadPersistentCandidateController
    return base.run()


if __name__ == "__main__":
    raise SystemExit(main())
