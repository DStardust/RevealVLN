#!/usr/bin/env python3
"""Hybrid K=3 adapter that scores the frozen ETP action as a control."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_hybrid_candidate_opp_worker_v5_9 as v59  # noqa: E402


V56_INITIAL_DECISION = (
    v59.v58.v57.v56.FullOPPActionController._initial_decision
)


class NativeControlFullOPPActionController(
    v59.HybridCandidateFullOPPActionController
):
    """Intervene only after scoring ETP's native action in the same head."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.global_current = {}
        self.local_persistent: tuple[str, ...] = ()
        self.native_control_comparisons = 0

    def _features(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ):
        if len(pilot._CURRENT_IDS) != 1 or len(pilot._LOCAL_FRONTIERS) != 1:
            raise RuntimeError("ETP graph identity hook is unavailable")
        self.global_current = {}
        for index, branch_id in enumerate(gmap_vp_ids[0]):
            if (
                index == 0 or branch_id is None
                or not bool(gmap_masks[0, index])
                or bool(gmap_visited_masks[0, index])
            ):
                continue
            self.global_current[str(branch_id)] = gmap_img_fts[0, index].detach()
        local_ids = set(self.global_current).intersection(
            pilot._LOCAL_FRONTIERS[0]
        )
        local_current = {
            branch_id: self.global_current[branch_id]
            for branch_id in sorted(local_ids)
        }
        globally_persistent = set(
            self.candidate_tracker.update(self.global_current)
        )
        persistent_local = tuple(sorted(local_ids & globally_persistent))
        checkpoint_id = pilot._CURRENT_IDS[0]
        self.local_persistent = self.ledger.untried(
            checkpoint_id, persistent_local
        )
        self.ledger_suppressions += (
            len(persistent_local) - len(self.local_persistent)
        )

        self.navigation_prefixes += 1
        self.current_count_histogram[len(self.global_current)] += 1
        self.local_count_histogram[len(local_current)] += 1
        self.persistent_count_histogram[len(self.local_persistent)] += 1
        return local_current, self.local_persistent

    def _evaluate(self, current, persistent):
        value = super()._evaluate(current, persistent)
        native = getattr(self, "_native_control_branch", None)
        if (
            value is not None and value["action"] == "explore"
            and value["macro"].branch_id == native
            and native not in self.local_persistent
        ):
            value["action"] = "follow"
            value["reason"] = "native_control_is_not_an_exploration_option"
        return value

    def _initial_decision(self, current, persistent, native_branch):
        if len(persistent) < 2:
            return None
        if native_branch is None:
            self.stop_suppressions += 1
            self.record(
                "opp_base_action_safety_suppression",
                reason="native_ETP_STOP_must_not_be_overridden",
                persistent_branch_count=len(persistent),
            )
            return None
        if native_branch not in self.global_current:
            self.native_outside_candidate_suppressions += 1
            self.record(
                "opp_base_action_safety_suppression",
                reason="native_ETP_action_is_not_an_unvisited_candidate",
                native_base_branch=native_branch,
                persistent_branch_count=len(persistent),
            )
            return None
        scoring = dict(current)
        scoring[native_branch] = self.global_current[native_branch]
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
            return None
        self.safe_decision_prefixes += 1
        self.native_control_comparisons += 1
        self._native_control_branch = native_branch
        try:
            return V56_INITIAL_DECISION(self, scoring, controls, native_branch)
        finally:
            self._native_control_branch = None

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value["native_control_comparisons"] = self.native_control_comparisons
        value["safety_contract"] = (
            "never override ETP STOP or a visited/backtrack action; score the "
            "native unvisited ETP action beside K=3 local options; keep total "
            "width within the 2-4 training support"
        )
        return value


def main() -> None:
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
            break
    if run_dir is None:
        raise SystemExit("--run-dir is required")
    v59.HybridCandidateFullOPPActionController = (
        NativeControlFullOPPActionController
    )
    v59.main()
    state = v59.v58.v57.v56.v55._CONTROLLER
    if not isinstance(state, NativeControlFullOPPActionController):
        raise RuntimeError("V5.10 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-full-opp-worker/5.10"
    summary["correctness_revision"] = (
        "score frozen ETP native action as the control before any hybrid K=3 "
        "local intervention"
    )
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
