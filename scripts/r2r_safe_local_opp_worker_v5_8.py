#!/usr/bin/env python3
"""Training-aligned local K=3 adapter with base-action safety guards."""

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
import r2r_full_opp_worker_v5_7 as v57  # noqa: E402


class SafeLocalFullOPPActionController(
    v57.ConsecutiveFullOPPActionController
):
    """Use local causal candidates and preserve unsupported ETP actions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stop_suppressions = 0
        self.native_outside_candidate_suppressions = 0
        self.candidate_width_suppressions = 0
        self.safe_decision_prefixes = 0

    def _features(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ):
        if len(pilot._CURRENT_IDS) != 1 or len(pilot._LOCAL_FRONTIERS) != 1:
            raise RuntimeError("ETP graph identity hook is unavailable")
        local_frontier = pilot._LOCAL_FRONTIERS[0]
        current = {}
        for index, branch_id in enumerate(gmap_vp_ids[0]):
            if (
                index == 0 or branch_id is None
                or not bool(gmap_masks[0, index])
                or bool(gmap_visited_masks[0, index])
            ):
                continue
            branch_id = str(branch_id)
            if branch_id in local_frontier:
                current[branch_id] = gmap_img_fts[0, index].detach()
        persistent = self.candidate_tracker.update(current)
        checkpoint_id = pilot._CURRENT_IDS[0]
        available = self.ledger.untried(checkpoint_id, persistent)
        self.ledger_suppressions += len(persistent) - len(available)

        self.navigation_prefixes += 1
        self.current_count_histogram[len(current)] += 1
        self.local_count_histogram[len(current)] += 1
        self.persistent_count_histogram[len(available)] += 1
        return current, available

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
        if native_branch not in persistent:
            self.native_outside_candidate_suppressions += 1
            self.record(
                "opp_base_action_safety_suppression",
                reason="native_ETP_action_absent_from_scored_candidate_set",
                native_base_branch=native_branch,
                persistent_branch_count=len(persistent),
            )
            return None
        if len(persistent) > 4:
            self.candidate_width_suppressions += 1
            self.record(
                "opp_base_action_safety_suppression",
                reason="candidate_width_outside_training_support",
                native_base_branch=native_branch,
                persistent_branch_count=len(persistent),
            )
            return None
        self.safe_decision_prefixes += 1
        return super()._initial_decision(current, persistent, native_branch)

    def safety_funnel(self) -> dict:
        return {
            "stop_suppressions": self.stop_suppressions,
            "native_outside_candidate_suppressions": (
                self.native_outside_candidate_suppressions
            ),
            "candidate_width_suppressions": self.candidate_width_suppressions,
            "safe_decision_prefixes": self.safe_decision_prefixes,
            "maximum_supported_candidate_width": 4,
            "safety_contract": (
                "never override ETP STOP; require native action in the same "
                "K=3 local candidate set; reject widths outside training support"
            ),
        }

    def candidate_funnel(self) -> dict:
        value = super().candidate_funnel()
        value["persistence_semantics"] = (
            "candidate identity present in K=3 consecutive ETP navigation "
            "prefixes within the current local automatic-front-end set"
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
    v57.ConsecutiveFullOPPActionController = SafeLocalFullOPPActionController
    v57.main()
    state = v57.v56.v55._CONTROLLER
    if not isinstance(state, SafeLocalFullOPPActionController):
        raise RuntimeError("V5.8 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-full-opp-worker/5.8"
    summary["correctness_revision"] = (
        "training-aligned local consecutive-prefix candidates plus base-action "
        "support guards"
    )
    summary["safety_funnel"] = state.safety_funnel()
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
