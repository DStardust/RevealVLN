#!/usr/bin/env python3
"""K=3 global identity with a local executable candidate set."""

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
import r2r_safe_local_opp_worker_v5_8 as v58  # noqa: E402


class HybridCandidateFullOPPActionController(
    v58.SafeLocalFullOPPActionController
):
    """Establish identity globally, then score only local executable branches."""

    def _features(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ):
        if len(pilot._CURRENT_IDS) != 1 or len(pilot._LOCAL_FRONTIERS) != 1:
            raise RuntimeError("ETP graph identity hook is unavailable")
        global_current = {}
        for index, branch_id in enumerate(gmap_vp_ids[0]):
            if (
                index == 0 or branch_id is None
                or not bool(gmap_masks[0, index])
                or bool(gmap_visited_masks[0, index])
            ):
                continue
            global_current[str(branch_id)] = gmap_img_fts[0, index].detach()
        local_ids = set(global_current).intersection(pilot._LOCAL_FRONTIERS[0])
        local_current = {
            branch_id: global_current[branch_id]
            for branch_id in sorted(local_ids)
        }
        globally_persistent = set(
            self.candidate_tracker.update(global_current)
        )
        persistent_local = tuple(sorted(local_ids & globally_persistent))
        checkpoint_id = pilot._CURRENT_IDS[0]
        available = self.ledger.untried(checkpoint_id, persistent_local)
        self.ledger_suppressions += len(persistent_local) - len(available)

        self.navigation_prefixes += 1
        self.current_count_histogram[len(global_current)] += 1
        self.local_count_histogram[len(local_current)] += 1
        self.persistent_count_histogram[len(available)] += 1
        return local_current, available

    def candidate_funnel(self) -> dict:
        value = super().candidate_funnel()
        value["persistence_semantics"] = (
            "candidate identity present in K=3 consecutive ETP global-map "
            "prefixes, intersected with the current local executable set"
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
    v58.SafeLocalFullOPPActionController = HybridCandidateFullOPPActionController
    v58.main()
    state = v58.v57.v56.v55._CONTROLLER
    if not isinstance(state, HybridCandidateFullOPPActionController):
        raise RuntimeError("V5.9 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-full-opp-worker/5.9"
    summary["correctness_revision"] = (
        "global consecutive-prefix identity intersected with local execution "
        "and V5.8 base-action safety guards"
    )
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
