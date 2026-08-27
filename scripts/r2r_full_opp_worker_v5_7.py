#!/usr/bin/env python3
"""V5.7 R2R adapter with RxR-aligned consecutive-prefix candidates."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import r2r_full_opp_worker_v5_6 as v56  # noqa: E402
from revealnav_mf2r4.temporal_candidates import (  # noqa: E402
    ConsecutiveCandidateTracker,
)


class ConsecutiveFullOPPActionController(v56.FullOPPActionController):
    """Keep V5.6 action order while correcting R2R candidate persistence."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.candidate_tracker = ConsecutiveCandidateTracker(
            self.fusion.persistence_k
        )
        self.navigation_prefixes = 0
        self.current_count_histogram: Counter[int] = Counter()
        self.local_count_histogram: Counter[int] = Counter()
        self.persistent_count_histogram: Counter[int] = Counter()

    def _features(
        self, gmap_vp_ids, gmap_img_fts, gmap_masks, gmap_visited_masks,
    ):
        if len(pilot._CURRENT_IDS) != 1 or len(pilot._LOCAL_FRONTIERS) != 1:
            raise RuntimeError("ETP graph identity hook is unavailable")
        current = {}
        for index, branch_id in enumerate(gmap_vp_ids[0]):
            if (
                index == 0 or branch_id is None
                or not bool(gmap_masks[0, index])
                or bool(gmap_visited_masks[0, index])
            ):
                continue
            current[str(branch_id)] = gmap_img_fts[0, index].detach()
        persistent = self.candidate_tracker.update(current)
        checkpoint_id = pilot._CURRENT_IDS[0]
        available = self.ledger.untried(checkpoint_id, persistent)
        self.ledger_suppressions += len(persistent) - len(available)

        local_ids = set(current).intersection(pilot._LOCAL_FRONTIERS[0])
        self.navigation_prefixes += 1
        self.current_count_histogram[len(current)] += 1
        self.local_count_histogram[len(local_ids)] += 1
        self.persistent_count_histogram[len(available)] += 1
        return current, available

    def candidate_funnel(self) -> dict:
        def encoded(counter: Counter[int]) -> dict[str, int]:
            return {str(key): counter[key] for key in sorted(counter)}

        return {
            "persistence_semantics": (
                "candidate identity present in K=3 consecutive ETP navigation "
                "prefixes over the global unvisited candidate set"
            ),
            "navigation_prefixes": self.navigation_prefixes,
            "current_candidate_count_histogram": encoded(
                self.current_count_histogram
            ),
            "local_candidate_count_histogram": encoded(
                self.local_count_histogram
            ),
            "persistent_candidate_count_histogram": encoded(
                self.persistent_count_histogram
            ),
            "prefixes_with_two_current": sum(
                count for width, count in self.current_count_histogram.items()
                if width >= 2
            ),
            "prefixes_with_two_local": sum(
                count for width, count in self.local_count_histogram.items()
                if width >= 2
            ),
            "prefixes_with_two_persistent": sum(
                count for width, count in self.persistent_count_histogram.items()
                if width >= 2
            ),
        }


def main() -> None:
    run_dir = None
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--run-dir":
            run_dir = Path(sys.argv[index + 1]).resolve()
            break
    if run_dir is None:
        raise SystemExit("--run-dir is required")
    v56.FullOPPActionController = ConsecutiveFullOPPActionController
    v56.main()
    state = v56.v55._CONTROLLER
    if not isinstance(state, ConsecutiveFullOPPActionController):
        raise RuntimeError("V5.7 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-full-opp-worker/5.7"
    summary["correctness_revision"] = (
        "RxR-aligned consecutive-prefix global candidate persistence"
    )
    summary["candidate_funnel"] = state.candidate_funnel()
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
