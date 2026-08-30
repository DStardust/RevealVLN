#!/usr/bin/env python3
"""Local-topology candidate collector for RxR V6.2."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_action_enabled_pilot_worker_v5 as pilot  # noqa: E402
import rxr_v6_1_counterfactual_worker as v61  # noqa: E402
import rxr_v6_counterfactual_worker as base  # noqa: E402


class LocalTopologyCandidateController(v61.BroadPersistentCandidateController):
    """Collect scoreable local branches; keep persistence for deployment."""

    @staticmethod
    def proposal_controls(
        current: dict, native: str | None, frontier_age: dict[str, int],
    ) -> tuple[str, ...]:
        if native is None or native not in current:
            return ()
        alternatives = sorted(
            (branch for branch in current if branch != native),
            key=lambda branch: (-int(frontier_age[branch]), branch),
        )[:3]
        return (native, *alternatives) if alternatives else ()

    def _initial_decision(self, current, persistent, native_branch):
        del persistent
        controls = self.proposal_controls(
            current, native_branch, pilot._LOCAL_FRONTIERS[0]
        )
        return super()._initial_decision(current, controls, native_branch)


def main() -> int:
    base.V6CounterfactualController = LocalTopologyCandidateController
    return base.run()


if __name__ == "__main__":
    raise SystemExit(main())
