#!/usr/bin/env python3
"""V5.10 native-control adapter with complete temporal prefix history."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r2r_native_control_opp_worker_v5_10 as v510  # noqa: E402


class TemporalNativeControlFullOPPActionController(
    v510.NativeControlFullOPPActionController
):
    """Retain every causal prefix before the K=3 decision gate opens."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temporal_prefixes_retained_before_gate = 0

    def _initial_decision(self, current, persistent, native_branch):
        controls = tuple(dict.fromkeys((*persistent, native_branch)))
        safe_to_evaluate = (
            len(persistent) >= 2
            and native_branch is not None
            and native_branch in self.global_current
            and len(controls) <= 4
        )
        if not safe_to_evaluate:
            self.rows.append((self.latest_history, current))
            self.temporal_prefixes_retained_before_gate += 1
        return super()._initial_decision(current, persistent, native_branch)

    def safety_funnel(self) -> dict:
        value = super().safety_funnel()
        value["temporal_prefixes_retained_before_gate"] = (
            self.temporal_prefixes_retained_before_gate
        )
        value["temporal_history_contract"] = (
            "append every causal navigation prefix exactly once; K=3 gates "
            "the decision, not temporal-head observation history"
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
    v510.NativeControlFullOPPActionController = (
        TemporalNativeControlFullOPPActionController
    )
    v510.main()
    state = v510.v59.v58.v57.v56.v55._CONTROLLER
    if not isinstance(state, TemporalNativeControlFullOPPActionController):
        raise RuntimeError("V5.11 controller was not installed")
    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "revealnav-r2r-full-opp-worker/5.11"
    summary["correctness_revision"] = (
        "retain every causal prefix for the frozen temporal heads before "
        "applying the unchanged V5.10 K=3 native-control decision gate"
    )
    summary["safety_funnel"] = state.safety_funnel()
    part = summary_path.with_name(summary_path.name + ".part")
    part.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(part, summary_path)


if __name__ == "__main__":
    main()
