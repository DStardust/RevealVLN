#!/usr/bin/env python3
"""Compact live monitor for the V5.16 method screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_16_method_screen"


def main() -> None:
    status_path = OUT / "RUN_STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    rows = []
    for path in sorted((OUT / "runs").glob("*/RUN_SUMMARY.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    funnels = [row.get("safety_funnel", {}) for row in rows]
    print(json.dumps({
        "status": status.get("status", "STARTING"),
        "completed": len(rows),
        "expected": 72,
        "passed": sum(row.get("status") == "PASS" for row in rows),
        "failed": len(status.get("failures", [])),
        "native_first_trials": sum(row.get("native_first_trials", 0) for row in funnels),
        "unanimous_returns": sum(row.get("unanimous_return_decisions", 0) for row in funnels),
        "ensemble_vetoes": sum(row.get("ensemble_disagreement_vetoes", 0) for row in funnels),
        "alternative_commits": sum(row.get("alternative_commits", 0) for row in funnels),
        "topology_restores": sum(row.get("topology_restores", 0) for row in funnels),
        "failures": status.get("failures", []),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
