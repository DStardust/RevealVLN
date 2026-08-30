#!/usr/bin/env python3
"""Print one compact expanded-Q training status snapshot."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v5_1"


def main() -> int:
    progress_path = OUT / "RXR_BRANCH_EXCURSION_Q_PROGRESS_V5_1.json"
    result_path = OUT / "RXR_BRANCH_EXCURSION_Q_COMPARISON_V5_1.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "status": "NOT_STARTED", "completed": 0, "total": 6, "failed": 0,
    }
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    completed = []
    for path in sorted(OUT.glob("*_seed_*/result.json")):
        row = json.loads(path.read_text())
        completed.append({
            "run": path.parent.name,
            "epochs": len(row["history"]),
            "overall_regret": row["metrics"]["all"]["mean_action_regret"],
            "human_regret": row["metrics"]["primary_human"]["mean_action_regret"],
        })
    print(json.dumps({
        "state": progress["status"],
        "completed": progress["completed"],
        "total": progress["total"],
        "failed": progress["failed"],
        "completed_runs": completed,
        "final_gate": None if result is None else result["status"],
        "passing_variants": [] if result is None else result["passing_variants"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
