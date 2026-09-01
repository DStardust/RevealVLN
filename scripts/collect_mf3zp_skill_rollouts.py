#!/usr/bin/env python3
"""Validate authorization for bounded high-level skill rollouts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.revealskill_protocol import OUTPUT, ROLLOUT_HORIZON, verify_protocol  # noqa: E402


def validate_skill_rollout(row: dict[str, object]) -> None:
    if row.get("controller_frozen") is not True or row.get("teleport") is not False:
        raise ValueError("skill rollout must use frozen controller without teleportation")
    if row.get("public_split") is not False:
        raise ValueError("public split rollout forbidden")
    skills = row.get("high_level_skills")
    if not isinstance(skills, list) or not 1 <= len(skills) <= ROLLOUT_HORIZON:
        raise ValueError("bounded high-level skill horizon violated")
    intended = row.get("intended_skill_index")
    changed = row.get("changed_skill_indices")
    if type(intended) is not int or changed != [intended]:
        raise ValueError("bounded counterfactual must change exactly the intended high-level skill")
    if row.get("frozen_continuation_sha256") != row.get("reference_continuation_sha256"):
        raise ValueError("frozen continuation drift after skill boundary")


def main() -> int:
    verify_protocol()
    ree = OUTPUT / "MF3ZP_REE_LEARNABILITY_RESULT.json"
    if not ree.is_file() or json.loads(ree.read_text()).get("status") != "MF3ZP_REE_LEARNABILITY_PASS":
        print(json.dumps({"status": "MF3ZP_SKILL_ROLLOUTS_NOT_AUTHORIZED", "reason": "ree_learnability_not_passed"}, indent=2))
        return 3
    print(json.dumps({"status": "MF3ZP_SKILL_ROLLOUT_INTERFACE_READY", "horizon": ROLLOUT_HORIZON}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
