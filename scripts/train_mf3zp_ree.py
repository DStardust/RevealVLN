#!/usr/bin/env python3
"""Fail-closed entrypoint for fixed Reveal/Expiry estimator training."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.revealskill_protocol import OUTPUT, verify_protocol  # noqa: E402


def main() -> int:
    verify_protocol()
    oracle = OUTPUT / "MF3ZP_ORACLE_HEADROOM_RESULT.json"
    if not oracle.is_file() or json.loads(oracle.read_text()).get("status") != "MF3ZP_ORACLE_HEADROOM_PASS":
        print(json.dumps({
            "status": "MF3ZP_REE_TRAINING_NOT_AUTHORIZED",
            "reason": "oracle_headroom_not_passed",
            "checkpoint_generated": False,
            "public_split_access": False,
        }, indent=2))
        return 3
    print(json.dumps({
        "status": "MF3ZP_REE_TRAINING_DATA_INTERFACE_REQUIRED",
        "reason": "authorized only after versioned oracle rollout and label artifacts exist",
        "checkpoint_generated": False,
    }, indent=2))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
