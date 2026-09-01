#!/usr/bin/env python3
"""Fail-closed fixed OPP/Q training entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.revealskill_protocol import OUTPUT, verify_protocol  # noqa: E402


def main() -> int:
    verify_protocol()
    manifest = OUTPUT / "MF3ZP_SKILL_ROLLOUT_MANIFEST.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "MF3ZP_SKILL_ROLLOUT_COLLECTION_PASS":
        print(json.dumps({"status": "MF3ZP_SKILL_POLICY_TRAINING_NOT_AUTHORIZED", "checkpoint_generated": False}, indent=2))
        return 3
    print(json.dumps({"status": "MF3ZP_SKILL_POLICY_FIXED_TRAINING_INPUT_READY", "checkpoint_generated": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
