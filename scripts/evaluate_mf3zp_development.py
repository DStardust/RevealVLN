#!/usr/bin/env python3
"""Fail-closed development-evaluation authorization check."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.revealskill_protocol import OUTPUT, verify_protocol  # noqa: E402


def main() -> int:
    verify_protocol()
    checkpoint = OUTPUT / "gates/MF3ZP_REVEALSKILL_MODEL.pt"
    ree = OUTPUT / "MF3ZP_REE_LEARNABILITY_RESULT.json"
    oracle = OUTPUT / "MF3ZP_ORACLE_HEADROOM_RESULT.json"
    ready = checkpoint.is_file() and ree.is_file() and oracle.is_file()
    print(json.dumps({
        "status": "MF3ZP_DEVELOPMENT_EVALUATION_READY" if ready else "MF3ZP_DEVELOPMENT_EVALUATION_NOT_AUTHORIZED",
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }, indent=2))
    return 0 if ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
