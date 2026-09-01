#!/usr/bin/env python3
"""Verify descendant ancestry plus byte-exact MF3ZP scientific inputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.single_expert_dec_scout import (  # noqa: E402
    verify_closure,
    verify_scout_protocol,
)


if __name__ == "__main__":
    closure = verify_closure()
    scout = verify_scout_protocol()
    print(json.dumps({
        "status": "MF3ZP_REPRODUCIBILITY_CLOSURE_PASS",
        "base_review_commit": closure["base_review_commit"],
        "scout_protocol_status": scout["status"],
        "public_split_access": scout["public_split_access"],
    }, indent=2, sort_keys=True))
