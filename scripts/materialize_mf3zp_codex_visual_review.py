#!/usr/bin/env python3
"""Validate and materialize the independent 80-event visual review."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.codex_visual_review import (  # noqa: E402
    materialize_visual_labels,
    seal_training_protocol,
)


def main() -> int:
    manifest = materialize_visual_labels()
    protocol = seal_training_protocol()
    print(json.dumps({
        "manifest": manifest,
        "training_protocol_status": protocol["status"],
    }, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
