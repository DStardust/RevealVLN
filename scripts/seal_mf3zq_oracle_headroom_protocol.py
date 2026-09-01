#!/usr/bin/env python3
"""Seal the MF3ZQ exploratory protocol exactly once."""

from __future__ import annotations

import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.oracle_headroom_protocol import ProtocolError, seal_protocol, sha256_file  # noqa: E402


def main() -> int:
    try:
        value = seal_protocol()
    except (OSError, KeyError, TypeError, ValueError, ProtocolError) as error:
        print(f"MF3ZQ_PROTOCOL_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    from revealnav_mf3.oracle_headroom_protocol import PROTOCOL_PATH
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_commit": value["source_commit"],
        "events": value["population"]["events"],
        "scenes": value["population"]["raw_scene_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
