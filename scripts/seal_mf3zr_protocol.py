#!/usr/bin/env python3
"""Seal the MF3ZR support protocol exactly once, before support results."""

from __future__ import annotations

import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zr_protocol import ProtocolError, PROTOCOL_PATH, seal_protocol  # noqa: E402


def main() -> int:
    try:
        value = seal_protocol()
    except (OSError, KeyError, TypeError, ValueError, ProtocolError) as error:
        print(f"MF3ZR_PROTOCOL_SEAL_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"path": str(PROTOCOL_PATH), "status": value["status"], "seal_commit": value["seal_commit"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
