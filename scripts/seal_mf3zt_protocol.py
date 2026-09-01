#!/usr/bin/env python3
"""Seal the MF3ZT protocol once, before the target-support result."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zt_protocol import (  # noqa: E402
    PROTOCOL_PATH,
    ProtocolError,
    seal_protocol,
)


def main() -> int:
    try:
        value = seal_protocol()
    except (OSError, KeyError, TypeError, ValueError, ProtocolError) as error:
        print(f"MF3ZT_PROTOCOL_SEAL_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "path": str(PROTOCOL_PATH),
                "status": value["status"],
                "source_commit": value["source_commit"],
                "seal_commit": value["seal_commit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
