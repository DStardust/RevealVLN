#!/usr/bin/env python3
"""Seal immutable MF3ZU RxR-only feasibility before downstream material."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zu_protocol import (  # noqa: E402
    PROTOCOL_PATH,
    ProtocolError,
    seal_protocol,
)


def main() -> int:
    try:
        value = seal_protocol()
    except (OSError, KeyError, TypeError, ValueError, ProtocolError) as error:
        print(f"MF3ZU_PROTOCOL_SEAL_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "path": str(PROTOCOL_PATH),
                "revision": value["revision"],
                "status": value["status"],
                "source_commit": value["source_commit"],
                "seal_commit": value["seal_commit"],
                "dataset": value["scope"]["dataset"],
                "R2R_in_scope": value["revision_relationship"]["R2R_in_scope"],
                "next_action": "BUILD_TARGET_BLIND_RXR_POPULATION",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
