#!/usr/bin/env python3
"""Finalize and verify the immutable MF3ZT target-support failure result."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.evidence_memory_probe import (  # noqa: E402
    ProbeAuditError,
    verify_result,
    write_fail_result,
)
from revealnav_mf3.mf3zt_protocol import ProtocolError, RESULT_PATH  # noqa: E402


def main() -> int:
    try:
        value = write_fail_result()
        verified = verify_result()
        if verified != value:
            raise ProbeAuditError("MF3ZT result changed during verification")
    except (OSError, KeyError, TypeError, ValueError, ProbeAuditError, ProtocolError) as error:
        print(f"MF3ZT_RESULT_AUDIT_ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "path": str(RESULT_PATH),
                "status": value["status"],
                "final_pass_fail": value["final_pass_fail"],
                "scientific_evidence_about_memory": value["scientific_evidence_about_memory"],
                "next_action": "STOP",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
