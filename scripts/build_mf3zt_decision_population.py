#!/usr/bin/env python3
"""Run MF3ZT target preflight; build no population when support fails.

The script name follows the sealed execution order.  Its first action is the
mandatory target-support audit, and the fail result deliberately stops before
any population file is created.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.evidence_memory_probe import (  # noqa: E402
    ProbeAuditError,
    write_target_support_audit,
)
from revealnav_mf3.mf3zt_protocol import (  # noqa: E402
    ProtocolError,
    TARGET_AUDIT_PATH,
)


def main() -> int:
    try:
        value = write_target_support_audit()
    except (OSError, KeyError, TypeError, ValueError, ProbeAuditError, ProtocolError) as error:
        print(f"MF3ZT_TARGET_SUPPORT_AUDIT_ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "path": str(TARGET_AUDIT_PATH),
                "status": value["status"],
                "R2R_legal_rankable_target_rows": value["domain_support"]["R2R"]["legal_rankable_target_rows"],
                "RxR_legal_rankable_target_rows": value["domain_support"]["RxR"]["legal_rankable_target_rows"],
                "decision_population_written": False,
                "next_action": "STOP",
            },
            indent=2,
            sort_keys=True,
        )
    )
    # A completed scientific support audit is a successful execution even when
    # its pre-registered gate fails.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
