#!/usr/bin/env python3
"""Precompute a disjoint tail of T_X round 2 while round 1 is running."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_rxr_expansion_tx_gate as gate  # noqa: E402


def main() -> int:
    expected = gate.build_plan()
    observed = json.loads(gate.PLAN.read_text(encoding="utf-8"))
    if observed != expected:
        raise SystemExit("sealed T_X plan drift")
    plan_sha = hashlib.sha256(gate.PLAN.read_bytes()).hexdigest()
    event_ids = expected["eligible_event_ids"][-300:]
    rows = gate.execute_round("round2", event_ids, [0, 1], plan_sha)
    failures = [row for row in rows if row["returncode"] != 0]
    marker = gate.TX / "RXR_EXPANSION_TX_ROUND2_TAIL.json"
    gate.atomic_json(marker, {
        "revision": "rxr-expansion-tx-round2-tail/1",
        "status": "PASS" if not failures else "FAIL",
        "plan_sha256": plan_sha,
        "event_count": len(event_ids),
        "event_ids": event_ids,
        "failures": failures,
        "gpus": [0, 1],
    })
    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "events": len(event_ids), "failures": len(failures),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
