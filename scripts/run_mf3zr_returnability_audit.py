#!/usr/bin/env python3
"""Audit whether a frozen ETP-R1 returnability callback is available.

No simulator is started by the current revision.  The fixed 80-event source
contains native observation metadata, but no sealed callback that can execute
an option-anchor return.  Every option is therefore recorded explicitly as
``EXECUTION_UNAVAILABLE``; this is not an optimistic geometry witness.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.frozen_returnability import ReturnabilityStatus, unavailable_adapter  # noqa: E402
from revealnav_mf3.mf3zr_protocol import (  # noqa: E402
    PROTOCOL_PATH,
    REVIEW_SOURCE_PATH,
    RETURNABILITY_AUDIT_PATH,
    RETURNABILITY_PATH,
    RETURN_HORIZON,
    verify_protocol,
)


class ReturnabilityAuditError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReturnabilityAuditError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise ReturnabilityAuditError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ReturnabilityAuditError(f"stale partial artifact: {partial}")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def run() -> dict[str, object]:
    verify_protocol(PROTOCOL_PATH)
    source = _read_json(REVIEW_SOURCE_PATH)
    adapter = unavailable_adapter()
    records: list[dict[str, object]] = []
    event_count = 0
    for event in source.get("events", ()):
        if not isinstance(event, dict):
            raise ReturnabilityAuditError("malformed review source event")
        event_count += 1
        # State/option mappings contain only causal identity.  They are not
        # passed to a simulator because no callback is sealed.
        for identity in event.get("option_identities", ()):
            if not isinstance(identity, dict):
                raise ReturnabilityAuditError("malformed option identity")
            first = int(identity["first_seen_step"])
            last = int(identity["last_seen_step"])
            option_id = str(identity["option_id"])
            for step in range(first, last + 1):
                result = adapter.audit(
                    event_id=str(event["event_id"]),
                    option_id=option_id,
                    from_step=step,
                    anchor_checkpoint_id=str(identity["anchor_checkpoint_id"]),
                    state={"step": step, "causal_only": True},
                    option={"option_id": option_id, "candidate_id": str(identity["candidate_id"])},
                )
                records.append(result.as_mapping())
    if not records:
        raise ReturnabilityAuditError("no option identities available")
    _write_new(RETURNABILITY_PATH, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in records))
    status_counts = Counter(str(row["status"]) for row in records)
    audit = {
        "schema_version": "revealnav-mf3zr-returnability-audit/1",
        "status": "MF3ZR_RETURNABILITY_AUDIT_COMPLETE",
        "events": event_count,
        "option_records": len(records),
        "callback_available": adapter.available,
        "callback_status": "CONTROL_RETURNABILITY_UNAVAILABLE",
        "return_horizon": RETURN_HORIZON,
        "status_counts": dict(sorted(status_counts.items())),
        "attempted_count": sum(bool(row["attempted"]) for row in records),
        "success_count": sum(bool(row["success"]) for row in records),
        "controller_sha256": None,
        "geometry_only_shortcut": False,
        "teleport_or_pose_reset": False,
        "snapshot_counted_as_return": False,
        "qwen_calls": 0,
        "qwen_reads": 0,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }
    _write_new(RETURNABILITY_AUDIT_PATH, json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return audit


def main() -> int:
    try:
        result = run()
    except (OSError, KeyError, TypeError, ValueError, ReturnabilityAuditError, RuntimeError) as error:
        print(f"MF3ZR_RETURNABILITY_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
