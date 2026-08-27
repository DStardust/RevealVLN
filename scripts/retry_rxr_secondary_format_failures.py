#!/usr/bin/env python3
"""Retry only fail-closed language prefixes with provider/JSON format errors."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
MULTIBRANCH = BASE / "multibranch"
GATE = MULTIBRANCH / "RXR_SECONDARY_CAUSAL_PREFIX_LANGUAGE_GATE.json"
RESULTS = MULTIBRANCH / "prefix_language_results"
AUDIT = MULTIBRANCH / "prefix_language_format_retry_v1"
PLAN = AUDIT / "RXR_SECONDARY_FORMAT_RETRY_PLAN.json"
REPORT = AUDIT / "RXR_SECONDARY_FORMAT_RETRY_REPORT.json"
LOG = AUDIT / "RXR_SECONDARY_FORMAT_RETRY.log"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def main() -> int:
    if PLAN.exists() or REPORT.exists() or AUDIT.is_symlink():
        raise RuntimeError("format retry has already been planned or executed")
    before = json.loads(GATE.read_text())
    if before.get("status") != "COMPLETE_CAUSAL_CONTROLS_REQUIRED":
        raise RuntimeError("original language gate is not complete")
    failed_events = {
        row["event_id"] for row in before["events"]
        if row["status"] == "CAUSAL_LANGUAGE_K3_FAIL"
    }
    selected = []
    for event_id in sorted(failed_events):
        for path in sorted((RESULTS / event_id).glob("P*.json")):
            value = json.loads(path.read_text())
            format_failure = (
                value.get("parse_error") is not None
                or value.get("status") == "PROVIDER_ERROR_FAIL_CLOSED"
            )
            if not format_failure:
                continue
            relative = path.relative_to(RESULTS)
            selected.append({
                "event_id": event_id,
                "prefix_file": str(relative),
                "original_path": str(path.relative_to(ROOT)),
                "audit_path": str((AUDIT / "attempt_001" / relative).relative_to(ROOT)),
                "original_bytes": path.stat().st_size,
                "original_sha256": sha256_file(path),
                "original_status": value["status"],
                "original_parse_error": value.get("parse_error"),
                "original_provider_json_abort_failures": value.get(
                    "provider_json_abort_failures", []
                ),
            })
    if not selected:
        raise RuntimeError("no failed-event format errors are eligible for retry")
    plan = {
        "schema_version": "revealnav-secondary-format-retry-plan/1",
        "status": "PLANNED_IDENTICAL_REQUEST_RETRY_ONLY",
        "original_gate_sha256": sha256_file(GATE),
        "original_counts": before["counts"],
        "failed_events_with_retry": sorted({row["event_id"] for row in selected}),
        "retry_prefix_count": len(selected),
        "semantic_invalid_prefixes_retried": 0,
        "prompt_or_threshold_change": False,
        "records": selected,
    }
    atomic_json(PLAN, plan)
    for row in selected:
        source = ROOT / row["original_path"]
        destination = ROOT / row["audit_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or not source.is_file() or source.is_symlink():
            raise RuntimeError("unsafe retry evidence move")
        os.replace(source, destination)

    with LOG.open("w") as log:
        code = subprocess.run(
            [
                sys.executable,
                "scripts/run_rxr_secondary_causal_prefix_language.py",
                "--execute", "--workers", "16",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code:
        raise RuntimeError(f"format retry language gate failed with code {code}")
    after = json.loads(GATE.read_text())
    if after.get("status") != "COMPLETE_CAUSAL_CONTROLS_REQUIRED":
        raise RuntimeError("retried language gate is incomplete")
    after_by_event = {row["event_id"]: row for row in after["events"]}
    reissued = []
    for row in selected:
        path = RESULTS / row["prefix_file"]
        if path.is_file():
            value = json.loads(path.read_text())
            reissued.append({
                "event_id": row["event_id"],
                "prefix_file": row["prefix_file"],
                "status": value["status"],
                "effective_language_closed": value[
                    "effective_language_closed"
                ],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    recovered = sorted(
        event_id for event_id in plan["failed_events_with_retry"]
        if after_by_event[event_id]["status"]
        == "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED"
    )
    report = {
        "schema_version": "revealnav-secondary-format-retry-report/1",
        "status": "FORMAT_ONLY_RETRY_COMPLETE",
        "plan_sha256": sha256_file(PLAN),
        "before": {
            "gate_sha256": plan["original_gate_sha256"],
            "counts": before["counts"],
        },
        "after": {
            "gate_sha256": sha256_file(GATE),
            "counts": after["counts"],
        },
        "retried_prefixes": len(selected),
        "reissued_prefixes": len(reissued),
        "recovered_event_ids": recovered,
        "recovered_events": len(recovered),
        "semantic_invalid_prefixes_retried": 0,
        "prompt_or_threshold_change": False,
        "original_responses_preserved": True,
        "reissued": reissued,
    }
    atomic_json(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "retried_prefixes": report["retried_prefixes"],
        "recovered_events": report["recovered_events"],
        "before_pass": before["counts"]["language_k3_pass"],
        "after_pass": after["counts"]["language_k3_pass"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
