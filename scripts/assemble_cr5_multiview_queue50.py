#!/usr/bin/env python3
"""Assemble immutable raw and retry branch proposals for queue50."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
RAW_RUN = BASE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_RUN.json"
RETRY_RUN = BASE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_RETRY_RUN.json"
OUT = BASE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_ACCEPTED_RUN.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    raw = load(RAW_RUN)
    retry = load(RETRY_RUN)
    if raw.get("event_count") != 50 or len(raw.get("results", [])) != 50:
        raise SystemExit("raw run is not the frozen 50-event run")
    retry_rows = {row["event_id"]: row for row in retry.get("results", [])}
    rows = []
    failures = []
    for raw_row in raw["results"]:
        event_id = raw_row["event_id"]
        chosen = raw_row
        source = "RAW_FIRST_PASS"
        if raw_row["status"] != "VALID_MLLM_PROPOSAL":
            candidate = retry_rows.get(event_id)
            if candidate and candidate["status"] == "VALID_MLLM_PROPOSAL":
                chosen = candidate
                source = "REAL_PROVIDER_RETRY"
            else:
                failures.append(event_id)
        path = ROOT / chosen["path"]
        if not path.is_file() or path.is_symlink():
            raise SystemExit("unsafe proposal path: " + str(path))
        if sha256_file(path) != chosen["sha256"]:
            raise SystemExit("proposal SHA drift: " + event_id)
        payload = load(path)
        if payload.get("status") != "VALID_MLLM_PROPOSAL":
            failures.append(event_id)
        rows.append({
            "event_id": event_id,
            "accepted_source": source,
            "accepted_proposal_path": chosen["path"],
            "accepted_proposal_sha256": chosen["sha256"],
            "raw_first_pass_status": raw_row["status"],
            "raw_first_pass_path": raw_row["path"],
            "raw_first_pass_sha256": raw_row["sha256"],
            "human_reviewed": False,
            "training_label": False,
        })
    failures = sorted(set(failures))
    output = {
        "manifest": "MF2-CR5 queue50 accepted offline multiview proposals",
        "revision": "cr5-queue50-multiview-accepted/1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "sources": {
            "raw_run": {
                "path": str(RAW_RUN.relative_to(ROOT)),
                "sha256": sha256_file(RAW_RUN),
            },
            "retry_run": {
                "path": str(RETRY_RUN.relative_to(ROOT)),
                "sha256": sha256_file(RETRY_RUN),
            },
        },
        "event_count": len(rows),
        "accepted_count": len(rows) - len(failures),
        "failed_event_ids": failures,
        "events": rows,
        "offline_annotation_only": True,
        "geometry_labels_created": 0,
        "online_causal_labels_created": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "accepted": output["accepted_count"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
