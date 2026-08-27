#!/usr/bin/env python3
"""Validate queue50 primary multi-view inputs before MLLM use."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
INPUT = ROOT / (
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/"
    "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
)
OUT = INPUT.with_name("CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS_ACCEPTANCE.json")
EXPECTED_INPUT_SHA = (
    "6b70a70e5eb1e25f9522b30209eb56dc2efbf6457377a1aabefdeca6886aee72"
)


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def valid_file(record):
    path = ROOT / record["path"]
    return (path.is_file() and not path.is_symlink()
            and ROOT.resolve() in path.resolve().parents
            and path.stat().st_size == record["bytes"]
            and sha(path) == record["sha256"])


def main():
    failures = []
    if (not INPUT.is_file() or INPUT.is_symlink()
            or sha(INPUT) != EXPECTED_INPUT_SHA):
        failures.append("input SHA/safety")
    value = json.loads(INPUT.read_text())
    events = value.get("events", [])
    media = value.get("media_manifest", [])
    if (value.get("event_count") != 50 or len(events) != 50
            or [row.get("queue_order") for row in events] != list(range(50))
            or len({row.get("episode_id") for row in events}) != 50
            or len({row.get("event_id") for row in events}) != 50):
        failures.append("50-event ordered closure")
    for event in events:
        if set(event.get("positions", {})) != {"A", "Q", "D"}:
            failures.append("position roles: " + event["event_id"])
            continue
        for role in ("A", "Q", "D"):
            position = event["positions"][role]
            views = position.get("views", [])
            if (len(views) != 12
                    or [row.get("view_id") for row in views]
                    != ["%s_V%02d" % (role, index)
                        for index in range(12)]
                    or [row.get("relative_yaw_deg") for row in views]
                    != [0.0, 30.0, 60.0, 90.0, 120.0, 150.0,
                        180.0, -150.0, -120.0, -90.0, -60.0, -30.0]
                    or not valid_file(position["contact_sheet"])):
                failures.append("role contract: %s/%s" %
                                (event["event_id"], role))
            if not all(valid_file(record) for record in views):
                failures.append("role media: %s/%s" %
                                (event["event_id"], role))
        context = event.get("chronological_context_frames", [])
        prefixes = [row.get("prefix_index") for row in context]
        if (not context or prefixes != sorted(set(prefixes))
                or not all(valid_file(record) for record in context)):
            failures.append("context contract: " + event["event_id"])
    if (len(media) != 1950 or value.get("media_file_count") != 1950
            or len({row.get("path") for row in media}) != 1950
            or value.get("media_total_bytes") !=
            sum(row.get("bytes", -1) for row in media)
            or not all(valid_file(record) for record in media)):
        failures.append("media aggregate/integrity")
    if (value.get("network_calls_made") != 0
            or value.get("branch_labels_created") != 0
            or value.get("geometry_verified_candidates") != 0
            or value.get("human_labels_created") != 0
            or value.get("training_authorized") is not False):
        failures.append("scope boundary")
    output = {
        "revision": "cr5-queue50-primary-multiview-acceptance/1",
        "status": "PASS" if not failures else "FAIL",
        "input_path": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha(INPUT),
        "event_count": len(events),
        "panorama_view_count": len(events) * 36,
        "media_file_count": len(media),
        "all_referenced_media_project_local_regular_hash_verified":
            not any("media" in item or "context" in item
                    or "role" in item for item in failures),
        "branch_labels_created": 0,
        "geometry_labels_created": 0,
        "human_labels_created": 0,
        "training_authorized": False,
        "failures": failures,
    }
    atomic_json(OUT, output)
    print(json.dumps({**output, "output": str(OUT.relative_to(ROOT)),
                      "output_sha256": sha(OUT)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
