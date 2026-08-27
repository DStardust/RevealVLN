#!/usr/bin/env python3
"""Freeze the candidates recovered only by target-route re-grounding."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
OLD = BASE / "multiview_primary/CR5_QUEUE50_DIRECTED_GEOMETRY.json"
NEW = BASE / "regrounding_v2/CR5_QUEUE50_TARGET_ROUTE_GEOMETRY_V2.json"
OUT = BASE / "regrounding_v2/CR5_QUEUE50_REGROUNDED_CANDIDATES.json"
EXPECTED = {
    OLD: "46609126537ddb9c4936bc93d683dd06243b91203d6bf2f7c03f30cca7deb850",
    NEW: "7a0044ab458f130d74b37331904ddad379f2270cdb4dbc0ef6213cab09df9fa0",
}
EXPECTED_RECOVERED = {
    "q02_ep37248_hv02",
    "q29_ep41108_hv01",
    "q36_ep1049_hv05",
    "q44_ep38032_hv04",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected):
            raise SystemExit("pinned geometry source drift: " + str(path))
    old_doc = json.loads(OLD.read_text())
    new_doc = json.loads(NEW.read_text())
    old = {row["event_id"]: row for row in old_doc["events"]}
    new = {row["event_id"]: row for row in new_doc["events"]}
    if set(old) != set(new) or len(new) != 50:
        raise SystemExit("geometry event closure failure")
    recovered = [new[event_id] for event_id in sorted(new)
                 if old[event_id]["status"] == "GEOMETRY_REJECT"
                 and new[event_id]["status"] ==
                 "GEOMETRY_PASS_CONTROLLER_REQUIRED"]
    if {row["event_id"] for row in recovered} != EXPECTED_RECOVERED:
        raise SystemExit("unexpected re-grounded candidate set")
    counts = Counter(row["status"] for row in recovered)
    output = {
        "manifest": "MF2-CR5 queue50 target-route re-grounded candidates",
        "revision": "cr5-target-route-regrounded-candidates/1",
        "status": "COMPLETE_CONTROLLER_GATE_REQUIRED",
        "scope": "four queue50 candidates recovered by generic v2 geometry",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "thresholds": new_doc["thresholds"],
        "target_direction_policy": new_doc["target_direction_policy"],
        "candidate_count": len(recovered),
        "status_counts": dict(sorted(counts.items())),
        "events": recovered,
        "original_geometry_artifact_modified": False,
        "network_calls_made": 0,
        "controller_rollouts_made": 0,
        "training_authorized": False,
    }
    temporary = OUT.with_name(OUT.name + ".part")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({
        "status": output["status"],
        "candidate_count": output["candidate_count"],
        "event_ids": sorted(EXPECTED_RECOVERED),
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
