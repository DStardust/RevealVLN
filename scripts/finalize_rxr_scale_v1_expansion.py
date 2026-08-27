#!/usr/bin/env python3
"""Adjudicate event-scale capacity after automatic scale-v1 extraction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
CURRENT_GATE = BASE / "expiry_r3/RXR_EXPIRY_R3_FEATURE_GATE.json"
CURRENT_MANIFEST = BASE / "expiry_r3/RXR_EXPIRY_R3_FEATURE_MANIFEST.json"
SCALE_GATE = BASE / "scale_v1/automatic/multibranch/RXR_SCALE_FEATURE_GATE.json"
SCALE_MANIFEST = BASE / "scale_v1/automatic/multibranch/RXR_SCALE_FEATURE_MANIFEST.json"
GOLD_PACKAGE = BASE / "scale_v1/new_gold/review_package/RXR_NEW_GOLD_REVIEW_MANIFEST.json"
OUT = BASE / "scale_v1/RXR_SCALE_V1_EVENT_EXPANSION_GATE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    current_gate = json.loads(CURRENT_GATE.read_text())
    current_manifest = json.loads(CURRENT_MANIFEST.read_text())
    scale_gate = json.loads(SCALE_GATE.read_text())
    scale_manifest = json.loads(SCALE_MANIFEST.read_text())
    gold = json.loads(GOLD_PACKAGE.read_text())
    current_ids = {row["event_id"] for row in current_manifest["records"]}
    scale_ids = {row["event_id"] for row in scale_manifest["records"]}
    automatic_total = len(current_ids | scale_ids)
    minimum_gold = 600
    projected_total = automatic_total + minimum_gold
    gates = {
        "current_feature_gate_pass": current_gate.get("status") == "EXPIRY_R3_FEATURE_GATE_PASS",
        "scale_feature_gate_pass": scale_gate.get("status") == "FEATURE_GATE_PASS_AUTOMATIC_SCALE_READY",
        "automatic_event_ids_disjoint": not (current_ids & scale_ids),
        "automatic_manifests_have_no_gold": all(
            row["split"] != "gold"
            for row in current_manifest["records"] + scale_manifest["records"]
        ),
        "new_gold_review_population_at_least_900": len(gold.get("items", [])) >= 900,
        "projected_total_at_minimum_gold_at_least_2000": projected_total >= 2000,
    }
    additional = max(0, 2000 - projected_total)
    output = {
        "schema_version": "revealnav-rxr-scale-expansion-gate/1",
        "status": "EVENT_SCALE_CAPACITY_PASS_GOLD_REVIEWS_REQUIRED" if all(gates.values()) else "ADDITIONAL_AUTOMATIC_EVENTS_REQUIRED",
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (CURRENT_GATE, CURRENT_MANIFEST, SCALE_GATE, SCALE_MANIFEST, GOLD_PACKAGE)
        },
        "counts": {
            "existing_automatic_events": len(current_ids),
            "new_automatic_events": len(scale_ids),
            "automatic_events_total": automatic_total,
            "minimum_new_gold_events": minimum_gold,
            "projected_total_at_minimum_gold": projected_total,
            "additional_automatic_events_required": additional,
        },
        "gates": gates,
        "old_gold_payload_read": False,
        "new_gold_human_reviews_complete": False,
        "gold_authorized": False,
        "training_authorized": False,
        "paper_result": False,
    }
    part = OUT.with_name(OUT.name + ".part")
    part.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(part, OUT)
    print(json.dumps({"status": output["status"], "counts": output["counts"], "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

