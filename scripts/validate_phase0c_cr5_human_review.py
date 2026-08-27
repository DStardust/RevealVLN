#!/usr/bin/env python3
"""Fail-closed integrity validator for the CR5 human review packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path("/mnt/daiyang/vla")
PACKET = ROOT / "artifacts/phase0/phase0c_cr5_human_review_v1"
MANIFEST = PACKET / "CR5_HUMAN_REVIEW_MANIFEST.json"
TEMPLATE = PACKET / "CR5_HUMAN_REVIEW_TEMPLATE.jsonl"
GUIDE = PACKET / "REVIEW_GUIDE_ZH.md"
LLM_GUIDE = PACKET / "LLM_REVIEW_PROMPT_EN.md"
EXPECTED_MANIFEST_SHA256 = (
    "88d77ecc3b9e2f39389d898604ce55e694d8f9f2b7d76b6262775881beae5285"
)
EXPECTED_TEMPLATE_SHA256 = (
    "b9508ee3ab4fe453c98b936289722313f2fd996fcf91a0fb07bf2da9c5b6b10c"
)
EXPECTED_GUIDE_SHA256 = (
    "4eb383369df094e240ca8dacb80cb40b570250ab5c66d7355635a825007db68a"
)
EXPECTED_LLM_GUIDE_SHA256 = (
    "283d3dd3d956f97ce5a49d5a875b4fb3560b891ac7718c3b349c977999769c11"
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def main() -> int:
    expected = {
        MANIFEST: EXPECTED_MANIFEST_SHA256,
        TEMPLATE: EXPECTED_TEMPLATE_SHA256,
        GUIDE: EXPECTED_GUIDE_SHA256,
        LLM_GUIDE: EXPECTED_LLM_GUIDE_SHA256,
    }
    for path, digest in expected.items():
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != digest):
            raise SystemExit("packet control file drift: " + str(path))
    manifest = json.loads(MANIFEST.read_text())
    if (manifest.get("status") != "READY_FOR_HUMAN_BRANCH_REVIEW"
            or manifest.get("board_count") != 10
            or manifest.get("training_authorized") is not False
            or manifest.get("labels_created") != 0
            or manifest.get("human_reviews_completed") != 0):
        raise SystemExit("manifest boundary failure")
    items = manifest["items"]
    if len({row["event_id"] for row in items}) != 10:
        raise SystemExit("manifest event uniqueness failure")
    for row in items:
        path = ROOT / row["board_path"]
        if (not path.is_file() or path.is_symlink()
                or ROOT.resolve() not in path.resolve().parents
                or path.stat().st_size != row["board_bytes"]
                or sha256_file(path) != row["board_sha256"]):
            raise SystemExit("board integrity failure: " + row["event_id"])
        with Image.open(path) as image:
            image.load()
            if list(image.size) != row["board_pixels"] \
                    or image.mode != "RGB":
                raise SystemExit("board decode/pixel failure: "
                                 + row["event_id"])
        if (row["geometry_status"] !=
                "GEOMETRY_PASS_CONTROLLER_REQUIRED"
                or row["controller_status"] !=
                "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
                or row["causal_prefix_status"] !=
                "PENDING_SEPARATE_GATE"
                or row["human_status"] != "PENDING"
                or row["training_label"] is not False):
            raise SystemExit("board semantic boundary failure: "
                             + row["event_id"])

    template_rows = [json.loads(line) for line in TEMPLATE.read_text().splitlines()
                     if line.strip()]
    if ({row["event_id"] for row in template_rows}
            != {row["event_id"] for row in items}):
        raise SystemExit("review template closure failure")
    for row in template_rows:
        if (row["reviewer_id"] is not None
                or row["reviewer_type"] != "HUMAN"
                or row["final_label"] is not None
                or row["reason_codes"] != []
                or row["comment_zh"] != ""):
            raise SystemExit("template contains a fabricated review")
        question_keys = [
            "two_distinct_executable_exits",
            "alternative_is_not_incoming_closed_or_duplicate",
            "instruction_uniquely_selects_target",
            "decision_center_and_temporal_order_are_reasonable",
        ]
        if any(row[key] is not None for key in question_keys):
            raise SystemExit("template contains a fabricated answer")
    print(json.dumps({
        "status": "PASS",
        "boards_verified": len(items),
        "templates_verified": len(template_rows),
        "labels_created": 0,
        "training_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
