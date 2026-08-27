#!/usr/bin/env python3
"""Validate the portable 100-event RxR full-set audit package."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
OUT = BASE / "multibranch_fullset_audit_100"
INDEX = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
SELECTION = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_SELECTION.json"
MANIFEST = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_MANIFEST.json"
TEMPLATE = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_TEMPLATE.jsonl"
REVIEWER = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_REVIEWER.html"
GUIDE = OUT / "审核说明.md"
REPORT = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_PACKAGE_ACCEPTANCE.json"
RANK_SALT = "revealnav-mf2-fullset-human-audit-100-v1"
QUOTAS = {"train": 43, "development": 10, "gold": 16}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(event_id: str, cohort: str) -> str:
    return hashlib.sha256(
        (RANK_SALT + "|" + cohort + "|" + event_id).encode()
    ).hexdigest()


def expected_selection(rows):
    mandatory = sorted(
        (row for row in rows if row["candidate_branch_count"] >= 3),
        key=lambda row: (rank(row["event_id"], "mandatory"), row["event_id"]),
    )
    selected = list(mandatory)
    for split, quota in QUOTAS.items():
        remaining = [row for row in rows
                     if row["candidate_branch_count"] == 2
                     and row["split"] == split]
        counts = Counter(row["scene_id"] for row in mandatory
                         if row["split"] == split)
        for _ in range(quota):
            chosen = min(
                remaining,
                key=lambda row: (
                    counts[row["scene_id"]],
                    rank(row["event_id"], "two-branch-" + split),
                    row["event_id"],
                ),
            )
            remaining.remove(chosen)
            selected.append(chosen)
            counts[chosen["scene_id"]] += 1
    return selected


def safe_file(path: Path, boundary: Path) -> bool:
    return (path.is_file() and not path.is_symlink()
            and boundary.resolve() in path.resolve().parents)


def main() -> int:
    failures = []
    for path in (INDEX, SELECTION, MANIFEST, TEMPLATE, REVIEWER, GUIDE):
        if not path.is_file() or path.is_symlink():
            failures.append("missing or unsafe package source: " + str(path))
    if failures:
        raise SystemExit("\n".join(failures))
    index = json.loads(INDEX.read_text())
    selection = json.loads(SELECTION.read_text())
    manifest = json.loads(MANIFEST.read_text())
    expected = expected_selection(index["records"])
    expected_ids = [row["event_id"] for row in expected]
    selected_ids = [row["event_id"] for row in selection["items"]]
    items = manifest["items"]
    item_ids = [row["event_id"] for row in items]
    if not (
        selection.get("status") == "SELECTION_FROZEN_BEFORE_HUMAN_LABELING"
        and selection.get("rank_salt") == RANK_SALT
        and selected_ids == expected_ids
        and len(selected_ids) == len(set(selected_ids)) == 100
        and selection.get("human_labels_created") == 0
        and selection.get("training_authorized") is False
    ):
        failures.append("selection protocol mismatch")
    if not (
        manifest.get("status") == "READY_FOR_FRESH_FULLSET_HUMAN_AUDIT"
        and item_ids == selected_ids
        and manifest.get("human_labels_created") == 0
        and manifest.get("training_authorized") is False
        and manifest.get("future_frames_in_online_evidence") == 0
        and manifest.get("panoramas_in_online_evidence") == 0
    ):
        failures.append("manifest contract mismatch")
    candidate_counts = Counter(row["candidate_branch_count"] for row in items)
    split_two = Counter(row["split"] for row in items
                        if row["candidate_branch_count"] == 2)
    if candidate_counts != Counter({2: 69, 3: 31}) or split_two != QUOTAS:
        failures.append("candidate or split quota mismatch")

    causal_count = 0
    logical_bytes = 0
    for row in items:
        board = ROOT / row["board_path"]
        if (not safe_file(board, OUT)
                or board.stat().st_size != row["board_bytes"]
                or sha256_file(board) != row["board_sha256"]):
            failures.append("board provenance: " + row["event_id"])
        else:
            try:
                with Image.open(board) as image:
                    if image.size != (3600, 1900) or image.format != "JPEG":
                        failures.append("board geometry: " + row["event_id"])
            except OSError:
                failures.append("board decode: " + row["event_id"])
            logical_bytes += row["board_bytes"]
        evidence = ROOT / row["language_evidence_path"]
        if (not evidence.is_file() or evidence.is_symlink()
                or ROOT not in evidence.resolve().parents
                or sha256_file(evidence) != row["language_evidence_sha256"]):
            failures.append("language evidence provenance: " + row["event_id"])
        frames = row["causal_media"]
        prefixes = [frame["prefix_index"] for frame in frames]
        if (not frames or prefixes != list(range(prefixes[0], prefixes[-1] + 1))
                or prefixes[-1] != row["confirmation_prefix"]
                or sum(frame["confirmation_frame"] for frame in frames) != 1
                or not frames[-1]["confirmation_frame"]):
            failures.append("causal ordering: " + row["event_id"])
        for frame in frames:
            path = ROOT / frame["path"]
            if (not safe_file(path, OUT)
                    or path.stat().st_size != frame["bytes"]
                    or sha256_file(path) != frame["sha256"]
                    or frame["hfov_deg"] != 63.0
                    or frame["prefix_index"] > row["confirmation_prefix"]
                    or "PANORAMA" in path.name.upper()):
                failures.append("causal frame provenance: " + row["event_id"])
            else:
                logical_bytes += frame["bytes"]
            causal_count += 1
    if causal_count != manifest["counts"]["causal_frames"]:
        failures.append("causal frame count mismatch")

    templates = [json.loads(line) for line in TEMPLATE.read_text().splitlines()
                 if line.strip()]
    if (len(templates) != 100
            or [row["event_id"] for row in templates] != item_ids
            or any(row["reviewer_id"] is not None
                   or row["final_label"] is not None
                   or row["reason_codes"] for row in templates)):
        failures.append("blank template mismatch")
    html = REVIEWER.read_text()
    if ("/mnt/" in html or "file://" in html
            or any("boards/" + Path(row["board_path"]).name not in html
                   for row in items)):
        failures.append("portable reviewer mismatch")
    for source, expected_sha in manifest["sources"].items():
        path = ROOT / source
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected_sha):
            failures.append("source drift: " + source)
    parts = list(OUT.rglob("*.part"))
    if parts:
        failures.append("stale part files")

    output = {
        "schema_version": "revealnav-mf2-fullset-audit-package-acceptance/1",
        "status": "PACKAGE_PASS_READY_FOR_HUMAN_REVIEW" if not failures
                  else "PACKAGE_FAIL",
        "failures": failures,
        "counts": {
            "items": len(items),
            "three_branch": candidate_counts[3],
            "two_branch": candidate_counts[2],
            "unique_scenes": len({row["scene_id"] for row in items}),
            "causal_frames": causal_count,
            "logical_package_bytes": logical_bytes,
        },
        "selection_sha256": sha256_file(SELECTION),
        "manifest_sha256": sha256_file(MANIFEST),
        "template_sha256": sha256_file(TEMPLATE),
        "reviewer_sha256": sha256_file(REVIEWER),
        "human_labels_created": 0,
        "training_authorized": False,
    }
    part = REPORT.with_name(REPORT.name + ".part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False,
                               sort_keys=True) + "\n")
    os.replace(part, REPORT)
    print(json.dumps({"status": output["status"],
                      "failures": failures, "counts": output["counts"],
                      "output": str(REPORT.relative_to(ROOT))},
                     indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
