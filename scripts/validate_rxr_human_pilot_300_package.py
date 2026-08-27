#!/usr/bin/env python3
"""Independently validate the RxR 300-event human review packet."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
OUT_DIR = BASE / "human_pilot_300"
ACCEPTANCE = BASE / "RXR_EXPANSION_AUTOMATIC_FILTER_ACCEPTANCE.json"
ANALYSIS = BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json"
SELECTION = OUT_DIR / "RXR_HUMAN_PILOT_300_SELECTION.json"
MANIFEST = OUT_DIR / "RXR_HUMAN_PILOT_300_MANIFEST.json"
TEMPLATE = OUT_DIR / "RXR_HUMAN_PILOT_300_TEMPLATE.jsonl"
REVIEWER = OUT_DIR / "RXR_HUMAN_PILOT_300_REVIEWER.html"
GUIDE = OUT_DIR / "审核说明.md"
REPORT = OUT_DIR / "RXR_HUMAN_PILOT_300_PACKAGE_ACCEPTANCE.json"
RANK_SALT = "revealnav-human-pilot-300-v1"
SPEC_HASHES = {
    ROOT / "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    ROOT / "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(event_id: str, cohort: str) -> str:
    return hashlib.sha256(
        (RANK_SALT + "|" + cohort + "|" + event_id).encode()).hexdigest()


def safe_file(path: Path, boundary: Path) -> bool:
    return (path.is_file() and not path.is_symlink()
            and boundary.resolve() in path.resolve().parents)


def expected_selection(eligible, analysis_by_id):
    ordered = sorted(eligible, key=lambda value: (rank(value, "core"), value))
    core = ordered[:250]
    remaining = set(ordered[250:])
    counts = Counter(analysis_by_id[value]["scene_id"] for value in core)
    all_scenes = {analysis_by_id[value]["scene_id"] for value in eligible}
    supplement = []
    for scene_id in sorted(all_scenes - set(counts)):
        candidates = [value for value in remaining
                      if analysis_by_id[value]["scene_id"] == scene_id]
        chosen = min(candidates,
                     key=lambda value: (rank(value, "supplement"), value))
        supplement.append(chosen)
        remaining.remove(chosen)
        counts[scene_id] += 1
    while len(supplement) < 50:
        chosen = min(
            remaining,
            key=lambda value: (
                counts[analysis_by_id[value]["scene_id"]],
                rank(value, "supplement"), value))
        supplement.append(chosen)
        remaining.remove(chosen)
        counts[analysis_by_id[chosen]["scene_id"]] += 1
    return core, supplement


def main() -> int:
    failures = []
    for path in (ACCEPTANCE, ANALYSIS, SELECTION, MANIFEST, TEMPLATE,
                 REVIEWER, GUIDE):
        if not path.is_file() or path.is_symlink():
            failures.append("missing or unsafe package source: " + str(path))
    if failures:
        raise SystemExit("\n".join(failures))
    for path, expected in SPEC_HASHES.items():
        if sha256_file(path) != expected:
            failures.append("frozen specification drift: " + str(path))

    acceptance = json.loads(ACCEPTANCE.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    selection = json.loads(SELECTION.read_text())
    manifest = json.loads(MANIFEST.read_text())
    analysis_by_id = {row["event_id"]: row for row in analysis["events"]}
    eligible = acceptance["eligible_event_ids"]
    expected_core, expected_supplement = expected_selection(
        eligible, analysis_by_id)
    rows = selection["items"]
    core = [row["event_id"] for row in rows
            if row["cohort"] == "AUDIT_CORE_UNIFORM_250"]
    supplement = [row["event_id"] for row in rows
                  if row["cohort"] == "SCENE_COVERAGE_SUPPLEMENT_50"]
    if (selection["status"] != "SELECTION_FROZEN_BEFORE_HUMAN_LABELING"
            or core != expected_core or supplement != expected_supplement
            or len(rows) != 300
            or [row["review_index"] for row in rows] != list(range(1, 301))
            or len({row["event_id"] for row in rows}) != 300
            or not {row["event_id"] for row in rows} <= set(eligible)):
        failures.append("selection protocol or closure mismatch")

    items = manifest["items"]
    item_ids = [row["event_id"] for row in items]
    if (manifest["status"] != "READY_FOR_HUMAN_REVIEW"
            or len(items) != 300 or item_ids != [row["event_id"] for row in rows]
            or manifest["human_labels_created"] != 0
            or manifest["training_authorized"] is not False
            or manifest["future_frames_in_human_causal_strips"] != 0
            or manifest["panoramas_in_human_causal_strips"] != 0):
        failures.append("review manifest contract mismatch")
    scenes = {row["scene_id"] for row in items}
    episodes = {row["episode_id"] for row in items}
    if len(scenes) != 52 or len(episodes) != 300:
        failures.append("scene or episode diversity mismatch")

    media_count = 0
    logical_bytes = 0
    for row in items:
        board_path = ROOT / row["board_path"]
        if (not safe_file(board_path, OUT_DIR)
                or board_path.stat().st_size != row["board_bytes"]
                or sha256_file(board_path) != row["board_sha256"]
                or row["board_pixels"] != [3000, 1800]):
            failures.append("board integrity: " + row["event_id"])
        else:
            logical_bytes += row["board_bytes"]
        records = row["causal_media"]
        prefixes = [value["prefix_index"] for value in records]
        if (not records or prefixes != sorted(prefixes)
                or len(prefixes) != len(set(prefixes))
                or max(prefixes) != row["confirmation_prefix"]
                or sum(value["confirmation_frame"] for value in records) != 1
                or not records[-1]["confirmation_frame"]):
            failures.append("causal ordering: " + row["event_id"])
        for value in records:
            path = ROOT / value["path"]
            if (not safe_file(path, OUT_DIR)
                    or path.stat().st_size != value["bytes"]
                    or sha256_file(path) != value["sha256"]
                    or value["hfov_deg"] != 63.0
                    or "PANORAMA" in path.name.upper()
                    or value["prefix_index"] > row["confirmation_prefix"]):
                failures.append("causal media integrity: " + row["event_id"])
            else:
                logical_bytes += value["bytes"]
            media_count += 1
    if media_count != manifest["causal_media_count"]:
        failures.append("causal media count mismatch")

    template_rows = [json.loads(line) for line in TEMPLATE.read_text().splitlines()
                     if line.strip()]
    if (len(template_rows) != 300
            or [row["event_id"] for row in template_rows] != item_ids
            or any(row["final_label"] is not None
                   or row["reviewer_id"] is not None
                   or row["reason_codes"]
                   for row in template_rows)):
        failures.append("blank review template mismatch")
    html = REVIEWER.read_text()
    if ("/mnt/" in html or "file://" in html
            or any(("boards/" + Path(row["board_path"]).name) not in html
                   for row in items)):
        failures.append("reviewer portability mismatch")
    for source, expected in manifest["sources"].items():
        path = ROOT / source
        if not path.is_file() or path.is_symlink() \
                or sha256_file(path) != expected:
            failures.append("source hash drift: " + source)
    parts = [str(path.relative_to(ROOT)) for path in OUT_DIR.rglob("*.part")]
    if parts:
        failures.append("stale part files")

    output = {
        "manifest": "RevealNav RxR human pilot 300 package acceptance",
        "revision": "rxr-human-pilot-300-package-acceptance/1",
        "status": "PACKAGE_PASS_READY_FOR_HUMAN_REVIEW" if not failures
                  else "PACKAGE_FAIL",
        "failures": failures,
        "counts": {
            "eligible_population": len(eligible),
            "audit_core": len(core),
            "scene_coverage_supplement": len(supplement),
            "review_items": len(items),
            "unique_episodes": len(episodes),
            "unique_scenes": len(scenes),
            "causal_media": media_count,
            "logical_package_bytes": logical_bytes,
        },
        "selection_sha256": sha256_file(SELECTION),
        "manifest_sha256": sha256_file(MANIFEST),
        "template_sha256": sha256_file(TEMPLATE),
        "reviewer_sha256": sha256_file(REVIEWER),
        "frozen_specifications_unchanged": not any(
            value.startswith("frozen specification drift")
            for value in failures),
        "human_labels_created": 0,
        "training_authorized": False,
    }
    part = REPORT.with_name(REPORT.name + ".part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, REPORT)
    print(json.dumps({
        "status": output["status"],
        "failures": failures,
        "counts": output["counts"],
        "output": str(REPORT.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
