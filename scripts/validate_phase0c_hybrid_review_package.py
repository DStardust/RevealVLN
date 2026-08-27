#!/usr/bin/env python3
"""Fail-closed acceptance gate for the blank MF2-CR3 review package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
PACKET = ROOT / ("artifacts/phase0/phase0c_language_review_35_v2_localmap/"
                 "PHASE0C_LANGUAGE_REVIEW_35_V2_LOCALMAP.json")
REVIEW_DIR = ROOT / "artifacts/phase0/phase0c_hybrid_review"
MANIFEST = REVIEW_DIR / "HYBRID_REVIEW_MANIFEST.json"
OUT = REVIEW_DIR / "HYBRID_REVIEW_PACKAGE_ACCEPTANCE.json"
EXPECTED = {
    PACKET: "3c3f650fa26ceb1d948614e3c1eb6800dca85504e1cad7690c52ab1294424c7c",
    ROOT / "METHOD_FREEZE_2_CORRECTNESS_REVISION_3.md":
        "1047e19f8b6144aea41fa622e0a26b9d5f6274e7e8ee04fc8faa5d2cb3b277d7",
    REVIEW_DIR / "FIXED_REVIEW_PROMPT.md":
        "841aff84841c480cefacb58c5bceef6efc313f07cd4b1701b9ffbc89b3be6365",
    MANIFEST: "542315625ad415d957c89dc8c5a14f3382baaa7704bc976dc997a245717730b9",
}
TRACKS = {
    "H": ("TRACK_H_HUMAN.csv", "human", "project researcher"),
    "M1": ("TRACK_M1_QWEN38MAX.csv", "model",
           "Qwen3.8-Max under Claude-Code control"),
    "M2": ("TRACK_M2_CODEX.csv", "model", "clean Codex session"),
}
JSON_FIELDS = {
    "screening_triggers", "target_exit_region", "causal_prefixes",
    "private_media", "local_map_geometry", "instruction_render",
    "frozen_cost_frontiers",
}
JUDGMENT_FIELDS = {
    "reviewed", "reviewer_id", "review_timestamp",
    "branch_dependent_instruction", "instruction_clause",
    "target_branch_matches_instruction", "causal_reveal_confirmed",
    "semantic_track_confirmed", "cost_expiry_interpretation_confirmed",
    "candidate_valid", "rejection_reason", "reviewer_notes",
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def immutable_payload(row):
    keys = [
        "row_order", "event_id", "episode_id", "scene_id", "instruction_id",
        "instruction_sha256", "language", "screening_triggers",
        "semantic_branch_id", "target_exit_region", "causal_prefixes",
        "private_media", "private_local_map", "private_review_board",
        "private_contact_sheet", "local_map_geometry", "instruction_render",
        "frozen_cost_frontiers",
    ]
    return {key: row[key] for key in keys}


def expected_csv_value(source, field):
    if field in JSON_FIELDS:
        return stable_json(source[field])
    return str(source[field])


def main() -> int:
    checks = []

    def record(name, passed, observed):
        checks.append({"name": name, "pass": bool(passed),
                       "observed": observed})
        print(("PASS " if passed else "FAIL ") + name)

    pinned = {str(path.relative_to(ROOT)): sha256_file(path)
              for path in EXPECTED}
    record("all_fixed_inputs_pinned",
           all(pinned[str(path.relative_to(ROOT))] == expected
               for path, expected in EXPECTED.items()), pinned)

    packet = json.loads(PACKET.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rows = packet["rows"]
    record("manifest_blank_scope",
           manifest.get("status") == "READY_UNREVIEWED"
           and manifest.get("packet", {}).get("rows") == 35
           and manifest.get("packet", {}).get("scenes") == 22
           and manifest.get("human_judgments_present") == 0
           and manifest.get("model_judgments_present") == 0
           and manifest.get("training_authorized") is False,
           {"status": manifest.get("status"), "rows": len(rows)})

    form_manifest = {item["track"]: item for item in manifest["forms"]}
    all_rows, forms_ok, provenance_ok, blank_ok = {}, True, True, True
    media_ok, immutable_ok = True, True
    for track, (filename, reviewer_type, reviewer_system) in TRACKS.items():
        path = REVIEW_DIR / filename
        item = form_manifest.get(track, {})
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != item.get("blank_sha256")):
            forms_ok = False
            continue
        with path.open(newline="") as handle:
            track_rows = list(csv.DictReader(handle))
        all_rows[track] = track_rows
        if len(track_rows) != 35:
            forms_ok = False
            continue
        for source, row in zip(rows, track_rows):
            if any(row.get(field) != "" for field in JUDGMENT_FIELDS):
                blank_ok = False
            if not (
                    row.get("reviewer_track") == track
                    and row.get("reviewer_type") == reviewer_type
                    and row.get("reviewer_system") == reviewer_system
                    and row.get("reviewer_version") == ""
                    and row.get("prompt_sha256") ==
                        manifest["prompt"]["sha256"]
                    and row.get("packet_sha256") ==
                        manifest["packet"]["sha256"]):
                provenance_ok = False
            payload = immutable_payload(source)
            expected_immutable = hashlib.sha256(
                stable_json(payload).encode("utf-8")).hexdigest()
            if (row.get("immutable_row_sha256") != expected_immutable
                    or manifest["immutable_row_sha256"].get(
                        source["event_id"]) != expected_immutable):
                immutable_ok = False
            for field in payload:
                if row.get(field) != expected_csv_value(source, field):
                    immutable_ok = False
            board = ROOT / source["private_review_board"]
            if (not board.is_file() or board.is_symlink()
                    or REVIEW_DIR == board.parent
                    or ROOT not in board.resolve().parents):
                media_ok = False

    record("three_sealed_blank_forms_35_rows", forms_ok,
           {track: len(value) for track, value in all_rows.items()})
    record("all_judgment_fields_blank", blank_ok, len(JUDGMENT_FIELDS))
    record("truthful_isolated_track_provenance", provenance_ok,
           {track: TRACKS[track][1] for track in TRACKS})
    record("immutable_rows_match_packet_and_manifest", immutable_ok,
           len(manifest.get("immutable_row_sha256", {})))
    record("private_review_board_paths_project_local", media_ok, len(rows))

    core_equal = (set(all_rows) == set(TRACKS) and all(
        all({key: value for key, value in left.items()
             if key not in JUDGMENT_FIELDS and not key.startswith("reviewer_")
             and key != "display_alias"} ==
            {key: value for key, value in right.items()
             if key not in JUDGMENT_FIELDS and not key.startswith("reviewer_")
             and key != "display_alias"}
            for left, right in zip(all_rows["H"], all_rows[track]))
        for track in ("M1", "M2")))
    record("same_core_payload_across_tracks", core_equal, 35)

    prompt = (REVIEW_DIR / "FIXED_REVIEW_PROMPT.md").read_text()
    guide = (REVIEW_DIR / "HUMAN_REVIEW_GUIDE_ZH.md").read_text()
    language_ok = (manifest.get("llm_prompt_language") == "en"
                   and not any("\u4e00" <= char <= "\u9fff" for char in prompt)
                   and any("\u4e00" <= char <= "\u9fff" for char in guide)
                   and "B-to-TARGET" in prompt and "P" in prompt
                   and "B→T" in guide)
    record("english_llm_prompt_chinese_human_guide", language_ok,
           {"prompt_sha256": sha256_file(
                REVIEW_DIR / "FIXED_REVIEW_PROMPT.md"),
            "guide_sha256": sha256_file(
                REVIEW_DIR / "HUMAN_REVIEW_GUIDE_ZH.md")})

    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    record("disk_free_at_least_8gib", free >= 8 * 1024 ** 3, free)
    passed = all(item["pass"] for item in checks)
    output = {
        "gate": "mf2_cr3_blank_hybrid_review_package_acceptance",
        "revision": "hybrid-review-package-acceptance/1-wide-instruction",
        "status": "PASS_READY_UNREVIEWED" if passed else "FAIL",
        "checks_passed": sum(item["pass"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "packet_sha256": EXPECTED[PACKET],
        "human_judgments_present": 0,
        "model_judgments_present": 0,
        "training_authorized": False,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"],
                      "checks": "%d/%d" % (output["checks_passed"],
                                             output["checks_total"]),
                      "output": str(OUT.relative_to(ROOT)),
                      "output_sha256": sha256_file(OUT)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
