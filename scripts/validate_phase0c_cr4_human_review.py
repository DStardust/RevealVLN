#!/usr/bin/env python3
"""Independent, fail-closed acceptance for the MF2-CR4 review package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
from pathlib import Path

from PIL import Image


ROOT = Path("/mnt/daiyang/vla")
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr4_human_review"
PACKET = OUT_DIR / "PHASE0C_CR4_HUMAN_REVIEW.json"
FORM = OUT_DIR / "HUMAN_REVIEW_CR4.csv"
GUIDE = OUT_DIR / "HUMAN_REVIEW_GUIDE_CR4_ZH.md"
LLM_PROMPT = OUT_DIR / "FIXED_LLM_AUDIT_PROMPT_CR4.md"
INPUT = ROOT / (
    "artifacts/phase0/phase0c_clause_grounding_mllm/"
    "MLLM_CLAUSE_GROUNDING_INPUTS.json")
MLLM_ACCEPTANCE = ROOT / (
    "artifacts/phase0/phase0c_clause_grounding_mllm/"
    "MLLM_CLAUSE_GROUNDING_ACCEPTANCE.json")
LOCAL_PACKET = ROOT / (
    "artifacts/phase0/phase0c_language_review_35_v2_localmap/"
    "PHASE0C_LANGUAGE_REVIEW_35_V2_LOCALMAP.json")
OUTPUT = OUT_DIR / "CR4_HUMAN_REVIEW_PACKAGE_ACCEPTANCE.json"
EXPECTED = {
    PACKET: "4821c83437acc9b3dae3ae4130b75d48bbc52ea431cdf37db317381ae83bb94a",
    FORM: "58a95f3e7569be21e04975e8900afe417a02629661676a158f83dae3c0b4b273",
    GUIDE: "f24b4c39d1f94518bc5454e4059591b2f67ca13e3a63eda640b6ff55ccf27419",
    LLM_PROMPT: "ea355a0f7e3506e7e72d35283c78603050225578b47eafd94b1eebc81968880c",
    INPUT: "d576b49122b7b3a90c71d6ff6648926afd6b063f433e8feb9a0f85488bb1f4ca",
    MLLM_ACCEPTANCE: "8a014c571b8d8715b057a547ff6c5ee409c358a70244ce0aa94919b485404bfb",
    LOCAL_PACKET: "3c3f650fa26ceb1d948614e3c1eb6800dca85504e1cad7690c52ab1294424c7c",
}
HUMAN_FIELDS = [
    "reviewed", "reviewer_id", "review_timestamp",
    "clause_alignment_decision", "human_selected_segment_ids",
    "instruction_clause", "branch_dependent_instruction",
    "target_branch_matches_instruction", "causal_reveal_confirmed",
    "semantic_track_confirmed", "cost_expiry_interpretation_confirmed",
    "candidate_valid", "rejection_reason", "reviewer_notes",
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def regular_project_file(path: Path) -> bool:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        return False
    return (stat.S_ISREG(info.st_mode) and not path.is_symlink()
            and ROOT.resolve() in resolved.parents)


def atomic_json(path: Path, value) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def main() -> int:
    failures = []
    for path, expected in EXPECTED.items():
        if not regular_project_file(path):
            failures.append(str(path.relative_to(ROOT)) + ":unsafe_or_missing")
        elif sha256_file(path) != expected:
            failures.append(str(path.relative_to(ROOT)) + ":sha256_drift")
    if failures:
        atomic_json(OUTPUT, {"status": "NO_GO", "failures": failures,
                             "training_authorized": False})
        return 1
    packet = json.loads(PACKET.read_text())
    inputs = json.loads(INPUT.read_text())
    input_events = {event["event_id"]: event for event in inputs["events"]}
    acceptance = json.loads(MLLM_ACCEPTANCE.read_text())
    if acceptance.get("status") != "PASS" or \
            acceptance.get("events_passed") != 35:
        failures.append("mllm_acceptance")
    if packet.get("status") != "PASS_PENDING_HUMAN_VERIFICATION":
        failures.append("packet_status")
    if packet.get("revision") != "phase0c-cr4-human-review/1":
        failures.append("packet_revision")
    if packet.get("row_count") != 35 or packet.get("board_count") != 35:
        failures.append("packet_cardinality")
    if packet.get("scene_count") != 22:
        failures.append("scene_count")
    if packet.get("reviewed_true_count") != 0:
        failures.append("reviewed_count")
    if packet.get("human_fields_prefilled") is not False:
        failures.append("human_prefill_flag")
    if packet.get("network_calls_made_by_builder") != 0:
        failures.append("builder_network_calls")
    if packet.get("private_distribution_authorized") is not False:
        failures.append("private_distribution")
    if packet.get("verified_language_reveal_events") != 0:
        failures.append("verified_events")
    if packet.get("training_authorized") is not False:
        failures.append("training_authorized")
    rows = packet.get("rows", [])
    if len(rows) != 35 or len({row.get("event_id") for row in rows}) != 35:
        failures.append("row_set")
    board_records = {item.get("path"): item
                     for item in packet.get("board_manifest", [])}
    observed_board_bytes = 0
    result_hashes = set()
    status_counts = {}
    for index, row in enumerate(rows):
        event_id = row.get("event_id")
        prefix = f"row[{index}]:{event_id}"
        if row.get("row_order") != index:
            failures.append(prefix + ":row_order")
        event = input_events.get(event_id)
        if event is None:
            failures.append(prefix + ":input_event")
            continue
        for key in ("episode_id", "scene_id", "instruction_sha256"):
            if row.get(key) != event.get(key):
                failures.append(prefix + ":" + key)
        for field in HUMAN_FIELDS:
            expected = False if field == "reviewed" else None
            if row.get(field) is not expected:
                failures.append(prefix + ":human_field:" + field)
        board_rel = row.get("cr4_review_board")
        board_path = ROOT / (board_rel or "__missing__")
        record = board_records.get(board_rel)
        if not regular_project_file(board_path) or record is None:
            failures.append(prefix + ":board_missing_or_unsafe")
        else:
            actual_sha = sha256_file(board_path)
            if actual_sha != row.get("cr4_review_board_sha256") \
                    or actual_sha != record.get("sha256"):
                failures.append(prefix + ":board_sha256")
            size = board_path.stat().st_size
            observed_board_bytes += size
            if size != record.get("bytes"):
                failures.append(prefix + ":board_bytes")
            try:
                with Image.open(board_path) as image:
                    image.verify()
                with Image.open(board_path) as image:
                    if image.size != (2560, 1340) or image.mode != "RGB" \
                            or image.format != "JPEG":
                        failures.append(prefix + ":board_decode_geometry")
            except Exception as exc:
                failures.append(prefix + ":board_decode:" + type(exc).__name__)
        result_path = ROOT / row.get("mllm_result_path", "__missing__")
        if not regular_project_file(result_path):
            failures.append(prefix + ":result_missing_or_unsafe")
            continue
        result_sha = sha256_file(result_path)
        result_hashes.add(result_sha)
        if result_sha != row.get("mllm_result_sha256"):
            failures.append(prefix + ":result_sha256")
        result = json.loads(result_path.read_text())
        proposal = result.get("proposal", {})
        if result.get("status") != "VALID_MLLM_PROPOSAL":
            failures.append(prefix + ":result_status")
        if result.get("provider_response_metadata", {}).get("model") != \
                "qwen3.8-max" or row.get("mllm_model") != "qwen3.8-max":
            failures.append(prefix + ":model")
        comparisons = {
            "mllm_status": proposal.get("status"),
            "mllm_selected_segment_ids": proposal.get("selected_segment_ids"),
            "mllm_evidence_frame_ids": proposal.get("evidence_frame_ids"),
            "mllm_confidence": proposal.get("confidence"),
        }
        for key, expected in comparisons.items():
            if row.get(key) != expected:
                failures.append(prefix + ":" + key)
        if row.get("mllm_proposal_is_ground_truth") is not False:
            failures.append(prefix + ":proposal_ground_truth")
        status = proposal.get("status")
        status_counts[status] = status_counts.get(status, 0) + 1
        valid_ids = {segment["segment_id"]
                     for segment in event["deterministic_segments"]}
        selected = proposal.get("selected_segment_ids", [])
        if not 1 <= len(selected) <= 3 or not set(selected) <= valid_ids:
            failures.append(prefix + ":selected_segments")
        spans = row.get("instruction_render", {}).get("selected_spans")
        expected_spans = [[segment["char_start"],
                           segment["char_end_exclusive"]]
                          for segment in event["deterministic_segments"]
                          if segment["segment_id"] in selected]
        if spans != expected_spans:
            failures.append(prefix + ":highlight_spans")
        if row.get("instruction_render", {}).get(
                "lossless_wrap_verified") is not True:
            failures.append(prefix + ":instruction_lossless")
    if len(board_records) != 35:
        failures.append("board_manifest_set")
    if len(result_hashes) != 35:
        failures.append("result_hash_uniqueness")
    if observed_board_bytes != packet.get("board_total_bytes"):
        failures.append("board_total_bytes")
    if status_counts != {"UNIQUE_MATCH": 35}:
        failures.append("proposal_status_counts")
    with FORM.open(newline="", encoding="utf-8") as handle:
        form_rows = list(csv.DictReader(handle))
    if len(form_rows) != 35:
        failures.append("csv_row_count")
    else:
        for packet_row, form_row in zip(rows, form_rows):
            if form_row.get("event_id") != packet_row.get("event_id"):
                failures.append("csv_event_order")
            if form_row.get("reviewed") != "False":
                failures.append("csv_reviewed_default")
            for field in HUMAN_FIELDS:
                if field != "reviewed" and form_row.get(field) != "":
                    failures.append("csv_human_field_prefilled:" + field)
    for path in (PACKET, FORM, GUIDE, LLM_PROMPT):
        if "sk-" in path.read_text(encoding="utf-8", errors="ignore"):
            failures.append(str(path.relative_to(ROOT)) + ":secret_pattern")
    unsafe_entries = []
    for path in OUT_DIR.rglob("*"):
        if path.is_symlink() or path.name.endswith(".part"):
            unsafe_entries.append(str(path.relative_to(ROOT)))
    if unsafe_entries:
        failures.append("unsafe_output_entries")
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free_bytes < 8 * 1024**3:
        failures.append("disk_below_8GiB")
    status = "PASS" if not failures else "NO_GO"
    output = {
        "status": status,
        "revision": "phase0c-cr4-review-acceptance/1",
        "packet_sha256": EXPECTED[PACKET],
        "form_sha256": EXPECTED[FORM],
        "guide_sha256": EXPECTED[GUIDE],
        "llm_prompt_sha256": EXPECTED[LLM_PROMPT],
        "rows": len(rows),
        "boards": len(board_records),
        "board_bytes": observed_board_bytes,
        "proposal_status_counts": status_counts,
        "human_fields_blank": not any(
            ":human_field:" in item or item.startswith("csv_human_field")
            for item in failures),
        "verified_language_reveal_events": 0,
        "human_verification_required": True,
        "training_authorized": False,
        "free_bytes": free_bytes,
        "unsafe_entries": unsafe_entries,
        "failures": failures,
    }
    atomic_json(OUTPUT, output)
    print(json.dumps({
        "status": status,
        "failures": len(failures),
        "rows": len(rows),
        "boards": len(board_records),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
    }, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
