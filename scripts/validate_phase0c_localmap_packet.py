#!/usr/bin/env python3
"""Fail-closed integrity/readability gate for the v2 local-map packet."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image


ROOT = Path("/mnt/daiyang/vla")
V1 = ROOT / "artifacts/phase0/phase0c_language_review_35/PHASE0C_LANGUAGE_REVIEW_35.json"
V2_DIR = ROOT / "artifacts/phase0/phase0c_language_review_35_v2_localmap"
V2 = V2_DIR / "PHASE0C_LANGUAGE_REVIEW_35_V2_LOCALMAP.json"
OUT = V2_DIR / "LOCALMAP_PACKET_ACCEPTANCE.json"
EXPECTED_V1 = "b97f546d454d09a57c21153adc55bc02c30a4c694b07cd925091fac0b07a6784"
EXPECTED_V2 = "3c3f650fa26ceb1d948614e3c1eb6800dca85504e1cad7690c52ab1294424c7c"
EXPECTED_FONT = "2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b"
FONT = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"
FONT_LICENSE = ROOT / "assets/fonts/Noto-CJK-LICENSE.txt"
FONT_PROVENANCE = ROOT / "assets/fonts/PROVENANCE.md"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def main() -> int:
    checks = []

    def record(name, passed, observed):
        checks.append({"name": name, "pass": bool(passed),
                       "observed": observed})
        print(("PASS " if passed else "FAIL ") + name)

    v1_sha, v2_sha = sha256_file(V1), sha256_file(V2)
    record("v1_packet_preserved", v1_sha == EXPECTED_V1, v1_sha)
    record("v2_packet_pinned", v2_sha == EXPECTED_V2, v2_sha)
    v1, v2 = json.loads(V1.read_text()), json.loads(V2.read_text())
    rows1, rows2 = v1["rows"], v2["rows"]
    record("packet_pending_35_rows_22_scenes",
           v2.get("status") == "PASS_PENDING_HUMAN_REVIEW"
           and len(rows2) == 35 and v2.get("scene_count") == 22
           and v2.get("reviewed_true_count") == 0
           and v2.get("all_rows_pending") is True,
           {"rows": len(rows2), "scenes": v2.get("scene_count")})
    record("fixed_selection_and_order_unchanged",
           [row["event_id"] for row in rows1] ==
           [row["event_id"] for row in rows2],
           len(rows2))
    immutable_fields = [
        "row_order", "event_id", "episode_id", "scene_id", "instruction_id",
        "instruction_sha256", "instruction_text_for_private_review",
        "semantic_branch_id", "target_exit_region", "causal_prefixes",
        "automatic_track_status", "frozen_cost_frontiers",
    ]
    immutable_ok = all(
        all(left[field] == right[field] for field in immutable_fields)
        for left, right in zip(rows1, rows2))
    record("scientific_row_fields_unchanged", immutable_ok,
           immutable_fields)
    human_fields = v2["human_fields"]
    pending_ok = all(
        row.get("reviewed") is False
        and row.get("annotation_status") ==
            "PENDING_HUMAN_REVIEW_V2_LOCALMAP"
        and all(row.get(field) is None for field in human_fields)
        for row in rows2)
    record("all_human_fields_null", pending_ok,
           {"rows": len(rows2), "fields": len(human_fields)})

    manifest = v2.get("media_manifest", [])
    paths = [item["path"] for item in manifest]
    media_ok, decoded, dimensions = True, 0, {}
    for item in manifest:
        path = ROOT / item["path"]
        try:
            if (not path.is_file() or path.is_symlink()
                    or V2_DIR.resolve() not in path.resolve().parents
                    and "phase0c_language_review_35/private_media" not in
                        item["path"]
                    or path.stat().st_size != item["bytes"]
                    or sha256_file(path) != item["sha256"]):
                media_ok = False
                continue
            with Image.open(path) as image:
                image.load()
                if path.name.endswith("_local_map.jpg"):
                    expected = (640, 640)
                    kind = "local_map"
                elif path.name.endswith("_review_board.jpg"):
                    expected = (1920, 920)
                    kind = "review_board"
                else:
                    expected = (224, 224)
                    kind = "causal_frame"
                dimensions[kind] = dimensions.get(kind, 0) + 1
                if image.size != expected or not all(
                        high - low > 10 for low, high in
                        image.convert("RGB").getextrema()):
                    media_ok = False
                decoded += 1
        except Exception:
            media_ok = False
    record("210_media_references_hash_size_decode",
           media_ok and len(manifest) == decoded == 210
           and len(set(paths)) == 210
           and dimensions == {"causal_frame": 140, "local_map": 35,
                               "review_board": 35},
           {"decoded": decoded, "dimensions": dimensions})
    actual_new = sorted((V2_DIR / "private_media").glob("*.jpg"))
    record("exactly_70_new_private_media_files",
           len(actual_new) == 70 and all(not path.is_symlink()
                                         for path in actual_new),
           len(actual_new))

    geometry = [row["local_map_geometry"] for row in rows2]
    geometry_ok = all(
        item["candidate_endpoints_drawn"] == 3
        and item["view_hfov_deg"] == 63.0
        and item["local_square_span_m"] >= 6.0
        and item["navigable_fraction_in_local_crop"] > 0.0
        and item["union_navigable_fraction_in_local_crop"] >=
            item["navigable_fraction_in_local_crop"]
        and math.isfinite(item["pre_reveal_to_branch_start_distance_m"])
        and item["pre_reveal_to_branch_start_distance_m"] >= 0.0
        for item in geometry)
    cross_level = sum(bool(item["auxiliary_branch_heights_m"])
                      for item in geometry)
    distinct_pb = sum(item["pre_reveal_to_branch_start_distance_m"] > 0.05
                      for item in geometry)
    record("local_geometry_complete", geometry_ok and cross_level > 0
           and distinct_pb > 0,
           {"events": len(geometry), "cross_level": cross_level,
            "distinct_pre_reveal_and_branch_start": distinct_pb,
            "min_event_nav_fraction": min(item[
                "navigable_fraction_in_local_crop"] for item in geometry),
            "min_union_nav_fraction": min(item[
                "union_navigable_fraction_in_local_crop"] for item in geometry)})
    record("v1_global_panels_deprecated_not_deleted",
           all((ROOT / row["private_contact_sheet_v1_deprecated"]).is_file()
               for row in rows2), len(rows2))
    record("map_is_explicitly_offline_only",
           v2["map_contract"].get("offline_only") is True
           and v2["map_contract"].get("model_input") is False,
           v2["map_contract"])
    render_ok = all(
        row["instruction_render"].get("source_text_sha256") ==
            row["instruction_sha256"]
        and row["instruction_render"].get("source_characters") ==
            len(row["instruction_text_for_private_review"])
        and row["instruction_render"].get("lossless_wrap_verified") is True
        and row["instruction_render"].get("rendered_lines", 0) >= 1
        and 18 <= row["instruction_render"].get("font_size_px", 0) <= 29
        and row["instruction_render"].get("panel_width_px") == 800
        and row["instruction_render"].get("panel_height_px") == 920
        for row in rows2)
    record("full_instruction_lossless_35_of_35", render_ok,
           {"rows": len(rows2),
            "max_characters": max(row["instruction_render"][
                "source_characters"] for row in rows2),
            "max_lines": max(row["instruction_render"]["rendered_lines"]
                             for row in rows2),
            "minimum_font_px": min(row["instruction_render"]["font_size_px"]
                                   for row in rows2)})
    contract = v2["map_contract"]
    record("wide_chinese_human_ui_english_llm_contract",
           contract.get("review_board_pixels") == [1920, 920]
           and contract.get("full_instruction_untruncated") is True
           and contract.get("human_visible_guidance_language") == "zh-CN"
           and contract.get("llm_prompt_language") == "en"
           and "distinct from pre-reveal camera pose P" in
               contract.get("target", ""), contract)
    font_record = v2.get("font_provenance", {})
    font_ok = (
        FONT.is_file() and not FONT.is_symlink()
        and sha256_file(FONT) == EXPECTED_FONT
        and FONT_LICENSE.is_file() and not FONT_LICENSE.is_symlink()
        and FONT_PROVENANCE.is_file() and not FONT_PROVENANCE.is_symlink()
        and font_record.get("sha256") == EXPECTED_FONT
        and font_record.get("path") ==
            str(FONT.relative_to(ROOT)))
    record("project_local_font_and_provenance", font_ok, font_record)
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    record("disk_free_at_least_8gib", free >= 8 * 1024 ** 3, free)

    passed = all(item["pass"] for item in checks)
    output = {
        "gate": "mf2_cr3_localmap_review_packet_acceptance",
        "revision": "localmap-packet-acceptance/2-wide-instruction",
        "status": "PASS_PRIVATE_UNREVIEWED" if passed else "FAIL",
        "decision": "READY_FOR_HYBRID_REVIEW" if passed else
                    "LOCALMAP_PACKET_NO_GO",
        "checks_passed": sum(item["pass"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "packet": {"path": str(V2.relative_to(ROOT)), "sha256": v2_sha},
        "human_validated_events": 0,
        "training_authorized": False,
        "private_distribution_authorized": False,
        "visual_inspection": {
            "representative_same_floor_board":
                "private_media/032_ep34158_turn03_review_board.jpg",
            "representative_cross_floor_board":
                "private_media/009_ep23895_turn03_review_board.jpg",
            "representative_separated_p_b_board":
                "private_media/000_ep41233_turn01_review_board.jpg",
            "performed_by_main_agent": True,
            "semantic_labels_assigned": False,
        },
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "checks": "%d/%d" % (output["checks_passed"],
                                output["checks_total"]),
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
