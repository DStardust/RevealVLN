#!/usr/bin/env python3
"""Validate CR5 3x12 multi-view preflight inputs and provenance."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import cv2


ROOT = Path("/mnt/daiyang/vla")
INPUT = ROOT / (
    "artifacts/phase0/phase0c_cr5_preflight/multiview_branch/"
    "CR5_MULTIVIEW_PREFLIGHT_INPUTS.json"
)
OUT = INPUT.with_name("CR5_MULTIVIEW_PREFLIGHT_INPUTS_ACCEPTANCE.json")
ROLES = ("A", "Q", "D")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def safe_file(relative: str) -> Path:
    path = ROOT / relative
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or ROOT.resolve() not in path.resolve().parents):
        raise RuntimeError("unsafe path: " + relative)
    if {"val_unseen", "test", "test_challenge"} & set(path.parts):
        raise RuntimeError("forbidden split path: " + relative)
    return path


def main() -> int:
    failures = []

    def check(condition, label):
        if not condition:
            failures.append(label)

    manifest = json.loads(INPUT.read_text())
    check(manifest.get("status") == "READY_FOR_VALIDATION", "status")
    check(manifest.get("source_scope") == "RxR train only", "scope")
    check(manifest.get("event_count") == 35, "event_count")
    check(manifest.get("network_calls_made") == 0, "no_network")
    check(manifest.get("branch_labels_created") == 0, "no_labels")
    check(manifest.get("geometry_verified_candidates") == 0,
          "no_geometry_claim")
    check(manifest.get("training_authorized") is False, "no_training")
    check(manifest["rendering"].get("positive_habitat_yaw_is_left") is True,
          "yaw_convention")
    for key in ("aggregated_source", "locator_input_source"):
        record = manifest[key]
        path = safe_file(record["path"])
        check(sha256_file(path) == record["sha256"], key + ":sha")
    for key in ("prompt", "schema"):
        record = manifest["contract"]
        path = safe_file(record[key + "_path"])
        check(sha256_file(path) == record[key + "_sha256"], key + ":sha")
        if key == "schema":
            json.loads(path.read_text())

    media = {}
    for record in manifest["media_manifest"]:
        relative = record["path"]
        check(relative not in media, "duplicate_media:" + relative)
        try:
            path = safe_file(relative)
            check(path.stat().st_size == record["bytes"],
                  "media_bytes:" + relative)
            check(sha256_file(path) == record["sha256"],
                  "media_sha:" + relative)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            check(image is not None and image.size > 0,
                  "media_decode:" + relative)
        except (OSError, RuntimeError) as exc:
            failures.append("unsafe_media:%s:%s" % (relative, exc))
        media[relative] = record
    check(len(media) == manifest["media_file_count"], "media_count")
    check(sum(record["bytes"] for record in media.values()) ==
          manifest["media_total_bytes"], "media_total_bytes")

    event_ids = []
    view_count = 0
    for event in manifest["events"]:
        event_id = event["event_id"]
        event_ids.append(event_id)
        label = event_id
        check(event.get("locator_free_text_in_model_input") is False,
              label + ":no_locator_text")
        check(event.get("legacy_bt_in_model_input") is False,
              label + ":no_legacy_bt")
        check(event.get("mllm_branch_proposal") is None,
              label + ":uncalled")
        check(event.get("geometry_verified") is False,
              label + ":geometry_unverified")
        check(event.get("training_label") is False,
              label + ":not_label")
        check(hashlib.sha256(event["instruction_text"].encode("utf-8"))
              .hexdigest() == event["instruction_sha256"],
              label + ":instruction_sha")
        center = int(event["candidate_interval"][
            "representative_center_frame_id"][1:])
        role_prefixes = []
        for role in ROLES:
            record = event["positions"][role]
            role_prefixes.append(record["trace_prefix"])
            check(record["frame_id"] == "P%04d" % record["trace_prefix"],
                  label + ":" + role + ":frame_id")
            if role == "Q":
                check(record["trace_prefix"] == center,
                      label + ":Q_center")
            views = record["views"]
            check(len(views) == 12, label + ":" + role + ":view_count")
            check([value["view_id"] for value in views] ==
                  ["%s_V%02d" % (role, index) for index in range(12)],
                  label + ":" + role + ":view_ids")
            expected_yaws = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0,
                             180.0, -150.0, -120.0, -90.0, -60.0, -30.0]
            check([value["relative_yaw_deg"] for value in views] ==
                  expected_yaws, label + ":" + role + ":yaw_values")
            for value in views:
                check(media.get(value["path"], {}).get("sha256") ==
                      value["sha256"],
                      label + ":" + role + ":view_media")
            board = record["contact_sheet"]
            check(media.get(board["path"], {}).get("sha256") ==
                  board["sha256"], label + ":" + role + ":board_media")
            check(board["view_ids"] == [value["view_id"] for value in views],
                  label + ":" + role + ":board_ids")
            view_count += len(views)
        check(role_prefixes[0] <= role_prefixes[1] <= role_prefixes[2],
              label + ":role_order")
        contexts = event["chronological_context_frames"]
        context_ids = [value["frame_id"] for value in contexts]
        check(context_ids == sorted(context_ids, key=lambda value: int(value[1:])),
              label + ":context_order")
        check(len(context_ids) == len(set(context_ids)),
              label + ":context_unique")
        check(event["candidate_interval"][
            "representative_center_frame_id"] in context_ids,
              label + ":center_in_context")
        for value in contexts:
            path = safe_file(value["path"])
            check(path.stat().st_size == value["bytes"]
                  and sha256_file(path) == value["sha256"],
                  label + ":context_integrity")

    check(len(event_ids) == len(set(event_ids)) == 35, "event_identity")
    check(view_count == 35 * 3 * 12, "total_view_count")
    output = {
        "gate": "MF2-CR5 multi-view preflight inputs",
        "status": "PASS" if not failures else "FAIL",
        "input_path": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256_file(INPUT),
        "checks": {
            "events": len(event_ids),
            "panorama_views": view_count,
            "media_files": len(media),
            "media_total_bytes": sum(value["bytes"]
                                     for value in media.values()),
            "failures": failures,
        },
        "offline_multiview_only": True,
        "online_causal_labels_created": 0,
        "geometry_labels_created": 0,
        "training_authorized": False,
    }
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "checks": output["checks"],
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
