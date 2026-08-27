#!/usr/bin/env python3
"""Fail-closed validator for CR5 full-trajectory preflight inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

import cv2


ROOT = Path("/mnt/daiyang/vla")
DEFAULT_INPUT = ROOT / (
    "artifacts/phase0/phase0c_cr5_preflight/hindsight_locator/"
    "CR5_HINDSIGHT_PREFLIGHT_INPUTS_V2.json"
)
EPISODES = {"41233", "34121", "46758", "43805", "7619", "56443"}
FORBIDDEN_PARTS = {"val_unseen", "test", "test_challenge"}


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
    if FORBIDDEN_PARTS & set(Path(relative).parts):
        raise RuntimeError("forbidden split path: " + relative)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    if ROOT.resolve() not in input_path.resolve().parents:
        raise SystemExit("input resolves outside project")
    output_path = input_path.with_name(input_path.stem + "_ACCEPTANCE.json")
    failures = []

    def check(condition, label):
        if not condition:
            failures.append(label)

    manifest = json.loads(input_path.read_text())
    check(manifest.get("source_scope") == "RxR train only", "train_scope")
    check(manifest.get("episode_count") == 6, "episode_count")
    check(manifest.get("network_calls_made") == 0, "no_network")
    check(manifest.get("branch_labels_created") == 0, "no_labels")
    check(manifest.get("training_authorized") is False, "no_training")
    check(manifest.get("future_frames_are_offline_annotation_only") is True,
          "hindsight_boundary")

    for key in ("prompt", "schema"):
        item = manifest["contract"]
        path = safe_file(item[key + "_path"])
        check(path.stat().st_size > 0, key + "_nonempty")
        check(sha256_file(path) == item[key + "_sha256"], key + "_sha")
        if key == "schema":
            json.loads(path.read_text())

    rxr = manifest["rxr_train"]
    rxr_path = safe_file(rxr["path"])
    check(rxr_path.stat().st_size == rxr["bytes"], "rxr_bytes")
    check(sha256_file(rxr_path) == rxr["sha256"], "rxr_sha")

    rows = manifest.get("episodes", [])
    check({row.get("episode_id") for row in rows} == EPISODES,
          "episode_identity")
    seen_media = {}
    for record in manifest.get("media_manifest", []):
        relative = record["path"]
        check(relative not in seen_media, "duplicate_media:" + relative)
        try:
            path = safe_file(relative)
            check(path.stat().st_size == record["bytes"],
                  "media_bytes:" + relative)
            check(sha256_file(path) == record["sha256"],
                  "media_sha:" + relative)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            check(image is not None and image.size > 0,
                  "media_decode:" + relative)
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            failures.append("media_unsafe:%s:%s" % (relative, exc))
        seen_media[relative] = record

    check(len(seen_media) == manifest.get("media_file_count"),
          "media_count")
    check(sum(row["bytes"] for row in seen_media.values()) ==
          manifest.get("media_total_bytes"), "media_total_bytes")

    for row in rows:
        label = "ep" + row["episode_id"]
        timeline = row["timeline_frame_ids"]
        prefixes = row["timeline_prefix_indices"]
        check(len(timeline) == len(prefixes) == len(set(timeline)),
              label + ":timeline_cardinality")
        check(prefixes == sorted(prefixes) and prefixes[0] == 0
              and prefixes[-1] == row["trace_length"] - 1,
              label + ":timeline_order")
        check(timeline == ["P%04d" % value for value in prefixes],
              label + ":frame_id_mapping")
        check(row.get("legacy_target_fields_in_model_input") is False,
              label + ":no_legacy_target")
        check(row.get("mllm_output") is None, label + ":uncalled")
        check(row["instruction_sha256"] == hashlib.sha256(
            row["instruction_text"].encode("utf-8")).hexdigest(),
              label + ":instruction_sha")
        segment_ids = [value["segment_id"]
                       for value in row["deterministic_segments"]]
        check(segment_ids == ["S%02d" % (index + 1)
                              for index in range(len(segment_ids))],
              label + ":segments_order")
        for value in row["deterministic_segments"]:
            source = row["instruction_text"][
                value["char_start"]:value["char_end_exclusive"]]
            check(source == value["text"], label + ":segment_exact")
            check(hashlib.sha256(source.encode("utf-8")).hexdigest() ==
                  value["text_sha256"], label + ":segment_sha")

        global_storyboard = row["global_storyboard"]
        check(all(value in timeline
                  for value in global_storyboard["frame_ids"]),
              label + ":global_subset")
        check(global_storyboard["frame_ids"] == sorted(
            global_storyboard["frame_ids"], key=timeline.index),
              label + ":global_order")
        global_path = safe_file(global_storyboard["path"])
        check(sha256_file(global_path) == global_storyboard["sha256"],
              label + ":global_sha")

        chunks = row["chunks"]
        check([value["chunk_id"] for value in chunks] ==
              ["C%02d" % index for index in range(len(chunks))],
              label + ":chunk_ids")
        union = set()
        previous = None
        for chunk in chunks:
            c_label = label + ":" + chunk["chunk_id"]
            frame_ids = chunk["frame_ids"]
            start = chunk["timeline_offset_start"]
            end = chunk["timeline_offset_end_exclusive"]
            check(frame_ids == timeline[start:end], c_label + ":slice")
            check(len(frame_ids) <= manifest["chunking"]["max_frames"],
                  c_label + ":max_frames")
            check(len(frame_ids) == len(chunk["frame_paths"]),
                  c_label + ":path_count")
            for frame_id, relative in zip(frame_ids,
                                          chunk["frame_paths"]):
                record = seen_media.get(relative, {})
                check(record.get("frame_id") == frame_id,
                      c_label + ":frame_path_mapping")
            board_path = safe_file(chunk["storyboard_path"])
            check(sha256_file(board_path) == chunk["storyboard_sha256"],
                  c_label + ":board_sha")
            if previous is None:
                check(chunk["overlap_with_previous"] == 0,
                      c_label + ":first_overlap")
            else:
                overlap = len(set(frame_ids) & set(previous))
                check(overlap == chunk["overlap_with_previous"]
                      and overlap >= manifest["chunking"][
                          "minimum_overlap_frames"],
                      c_label + ":overlap")
            previous = frame_ids
            union.update(frame_ids)
        check(union == set(timeline), label + ":chunk_coverage")

    expected_total = sum(value["bytes"] for value in seen_media.values())
    output = {
        "gate": "MF2-CR5 full-trajectory hindsight preflight inputs",
        "status": "PASS" if not failures else "FAIL",
        "input_path": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256_file(input_path),
        "checks": {
            "episodes": len(rows),
            "chunks": sum(len(row["chunks"]) for row in rows),
            "timeline_frames": sum(len(row["timeline_frame_ids"])
                                   for row in rows),
            "media_files": len(seen_media),
            "media_total_bytes": expected_total,
            "failures": failures,
        },
        "model_inputs_contain_legacy_bt": False,
        "future_frames_authorized_only_for_offline_locator": True,
        "branch_labels_created": 0,
        "training_authorized": False,
    }
    part = output_path.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, output_path)
    print(json.dumps({
        "status": output["status"],
        "checks": output["checks"],
        "output": str(output_path.relative_to(ROOT)),
        "output_sha256": sha256_file(output_path),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
