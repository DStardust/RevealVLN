#!/usr/bin/env python3
"""Fail-closed integrity acceptance for frozen queue50 hindsight inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
INPUT = ROOT / (
    "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/"
    "CR5_QUEUE50_HINDSIGHT_INPUTS.json"
)
QUEUE = ROOT / "artifacts/phase0/REVEAL_QUEUE_50_MAPPING.json"
OUT = INPUT.with_name("CR5_QUEUE50_HINDSIGHT_INPUTS_ACCEPTANCE.json")
EXPECTED_INPUT_SHA = "8e00000ee306369e305c53d580444e1ac3228a6e94c3c424d84f9db5d16ea151"
EXPECTED_QUEUE_SHA = "fe8dfd9c3af01a67a28035787fdfbe4844ca68c47ca1c7c9f363361c6c331ece"


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)
                         + "\n")
    os.replace(temporary, path)


def main():
    failures = []
    if (not INPUT.is_file() or INPUT.is_symlink()
            or sha256_file(INPUT) != EXPECTED_INPUT_SHA):
        failures.append("input manifest SHA/safety")
    if (not QUEUE.is_file() or QUEUE.is_symlink()
            or sha256_file(QUEUE) != EXPECTED_QUEUE_SHA):
        failures.append("queue mapping SHA/safety")
    manifest = json.loads(INPUT.read_text())
    queue = json.loads(QUEUE.read_text())
    items = sorted(queue["items"], key=lambda row: row["queue_order"])
    episodes = manifest.get("episodes", [])
    if (manifest.get("status") != "READY_FOR_MLLM_DRY_RUN"
            or manifest.get("episode_count") != 50
            or len(episodes) != 50
            or [row.get("queue_order") for row in episodes]
            != list(range(50))):
        failures.append("50-episode ordered closure")
    for episode, item in zip(episodes, items):
        if any((episode.get("episode_id") != str(item["episode_id"]),
                episode.get("trajectory_id") != item["trajectory_id"],
                episode.get("instruction_id") != item["instruction_id"],
                episode.get("scene_id") != item["scene_id"],
                episode.get("language") != item["language"],
                episode.get("instruction_sha256") !=
                item["instruction_sha256_queue"])):
            failures.append("queue identity: order%02d" %
                            item["queue_order"])
        timeline = episode.get("timeline_prefix_indices", [])
        frame_ids = episode.get("timeline_frame_ids", [])
        if (not timeline or timeline != sorted(set(timeline))
                or frame_ids != ["P%04d" % value for value in timeline]
                or timeline[0] != 0
                or timeline[-1] != episode.get("trace_length") - 1):
            failures.append("timeline contract: " + episode["episode_id"])
        chunks = episode.get("chunks", [])
        covered = set()
        for index, chunk in enumerate(chunks):
            chunk_ids = chunk.get("frame_ids", [])
            if (not chunk_ids or len(chunk_ids) > 20
                    or chunk_ids != frame_ids[
                        chunk["timeline_offset_start"]:
                        chunk["timeline_offset_end_exclusive"]]
                    or len(chunk_ids) != len(chunk.get("frame_paths", []))):
                failures.append("chunk contract: %s/%s" %
                                (episode["episode_id"], chunk["chunk_id"]))
            if index and chunk.get("overlap_with_previous", 0) < 5:
                failures.append("chunk overlap: %s/%s" %
                                (episode["episode_id"], chunk["chunk_id"]))
            covered.update(chunk_ids)
        if covered != set(frame_ids):
            failures.append("chunk coverage: " + episode["episode_id"])

    media = manifest.get("media_manifest", [])
    if (manifest.get("media_file_count") != len(media)
            or len(media) != 2107
            or manifest.get("media_total_bytes") !=
            sum(row.get("bytes", -1) for row in media)
            or len({row.get("path") for row in media}) != len(media)):
        failures.append("media manifest aggregate")
    media_root = INPUT.parent / "private_media"
    forbidden = {"val_unseen", "test", "test_challenge"}
    for record in media:
        path = ROOT / record["path"]
        try:
            safe = (path.is_file() and not path.is_symlink()
                    and media_root.resolve() in path.resolve().parents
                    and ROOT.resolve() in path.resolve().parents
                    and not forbidden.intersection(path.parts)
                    and path.stat().st_size == record["bytes"]
                    and sha256_file(path) == record["sha256"])
        except OSError:
            safe = False
        if not safe:
            failures.append("unsafe/drifted media: " + record["path"])
            break
    if (manifest.get("network_calls_made") != 0
            or manifest.get("branch_labels_created") != 0
            or manifest.get("human_labels_created") != 0
            or manifest.get("training_authorized") is not False
            or manifest.get("future_frames_are_offline_annotation_only")
            is not True
            or manifest.get("online_prefix_causality_verified") is not False):
        failures.append("scope boundary")
    output = {
        "revision": "cr5-queue50-hindsight-input-acceptance/1",
        "status": "PASS" if not failures else "FAIL",
        "input_path": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256_file(INPUT),
        "queue_mapping_sha256": sha256_file(QUEUE),
        "episode_count": len(episodes),
        "chunk_count": sum(len(row["chunks"]) for row in episodes),
        "media_file_count": len(media),
        "media_total_bytes": sum(row["bytes"] for row in media),
        "all_media_project_local_regular_hash_verified": not any(
            value.startswith("unsafe/drifted media") for value in failures),
        "future_frames_authorized_only_for_offline_locator": True,
        "online_labels_created": 0,
        "human_labels_created": 0,
        "training_authorized": False,
        "failures": failures,
    }
    atomic_json(OUT, output)
    print(json.dumps({**output, "output": str(OUT.relative_to(ROOT)),
                      "output_sha256": sha256_file(OUT)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
