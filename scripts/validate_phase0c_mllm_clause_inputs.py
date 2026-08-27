#!/usr/bin/env python3
"""Independently validate private RxR-train MLLM clause-grounding inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from PIL import Image


ROOT = Path("/mnt/daiyang/vla")
ARTIFACT_DIR = ROOT / "artifacts/phase0/phase0c_clause_grounding_mllm"
INPUT = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_INPUTS.json"
OUTPUT = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_INPUTS_ACCEPTANCE.json"
EXPECTED_SHA256 = (
    "d576b49122b7b3a90c71d6ff6648926afd6b063f433e8feb9a0f85488bb1f4ca"
)
EXPECTED = {
    "event_count": 35,
    "episode_count": 25,
    "scene_count": 22,
    "media_file_count": 893,
    "media_total_bytes": 40900561,
}
FRAME_RE = re.compile(r"^EP(?P<episode>[0-9]+)_P(?P<prefix>[0-9]{4})$")
SEGMENT_RE = re.compile(r"^S(?P<index>[0-9]{2})$")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def regular_project_file(path: Path) -> bool:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and ROOT.resolve() in resolved.parents
    )


def atomic_json(path: Path, value) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)


def main() -> int:
    failures: list[str] = []
    check(regular_project_file(INPUT), "input_not_regular_project_file", failures)
    actual_input_sha = sha256_file(INPUT) if INPUT.is_file() else None
    check(actual_input_sha == EXPECTED_SHA256, "input_sha256_drift", failures)
    if failures:
        atomic_json(OUTPUT, {"status": "FAIL", "failures": failures})
        return 1

    manifest = json.loads(INPUT.read_text())
    check(manifest.get("revision") ==
          "phase0c-mllm-clause-inputs/1-ordered-images",
          "revision", failures)
    check(manifest.get("status") == "READY_FOR_MLLM_PROPOSAL_UNCALLED",
          "status", failures)
    check(manifest.get("source_scope") == "RxR train only", "scope", failures)
    check(manifest.get("rxr_train", {}).get("path") ==
          ("third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/"
           "train/train_guide.json.gz"),
          "rxr_train_source", failures)
    check(manifest.get("network_calls_made") == 0, "network_calls", failures)
    check(manifest.get("training_authorized") is False,
          "training_authorized", failures)
    check(manifest.get("private_distribution_authorized") is False,
          "private_distribution", failures)
    for key, expected in EXPECTED.items():
        check(manifest.get(key) == expected, key, failures)

    contract = manifest.get("model_request_contract", {})
    check(contract.get("provider") == "DashScope OpenAI-compatible",
          "provider", failures)
    check(contract.get("base_url") ==
          "https://dashscope.aliyuncs.com/compatible-mode/v1",
          "base_url", failures)
    check(contract.get("model") == "qwen3.8-max", "model", failures)
    check(contract.get("input") ==
          "chronological image_url data-URI sequence plus text",
          "input_contract", failures)
    check(contract.get("temperature") == 0, "temperature", failures)
    check(contract.get("proposal_is_ground_truth") is False,
          "proposal_ground_truth", failures)
    prompt = ROOT / contract.get("system_prompt_path", "__missing__")
    check(regular_project_file(prompt), "prompt_file", failures)
    if regular_project_file(prompt):
        check(sha256_file(prompt) == contract.get("system_prompt_sha256"),
              "prompt_sha256", failures)

    media = manifest.get("media_manifest", [])
    check(len(media) == EXPECTED["media_file_count"], "media_length", failures)
    media_by_id = {}
    observed_bytes = 0
    media_episodes = set()
    for index, record in enumerate(media):
        label = f"media[{index}]"
        frame_id = record.get("frame_id")
        match = FRAME_RE.fullmatch(frame_id or "")
        check(match is not None, label + ":frame_id", failures)
        check(frame_id not in media_by_id, label + ":duplicate_frame_id", failures)
        path = ROOT / record.get("path", "__missing__")
        check(regular_project_file(path), label + ":path", failures)
        if not regular_project_file(path):
            continue
        check(path.suffix.lower() == ".jpg", label + ":suffix", failures)
        size = path.stat().st_size
        observed_bytes += size
        check(size == record.get("bytes"), label + ":bytes", failures)
        check(sha256_file(path) == record.get("sha256"), label + ":sha256", failures)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                check(image.size == (448, 448), label + ":pixels", failures)
                check(image.mode == "RGB", label + ":mode", failures)
        except Exception as exc:
            failures.append(label + ":decode:" + type(exc).__name__)
        if match:
            check(record.get("episode_id") == match.group("episode"),
                  label + ":episode", failures)
            check(record.get("prefix_index") == int(match.group("prefix")),
                  label + ":prefix", failures)
            media_episodes.add(match.group("episode"))
        check(record.get("pixels") == [448, 448], label + ":manifest_pixels",
              failures)
        check(isinstance(record.get("action"), str), label + ":action", failures)
        media_by_id[frame_id] = record
    check(observed_bytes == EXPECTED["media_total_bytes"],
          "media_total_bytes_observed", failures)

    events = manifest.get("events", [])
    check(len(events) == EXPECTED["event_count"], "event_length", failures)
    event_ids = set()
    event_episodes = set()
    event_scenes = set()
    for row_order, event in enumerate(events):
        label = f"event[{row_order}]"
        check(event.get("row_order") == row_order, label + ":row_order", failures)
        event_id = event.get("event_id")
        check(isinstance(event_id, str) and event_id not in event_ids,
              label + ":event_id", failures)
        event_ids.add(event_id)
        episode_id = event.get("episode_id")
        event_episodes.add(episode_id)
        event_scenes.add(event.get("scene_id"))
        instruction = event.get("instruction_text")
        check(isinstance(instruction, str) and instruction,
              label + ":instruction", failures)
        if isinstance(instruction, str):
            check(sha256_text(instruction) == event.get("instruction_sha256"),
                  label + ":instruction_sha256", failures)
        segments = event.get("deterministic_segments", [])
        check(bool(segments), label + ":segments_empty", failures)
        prior_end = 0
        for segment_index, segment in enumerate(segments, 1):
            segment_label = f"{label}:segment[{segment_index}]"
            segment_id = segment.get("segment_id")
            match = SEGMENT_RE.fullmatch(segment_id or "")
            check(match is not None and int(match.group("index")) == segment_index,
                  segment_label + ":id", failures)
            start = segment.get("char_start")
            end = segment.get("char_end_exclusive")
            check(isinstance(start, int) and isinstance(end, int)
                  and 0 <= start < end <= len(instruction),
                  segment_label + ":span", failures)
            if isinstance(start, int) and isinstance(end, int) \
                    and 0 <= start < end <= len(instruction):
                text = instruction[start:end]
                check(text == segment.get("text"), segment_label + ":exact_text",
                      failures)
                check(sha256_text(text) == segment.get("text_sha256"),
                      segment_label + ":sha256", failures)
                check(start >= prior_end, segment_label + ":order", failures)
                prior_end = end
        segmentation = event.get("segmentation_contract", {})
        check(segmentation.get("official_rxr_alignment") is False,
              label + ":not_official", failures)
        check(segmentation.get("model_may_rewrite_segments") is False,
              label + ":no_rewrite", failures)
        check(segmentation.get("maximum_adjacent_selection") == 3,
              label + ":max_adjacent", failures)
        sequence = event.get("sequence_frame_ids", [])
        check(20 <= len(sequence) <= 34, label + ":sequence_length", failures)
        check(len(sequence) == len(set(sequence)), label + ":sequence_unique",
              failures)
        prefix_indices = []
        for frame_id in sequence:
            record = media_by_id.get(frame_id)
            check(record is not None, label + ":missing_frame:" + str(frame_id),
                  failures)
            if record:
                check(record.get("episode_id") == episode_id,
                      label + ":cross_episode_frame", failures)
                prefix_indices.append(record.get("prefix_index"))
        check(prefix_indices == sorted(prefix_indices), label + ":chronology",
              failures)
        check(event.get("sequence_is_chronological") is True,
              label + ":chronology_flag", failures)
        roles = event.get("causal_frame_roles", {})
        check(set(roles) == {"pre_reveal", "d1", "d2", "d3"},
              label + ":causal_roles", failures)
        check(all(value in sequence for value in roles.values()),
              label + ":causal_coverage", failures)
        check(event.get("mllm_proposal") is None, label + ":mllm_blank", failures)
        check(event.get("human_judgment") is None, label + ":human_blank", failures)

    check(len(event_episodes) == EXPECTED["episode_count"],
          "event_episode_count", failures)
    check(len(event_scenes) == EXPECTED["scene_count"],
          "event_scene_count", failures)
    check(media_episodes == event_episodes, "media_event_episode_set", failures)
    check(set(manifest.get("episodes", {})) == event_episodes,
          "episode_metadata_set", failures)
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    check(free_bytes >= 8 * 1024**3, "disk_below_8GiB", failures)
    status = "PASS" if not failures else "FAIL"
    output = {
        "status": status,
        "validator_revision": "phase0c-mllm-input-acceptance/1",
        "input_manifest_sha256": actual_input_sha,
        "counts": {
            "events": len(events),
            "episodes": len(event_episodes),
            "scenes": len(event_scenes),
            "media_files": len(media),
            "media_bytes": observed_bytes,
        },
        "checks": {
            "media_regular_non_symlink_project_local_sha_decode": not any(
                value.startswith("media[") for value in failures),
            "exact_instruction_segments": not any(
                ":segment[" in value for value in failures),
            "chronological_causal_frame_coverage": not any(
                any(token in value for token in (
                    ":chronology", ":causal_", ":missing_frame",
                    ":cross_episode_frame")) for value in failures),
            "mllm_and_human_labels_blank": not any(
                value.endswith((":mllm_blank", ":human_blank"))
                for value in failures),
            "no_network_no_training": manifest.get("network_calls_made") == 0
            and manifest.get("training_authorized") is False,
        },
        "free_bytes": free_bytes,
        "failures": failures,
        "training_authorized": False,
    }
    atomic_json(OUTPUT, output)
    print(json.dumps({
        "status": status,
        "failures": len(failures),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
    }, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
