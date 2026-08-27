#!/usr/bin/env python3
"""Render contact-sheet-only A/Q/D panoramas for expansion primaries."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2

import build_phase0c_cr5_hindsight_preflight as timeline_base
import build_phase0c_cr5_multiview_preflight as view_base


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
CANDIDATES = BASE / (
    "hindsight_factory/RXR_HINDSIGHT_EVENT_CANDIDATES.json")
QUEUE = BASE / "RXR_TRAIN_UNBIASED_EXPANSION_QUEUE.json"
RUNTIME = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz")
PROMPT = BASE / (
    "contract/RXR_MULTIVIEW_BRANCH_PROPOSAL_PROMPT_V3.md")
SCHEMA = ROOT / (
    "artifacts/phase0/phase0c_cr5_contract/"
    "CR5_MLLM_BRANCH_PROPOSAL_SCHEMA.json")
OUT_DIR = BASE / "multiview_factory"
MEDIA_DIR = OUT_DIR / "panoramas"
SHARD_DIR = OUT_DIR / "shards"
TMP_DIR = OUT_DIR / "tmp"
OUT = OUT_DIR / "RXR_PRIMARY_MULTIVIEW_INPUTS.json"
EXPECTED = {
    QUEUE: "7b3578afae71dc35327c9ad31b4a97df1a3ccd4960109a2e1fd78f4fa4facbab",
    RUNTIME: "f06b2ef4dc947ca15d6c4a5a3d629c9212328f4cbdd38a13bed9c5c1fc224a94",
    PROMPT: "448643e06acfbb0b104fdf434ac4e971c37240598ec433a249ea598723572073",
    SCHEMA: "d3c76ee4c26b47f9f9b3d03d9a1244d2dad6565331e44e014ff823514f8e5f33",
}
SHARDS = 28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_image(path: Path, image) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.jpg")
    if not cv2.imwrite(str(temporary), image, [
            int(cv2.IMWRITE_JPEG_QUALITY), view_base.JPEG_QUALITY]):
        raise RuntimeError("failed to encode panorama contact sheet")
    new_sha = sha256_file(temporary)
    if path.exists():
        if path.is_symlink() or sha256_file(path) != new_sha:
            temporary.unlink()
            raise RuntimeError("existing panorama differs: " + str(path))
        temporary.unlink()
    else:
        os.replace(temporary, path)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": new_sha,
        "pixels": [int(image.shape[1]), int(image.shape[0])],
    }


def sources():
    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected):
            raise RuntimeError("pinned multiview source drift: " + str(path))
    if not CANDIDATES.is_file() or CANDIDATES.is_symlink():
        raise RuntimeError("hindsight candidates are not ready")
    source_sha = sha256_file(CANDIDATES)
    document = json.loads(CANDIDATES.read_text())
    if (document["status"] != "PASS_PENDING_MULTIVIEW_AND_3D_GATES"
            or document["trajectory_count"] != 2303
            or document["human_labels_created"] != 0
            or document["training_authorized"] is not False):
        raise RuntimeError("hindsight candidate source contract failed")
    candidates = {row["hindsight_candidate_id"]: row
                  for row in document["candidates"]}
    primaries = []
    for plan in document["cascade_review_plan"]:
        event_id = plan["primary_candidate_id"]
        if event_id is not None:
            if event_id not in candidates:
                raise RuntimeError("primary candidate closure failure")
            primaries.append(candidates[event_id])
    if len({row["expansion_order"] for row in primaries}) != len(primaries):
        raise RuntimeError("duplicate primary trajectory")
    return source_sha, document, primaries


def context_storyboards(candidate: dict, result: dict) -> list[dict]:
    request = result["request_evidence"]
    timeline = request["timeline_prefix_indices"]
    center = view_base.prefix_number(candidate["interval"][
        "representative_center_frame_id"])
    media = {Path(row["path"]).name: row for row in request["media"]}
    selected = []
    for index, (start, end) in enumerate(timeline_base.chunk_ranges(
            len(timeline))):
        prefixes = timeline[start:end]
        if center not in prefixes:
            continue
        name = "C%02d.jpg" % index
        if name not in media:
            raise RuntimeError("missing persisted context storyboard")
        record = dict(media[name])
        record.update({
            "chunk_id": "C%02d" % index,
            "timeline_offset_start": start,
            "timeline_offset_end_exclusive": end,
            "frame_ids": ["P%04d" % value for value in prefixes],
        })
        path = ROOT / record["path"]
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]):
            raise RuntimeError("context storyboard drift")
        selected.append(record)
    if not selected or len(selected) > 2:
        raise RuntimeError("candidate center context closure failure")
    return selected


def build_event(candidate: dict, queue_row: dict, episode: dict,
                sim, gpu: int) -> tuple[dict, list[dict]]:
    result_path = ROOT / candidate["source"]["proposal_path"]
    if (not result_path.is_file() or result_path.is_symlink()
            or sha256_file(result_path) !=
            candidate["source"]["proposal_sha256"]):
        raise RuntimeError("hindsight proposal source drift")
    result = json.loads(result_path.read_text())
    request = result["request_evidence"]
    trace = view_base.build_lowlevel_trace(sim.pathfinder, episode)
    if not trace:
        raise RuntimeError("empty low-level trace at multiview gate")
    center = view_base.prefix_number(candidate["interval"][
        "representative_center_frame_id"])
    if not 0 <= center < len(trace):
        raise RuntimeError("candidate center outside low-level trace")
    a_prefix, a_distance = view_base.position_prefix(
        trace, center, -1, view_base.POSITION_DISTANCE_M)
    d_prefix, d_distance = view_base.position_prefix(
        trace, center, 1, view_base.POSITION_DISTANCE_M)
    roles = [("A", a_prefix, a_distance), ("Q", center, 0.0),
             ("D", d_prefix, d_distance)]
    event_dir = MEDIA_DIR / candidate["hindsight_candidate_id"]
    role_records = {}
    media = []
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=candidate["hindsight_candidate_id"] + "_", dir=TMP_DIR) \
            as temporary_name:
        temporary_dir = Path(temporary_name)
        for role, trace_prefix, achieved in roles:
            route_heading = view_base.route_forward_heading(trace, trace_prefix)
            view_paths = []
            views = []
            for view_index in range(view_base.PANORAMA_HEADINGS):
                view_id = "%s_V%02d" % (role, view_index)
                offset = view_base.relative_yaw(view_index)
                heading = (route_heading + math.radians(offset)) % (
                    2 * math.pi)
                direction = "LEFT" if offset > 0 else (
                    "RIGHT" if offset < 0 else "FORWARD")
                label = "%s | %+d deg %s" % (
                    view_id, int(offset), direction)
                path = temporary_dir / (view_id + ".jpg")
                image = view_base.render(
                    sim, trace[trace_prefix]["position"], heading, label)
                if not cv2.imwrite(str(path), image, [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        view_base.JPEG_QUALITY]):
                    raise RuntimeError("failed to encode temporary view")
                view_paths.append(path)
                views.append({
                    "view_id": view_id,
                    "relative_yaw_deg": offset,
                    "habitat_heading_rad": round(heading, 6),
                })
            board = view_base.contact_sheet(
                view_paths, [row["view_id"] for row in views])
            board_record = write_image(
                event_dir / (role + "_PANORAMA.jpg"), board)
            board_record.update({
                "kind": "panorama_contact_sheet",
                "event_id": candidate["hindsight_candidate_id"],
                "role": role,
                "view_ids": [row["view_id"] for row in views],
            })
            media.append(board_record)
            role_records[role] = {
                "trace_prefix": trace_prefix,
                "frame_id": view_base.prefix_id(trace_prefix),
                "requested_route_distance_m":
                    view_base.POSITION_DISTANCE_M if role != "Q" else 0.0,
                "achieved_route_distance_m": round(achieved, 6),
                "position_q": view_base.q(trace[trace_prefix]["position"]),
                "route_forward_heading_rad": round(route_heading, 6),
                "views": views,
                "contact_sheet": board_record,
            }
    contexts = context_storyboards(candidate, result)
    context_ids = sorted({frame_id for row in contexts
                          for frame_id in row["frame_ids"]},
                         key=lambda value: int(value[1:]))
    event = {
        "event_id": candidate["hindsight_candidate_id"],
        "expansion_order": candidate["expansion_order"],
        "episode_id": candidate["episode_id"],
        "trajectory_id": candidate["trajectory_id"],
        "scene_id": candidate["scene_id"],
        "instruction_id": candidate["instruction_id"],
        "language": queue_row["language"],
        "instruction_text": queue_row["instruction_text"],
        "instruction_sha256": queue_row["instruction_sha256"],
        "deterministic_segments": request["deterministic_segments"],
        "candidate_interval": candidate["interval"],
        "candidate_center_source":
            "full_trajectory_hindsight_primary_shortlist",
        "locator_free_text_in_model_input": False,
        "legacy_bt_in_model_input": False,
        "positions": role_records,
        "chronological_context_storyboards": contexts,
        "chronological_context_frame_ids": context_ids,
        "mllm_branch_proposal": None,
        "geometry_verified": False,
        "human_reviewed": False,
        "training_label": False,
    }
    return event, media


def run_shard(index: int, gpu: int) -> int:
    if not 0 <= index < SHARDS:
        raise SystemExit("invalid shard index")
    source_sha, _, primaries = sources()
    selected = [row for row in primaries
                if row["expansion_order"] % SHARDS == index]
    queue = {row["expansion_order"]: row
             for row in json.loads(QUEUE.read_text())["candidates"]}
    wanted = {row["episode_id"] for row in selected}
    with gzip.open(RUNTIME, "rt", encoding="utf-8") as handle:
        episodes = {str(row["episode_id"]): row
                    for row in json.load(handle)["episodes"]
                    if str(row["episode_id"]) in wanted}
    if set(episodes) != wanted:
        raise SystemExit("multiview runtime episode closure failure")
    view_base.GPU_DEVICE = gpu
    by_scene = {}
    for candidate in selected:
        by_scene.setdefault(candidate["scene_id"], []).append(candidate)
    events, media, failures = [], [], []
    for scene in sorted(by_scene):
        try:
            sim = view_base.build_sim(scene)
        except Exception as error:
            for candidate in by_scene[scene]:
                failures.append({
                    "event_id": candidate["hindsight_candidate_id"],
                    "expansion_order": candidate["expansion_order"],
                    "failure_stage": "SIMULATOR_CONSTRUCTION",
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                    "replacement_sample_created": False,
                })
            continue
        try:
            for candidate in sorted(by_scene[scene], key=lambda row:
                                    row["expansion_order"]):
                try:
                    event, records = build_event(
                        candidate, queue[candidate["expansion_order"]],
                        episodes[candidate["episode_id"]], sim, gpu)
                    events.append(event)
                    media.extend(records)
                    print("shard", index, "order",
                          candidate["expansion_order"], event["event_id"],
                          flush=True)
                except Exception as error:
                    failures.append({
                        "event_id": candidate["hindsight_candidate_id"],
                        "expansion_order": candidate["expansion_order"],
                        "failure_stage": "PRIMARY_MULTIVIEW_RENDER",
                        "error_type": type(error).__name__,
                        "error": str(error)[:2000],
                        "replacement_sample_created": False,
                    })
        finally:
            sim.close()
    output = {
        "status": "PASS" if not failures else
                  "PASS_WITH_FAIL_CLOSED_INPUT_FAILURES",
        "revision": "rxr-primary-multiview-shard/1-contact-sheets-only",
        "shard_index": index,
        "shard_count": SHARDS,
        "gpu_device_id": gpu,
        "hindsight_candidates_sha256": source_sha,
        "selected_primary_count": len(selected),
        "event_count": len(events),
        "failure_count": len(failures),
        "events": sorted(events, key=lambda row: row["expansion_order"]),
        "failures": sorted(failures, key=lambda row: row["expansion_order"]),
        "media_manifest": sorted(media, key=lambda row: row["path"]),
        "individual_panorama_views_persisted": 0,
        "network_calls_made": 0,
        "branch_labels_created": 0,
        "human_labels_created": 0,
        "replacement_samples_created": 0,
        "training_authorized": False,
    }
    path = SHARD_DIR / ("shard_%02d.json" % index)
    atomic_json(path, output)
    print(json.dumps({
        "status": output["status"], "selected": len(selected),
        "events": len(events), "failures": len(failures),
        "output": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
    }, indent=2))
    return 0


def aggregate() -> int:
    source_sha, _, primaries = sources()
    expected_ids = {row["hindsight_candidate_id"] for row in primaries}
    events, failures, media, shards = [], [], [], []
    for index in range(SHARDS):
        path = SHARD_DIR / ("shard_%02d.json" % index)
        if not path.is_file() or path.is_symlink():
            raise SystemExit("missing multiview shard")
        value = json.loads(path.read_text())
        if (value["status"] not in {
                    "PASS", "PASS_WITH_FAIL_CLOSED_INPUT_FAILURES"}
                or value["shard_index"] != index
                or value["shard_count"] != SHARDS
                or value["hindsight_candidates_sha256"] != source_sha
                or value["selected_primary_count"] !=
                value["event_count"] + value["failure_count"]
                or value["network_calls_made"] != 0
                or value["replacement_samples_created"] != 0):
            raise SystemExit("multiview shard contract failure")
        events.extend(value["events"])
        failures.extend(value["failures"])
        media.extend(value["media_manifest"])
        shards.append({"path": str(path.relative_to(ROOT)),
                       "sha256": sha256_file(path),
                       "event_count": value["event_count"],
                       "failure_count": value["failure_count"]})
    observed_ids = {row["event_id"] for row in events} | {
        row["event_id"] for row in failures}
    if observed_ids != expected_ids or len(observed_ids) != len(primaries):
        raise SystemExit("multiview primary exact closure failure")
    if len(media) != len(events) * 3 or len({row["path"] for row in media}) \
            != len(media):
        raise SystemExit("contact-sheet media closure failure")
    for record in media:
        path = ROOT / record["path"]
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]):
            raise SystemExit("contact-sheet media drift")
    output = {
        "manifest": "RevealNav RxR expansion primary multi-view inputs",
        "revision": "rxr-primary-multiview-inputs/1-contact-sheets-only",
        "status": "READY_FOR_BRANCH_PROPOSER",
        "sources": {
            "hindsight_candidates": {
                "path": str(CANDIDATES.relative_to(ROOT)),
                "sha256": source_sha,
            },
            "queue": {"path": str(QUEUE.relative_to(ROOT)),
                      "sha256": EXPECTED[QUEUE]},
            "runtime": {"path": str(RUNTIME.relative_to(ROOT)),
                        "sha256": EXPECTED[RUNTIME]},
            "shards": shards,
        },
        "contract": {
            "prompt_path": str(PROMPT.relative_to(ROOT)),
            "prompt_sha256": EXPECTED[PROMPT],
            "schema_path": str(SCHEMA.relative_to(ROOT)),
            "schema_sha256": EXPECTED[SCHEMA],
        },
        "rendering": {
            "positions": ["A", "Q", "D"],
            "requested_position_spacing_m": view_base.POSITION_DISTANCE_M,
            "headings_per_position": view_base.PANORAMA_HEADINGS,
            "heading_step_deg": view_base.PANORAMA_STEP_DEG,
            "v00_is_local_reference_route_forward": True,
            "rgb_pixels": [view_base.FRAME_SIZE, view_base.FRAME_SIZE],
            "hfov_deg": 63.0,
            "sensor_height_m": 0.88,
            "individual_views_are_ephemeral": True,
            "persisted_contact_sheets_per_event": 3,
        },
        "selected_primary_count": len(primaries),
        "event_count": len(events),
        "failure_count": len(failures),
        "events": sorted(events, key=lambda row: row["expansion_order"]),
        "failures": sorted(failures, key=lambda row: row["expansion_order"]),
        "media_manifest": sorted(media, key=lambda row: row["path"]),
        "media_file_count": len(media),
        "media_total_bytes": sum(row["bytes"] for row in media),
        "network_calls_made": 0,
        "branch_labels_created": 0,
        "human_labels_created": 0,
        "replacement_samples_created": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"], "selected": len(primaries),
        "events": len(events), "failures": len(failures),
        "media_total_bytes": output["media_total_bytes"],
        "output": str(OUT.relative_to(ROOT)), "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate()
    if args.shard_index is None or args.gpu is None:
        raise SystemExit("choose --aggregate or both --shard-index and --gpu")
    return run_shard(args.shard_index, args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
