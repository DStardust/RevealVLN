#!/usr/bin/env python3
"""Render and aggregate train-only conditional secondary multiview events."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path


# Keep the canonical project alias here because the frozen upstream renderer
# serializes paths relative to that exact root.  Resolving only this wrapper's
# root would mix the migrated physical path with the canonical alias and make
# otherwise valid project-local media fail ``Path.relative_to``.
ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_rxr_primary_multiview_factory as primary  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SECONDARY = BASE / "secondary_expansion_v1"
SELECTION = SECONDARY / "RXR_SECONDARY_EXPANSION_SELECTION.json"
HINDSIGHT = BASE / "hindsight_factory/RXR_HINDSIGHT_EVENT_CANDIDATES.json"
QUEUE = BASE / "RXR_TRAIN_UNBIASED_EXPANSION_QUEUE.json"
RUNTIME = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
MEDIA_DIR = SECONDARY / "multiview_factory/panoramas"
TMP_DIR = SECONDARY / "multiview_factory/tmp"
RUN_DIR = SECONDARY / "multiview_factory/runs"
OUT = SECONDARY / "multiview_factory/RXR_SECONDARY_MULTIVIEW_INPUTS.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def documents():
    selection = json.loads(SELECTION.read_text())
    hindsight = json.loads(HINDSIGHT.read_text())
    if not (
        selection.get("status") == "FROZEN_TRAIN_ONLY_SECONDARY_SELECTION"
        and selection.get("counts", {}).get("selected_events") == 903
        and hindsight.get("status") == "PASS_PENDING_MULTIVIEW_AND_3D_GATES"
    ):
        raise RuntimeError("secondary multiview precondition failed")
    candidates = {
        row["hindsight_candidate_id"]: row for row in hindsight["candidates"]
    }
    return selection, candidates


def run_shard(shard_index: int, shard_count: int, gpu: int) -> int:
    selection, candidates = documents()
    selected = [
        row for row in selection["items"]
        if row["secondary_order"] % shard_count == shard_index
    ]
    queue = {
        row["expansion_order"]: row
        for row in json.loads(QUEUE.read_text())["candidates"]
    }
    wanted = {row["episode_id"] for row in selected}
    with gzip.open(RUNTIME, "rt", encoding="utf-8") as stream:
        episodes = {
            str(row["episode_id"]): row
            for row in json.load(stream)["episodes"]
            if str(row["episode_id"]) in wanted
        }
    if set(episodes) != wanted:
        raise RuntimeError("secondary runtime episode closure failure")
    primary.MEDIA_DIR = MEDIA_DIR
    primary.TMP_DIR = TMP_DIR
    primary.view_base.GPU_DEVICE = gpu
    by_scene = {}
    for row in selected:
        by_scene.setdefault(row["scene_id"], []).append(row)
    events, media, failures = [], [], []
    completed = 0
    for scene in sorted(by_scene):
        try:
            simulator = primary.view_base.build_sim(scene)
        except Exception as error:
            for row in by_scene[scene]:
                failures.append({
                    "event_id": row["event_id"],
                    "secondary_order": row["secondary_order"],
                    "failure_stage": "SIMULATOR_CONSTRUCTION",
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                })
            continue
        try:
            for row in sorted(
                by_scene[scene], key=lambda value: value["secondary_order"]
            ):
                candidate = candidates[row["event_id"]]
                try:
                    event, records = primary.build_event(
                        candidate,
                        queue[row["expansion_order"]],
                        episodes[row["episode_id"]],
                        simulator,
                        gpu,
                    )
                    event.update({
                        "secondary_order": row["secondary_order"],
                        "cascade_role": "CONDITIONAL_SECONDARY",
                        "conditional_primary_event_id": row[
                            "conditional_primary_event_id"
                        ],
                        "scene_split": "train",
                    })
                    events.append(event)
                    media.extend(records)
                    completed += 1
                except Exception as error:
                    failures.append({
                        "event_id": row["event_id"],
                        "secondary_order": row["secondary_order"],
                        "failure_stage": "EVENT_RENDER",
                        "error_type": type(error).__name__,
                        "error": str(error)[:2000],
                    })
                print(
                    f"[{completed + len(failures)}/{len(selected)}] "
                    f"{row['event_id']}", flush=True
                )
        finally:
            simulator.close()
    result = {
        "schema_version": "revealnav-rxr-secondary-multiview-shard/1",
        "status": "PASS" if not failures else "PASS_WITH_FAIL_CLOSED_FAILURES",
        "selection_sha256": sha256_file(SELECTION),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "gpu": gpu,
        "selected_count": len(selected),
        "completed_count": len(events),
        "failure_count": len(failures),
        "events": events,
        "media_manifest": media,
        "failures": failures,
        "gold_payload_read": False,
        "training_authorized": False,
    }
    path = RUN_DIR / f"shard_{shard_index:02d}.json"
    atomic_json(path, result)
    print(json.dumps({
        "status": result["status"],
        "shard": shard_index,
        "completed": len(events),
        "failures": len(failures),
        "output": str(path.relative_to(ROOT)),
    }, indent=2))
    return 0


def aggregate(shard_count: int) -> int:
    selection, _ = documents()
    expected = {row["event_id"] for row in selection["items"]}
    events, media, failures, sources = [], [], [], []
    observed = set()
    for shard_index in range(shard_count):
        path = RUN_DIR / f"shard_{shard_index:02d}.json"
        value = json.loads(path.read_text())
        if not (
            value.get("status") in {"PASS", "PASS_WITH_FAIL_CLOSED_FAILURES"}
            and value.get("selection_sha256") == sha256_file(SELECTION)
            and value.get("shard_index") == shard_index
            and value.get("shard_count") == shard_count
        ):
            raise RuntimeError(f"secondary shard contract failure: {shard_index}")
        ids = {row["event_id"] for row in value["events"]}
        ids.update(row["event_id"] for row in value["failures"])
        if observed & ids:
            raise RuntimeError("duplicate secondary shard event")
        observed.update(ids)
        events.extend(value["events"])
        media.extend(value["media_manifest"])
        failures.extend(value["failures"])
        sources.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        })
    if observed != expected:
        raise RuntimeError("secondary aggregate event closure failure")
    events.sort(key=lambda row: row["secondary_order"])
    failures.sort(key=lambda row: row["secondary_order"])
    result = {
        "manifest": "RevealNav train-only conditional secondary multiview inputs",
        "revision": "rxr-secondary-multiview-inputs/1",
        "status": "READY_FOR_BRANCH_PROPOSER",
        "sources": {
            "selection": {
                "path": str(SELECTION.relative_to(ROOT)),
                "sha256": sha256_file(SELECTION),
            },
            "hindsight": {
                "path": str(HINDSIGHT.relative_to(ROOT)),
                "sha256": sha256_file(HINDSIGHT),
            },
            "shards": sources,
        },
        "contract": {
            "cascade_role": "CONDITIONAL_SECONDARY",
            "scene_split": "train",
            "full_instruction_retained": True,
            "panorama_offline_annotation_only": True,
        },
        "rendering": {
            "shard_count": shard_count,
            "a_q_d_panoramas": True,
            "chronological_context_reused_from_sealed_hindsight_request": True,
        },
        "selected_secondary_count": len(selection["items"]),
        "event_count": len(events),
        "failure_count": len(failures),
        "events": events,
        "failures": failures,
        "media_manifest": media,
        "media_file_count": len(media),
        "media_total_bytes": sum(row["bytes"] for row in media),
        "network_calls_made": 0,
        "branch_labels_created": 0,
        "human_labels_created": 0,
        "replacement_samples_created": 0,
        "gold_payload_read": False,
        "training_authorized": False,
    }
    atomic_json(OUT, result)
    print(json.dumps({
        "status": result["status"],
        "event_count": len(events),
        "failure_count": len(failures),
        "media_file_count": len(media),
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate(args.shard_count)
    if (
        args.shard_index is None
        or args.gpu is None
        or not 0 <= args.shard_index < args.shard_count
    ):
        parser.error("valid --shard-index and --gpu are required")
    return run_shard(args.shard_index, args.shard_count, args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
