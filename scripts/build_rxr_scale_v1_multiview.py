#!/usr/bin/env python3
"""Render and aggregate scale-v1 A/Q/D panorama inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_rxr_primary_multiview_factory as primary  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SCALE = BASE / "scale_v1"
SELECTION = SCALE / "RXR_SCALE_V1_SELECTION.json"
HINDSIGHT = BASE / "hindsight_factory/RXR_HINDSIGHT_EVENT_CANDIDATES.json"
QUEUE = BASE / "RXR_TRAIN_UNBIASED_EXPANSION_QUEUE.json"
RUNTIME = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)


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


def paths(lane: str) -> tuple[Path, Path, Path, Path]:
    root = SCALE / lane / "multiview"
    return root, root / "panoramas", root / "tmp", root / "runs"


def documents(lane: str) -> tuple[dict, dict]:
    selection = json.loads(SELECTION.read_text())
    hindsight = json.loads(HINDSIGHT.read_text())
    if not (
        selection.get("status") == "SCALE_V1_SELECTION_FROZEN"
        and hindsight.get("status") == "PASS_PENDING_MULTIVIEW_AND_3D_GATES"
        and lane in {"automatic", "new_gold"}
    ):
        raise RuntimeError("scale multiview precondition failed")
    candidates = {
        row["hindsight_candidate_id"]: row for row in hindsight["candidates"]
    }
    return selection, candidates


def run_shard(lane: str, shard_index: int, shard_count: int, gpu: int) -> int:
    selection, candidates = documents(lane)
    selected = [
        row for row in selection[lane]
        if row["scale_order"] % shard_count == shard_index
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
        raise RuntimeError("scale runtime episode closure failure")

    root, media_dir, tmp_dir, run_dir = paths(lane)
    primary.MEDIA_DIR = media_dir
    primary.TMP_DIR = tmp_dir
    primary.view_base.GPU_DEVICE = gpu
    by_scene: dict[str, list[dict]] = {}
    for row in selected:
        by_scene.setdefault(row["scene_id"], []).append(row)
    events: list[dict] = []
    media: list[dict] = []
    failures: list[dict] = []
    for scene in sorted(by_scene):
        try:
            simulator = primary.view_base.build_sim(scene)
        except Exception as error:
            failures.extend({
                "event_id": row["event_id"],
                "scale_order": row["scale_order"],
                "failure_stage": "SIMULATOR_CONSTRUCTION",
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            } for row in by_scene[scene])
            continue
        try:
            for row in sorted(by_scene[scene], key=lambda value: value["scale_order"]):
                try:
                    event, records = primary.build_event(
                        candidates[row["event_id"]],
                        queue[row["expansion_order"]],
                        episodes[row["episode_id"]],
                        simulator,
                        gpu,
                    )
                    event.update({
                        "scale_order": row["scale_order"],
                        "scale_lane": lane,
                        "scene_split": row["scene_split"],
                        "cascade_role": "SCALE_UNCONSUMED_CANDIDATE",
                    })
                    if lane == "new_gold":
                        event["gold_wave"] = row["gold_wave"]
                    events.append(event)
                    media.extend(records)
                except Exception as error:
                    failures.append({
                        "event_id": row["event_id"],
                        "scale_order": row["scale_order"],
                        "failure_stage": "EVENT_RENDER",
                        "error_type": type(error).__name__,
                        "error": str(error)[:2000],
                    })
                print(
                    f"[{len(events) + len(failures)}/{len(selected)}] "
                    f"{lane} {row['event_id']}",
                    flush=True,
                )
        finally:
            simulator.close()

    result = {
        "schema_version": "revealnav-rxr-scale-multiview-shard/1",
        "status": "PASS" if not failures else "PASS_WITH_FAIL_CLOSED_FAILURES",
        "selection_sha256": sha256_file(SELECTION),
        "lane": lane,
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
    path = run_dir / f"shard_{shard_index:02d}.json"
    atomic_json(path, result)
    print(json.dumps({
        "status": result["status"],
        "lane": lane,
        "shard": shard_index,
        "completed": len(events),
        "failures": len(failures),
    }, indent=2))
    return 0


def aggregate(lane: str, shard_count: int) -> int:
    selection, _ = documents(lane)
    expected = {row["event_id"] for row in selection[lane]}
    root, _, _, run_dir = paths(lane)
    events: list[dict] = []
    media: list[dict] = []
    failures: list[dict] = []
    sources: list[dict] = []
    observed: set[str] = set()
    for shard_index in range(shard_count):
        path = run_dir / f"shard_{shard_index:02d}.json"
        value = json.loads(path.read_text())
        if not (
            value.get("status") in {"PASS", "PASS_WITH_FAIL_CLOSED_FAILURES"}
            and value.get("selection_sha256") == sha256_file(SELECTION)
            and value.get("lane") == lane
            and value.get("shard_index") == shard_index
            and value.get("shard_count") == shard_count
        ):
            raise RuntimeError(f"scale shard contract failure: {lane}/{shard_index}")
        ids = {row["event_id"] for row in value["events"]}
        ids.update(row["event_id"] for row in value["failures"])
        if observed & ids:
            raise RuntimeError("duplicate scale shard event")
        observed.update(ids)
        events.extend(value["events"])
        media.extend(value["media_manifest"])
        failures.extend(value["failures"])
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    if observed != expected:
        raise RuntimeError("scale aggregate event closure failure")
    events.sort(key=lambda row: row["scale_order"])
    failures.sort(key=lambda row: row["scale_order"])
    output = {
        "schema_version": "revealnav-rxr-scale-multiview-inputs/1",
        "status": "READY_FOR_BRANCH_PROPOSER",
        "lane": lane,
        "sources": {
            "selection": {"path": str(SELECTION.relative_to(ROOT)), "sha256": sha256_file(SELECTION)},
            "hindsight": {"path": str(HINDSIGHT.relative_to(ROOT)), "sha256": sha256_file(HINDSIGHT)},
            "shards": sources,
        },
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
        "old_gold_payload_read": False,
        "training_authorized": False,
    }
    out = root / "RXR_SCALE_MULTIVIEW_INPUTS.json"
    atomic_json(out, output)
    print(json.dumps({
        "status": output["status"],
        "lane": lane,
        "event_count": len(events),
        "failure_count": len(failures),
        "output": str(out.relative_to(ROOT)),
        "sha256": sha256_file(out),
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("automatic", "new_gold"), required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate(args.lane, args.shard_count)
    if args.shard_index is None or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("valid --shard-index required")
    return run_shard(args.lane, args.shard_index, args.shard_count, args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
