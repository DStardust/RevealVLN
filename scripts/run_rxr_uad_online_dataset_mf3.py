#!/usr/bin/env python3
"""Collect a scene-disjoint exact-online RxR-train dataset for MF3 UAD."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
WORKER = ROOT / "scripts/rxr_uad_shadow_worker_mf3.py"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/"
    "train/train_guide.json.gz"
)
SEED = 20260828


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def selection(
    fit_per_scene: int,
    other_per_scene: int,
    episode_rank_split: tuple[int, ...] | None = None,
) -> list[dict]:
    with gzip.open(DATASET, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    by_scene = defaultdict(list)
    for row in episodes:
        if row["instruction"].get("language") not in ("en-US", "en-IN"):
            continue
        scene = Path(row["scene_id"]).stem
        by_scene[scene].append(str(row["episode_id"]))
    scenes = sorted(by_scene, key=lambda value: hashlib.sha256(
        f"{SEED}:scene:{value}".encode()
    ).hexdigest())
    rows = []
    if episode_rank_split is not None:
        if len(episode_rank_split) == 3:
            split_counts = tuple(zip(
                ("fit", "calibration", "shadow"), episode_rank_split
            ))
        elif len(episode_rank_split) == 4:
            split_counts = tuple(zip(
                ("fit", "calibration", "diagnostic", "shadow"),
                episode_rank_split,
            ))
        else:
            raise ValueError("episode rank split must have three or four counts")
        for scene in scenes:
            candidates = sorted(by_scene[scene], key=lambda value: hashlib.sha256(
                f"{SEED}:episode:{value}".encode()
            ).hexdigest())
            start = 0
            for split, count in split_counts:
                end = start + count
                rows.extend({
                    "episode_id": episode_id, "scene_id": scene, "split": split,
                } for episode_id in candidates[start:end])
                start = end
        return rows

    fit_end = round(len(scenes) * 0.65)
    calibration_end = fit_end + round(len(scenes) * 0.15)
    partitions = {
        **{scene: "fit" for scene in scenes[:fit_end]},
        **{scene: "calibration" for scene in scenes[fit_end:calibration_end]},
        **{scene: "shadow" for scene in scenes[calibration_end:]},
    }
    for scene in scenes:
        count = fit_per_scene if partitions[scene] == "fit" else other_per_scene
        candidates = sorted(by_scene[scene], key=lambda value: hashlib.sha256(
            f"{SEED}:episode:{value}".encode()
        ).hexdigest())
        for episode_id in candidates[:count]:
            rows.append({
                "episode_id": episode_id,
                "scene_id": scene,
                "split": partitions[scene],
            })
    return rows


def selection_rank_range(start: int, end: int) -> list[dict]:
    """Select a disclosed-free episode-rank interval within each scene."""
    with gzip.open(DATASET, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    by_scene = defaultdict(list)
    for row in episodes:
        if row["instruction"].get("language") in ("en-US", "en-IN"):
            by_scene[Path(row["scene_id"]).stem].append(str(row["episode_id"]))
    scenes = sorted(by_scene, key=lambda value: hashlib.sha256(
        f"{SEED}:scene:{value}".encode()
    ).hexdigest())
    rows = []
    for scene in scenes:
        candidates = sorted(by_scene[scene], key=lambda value: hashlib.sha256(
            f"{SEED}:episode:{value}".encode()
        ).hexdigest())
        for episode_id in candidates[start:end]:
            rows.append({
                "episode_id": episode_id, "scene_id": scene, "split": "shadow",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    parser.add_argument("--fit-per-scene", type=int, default=3)
    parser.add_argument("--other-per-scene", type=int, default=2)
    parser.add_argument("--episode-rank-split")
    parser.add_argument("--episode-rank-range")
    parser.add_argument("--reuse-manifest", type=Path)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--worker-slots")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--contextual-gmap-features", action="store_true")
    parser.add_argument("--policy-fusion-features", action="store_true")
    args = parser.parse_args()
    if args.contextual_gmap_features and args.policy_fusion_features:
        raise SystemExit("select at most one contextual feature source")
    output = args.output_dir.resolve()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    if (
        ROOT not in output.parents or not gpus or len(gpus) != len(set(gpus))
        or args.fit_per_scene < 1 or args.other_per_scene < 1
        or args.workers_per_gpu < 1
    ):
        raise SystemExit("invalid project-local output, GPUs, or sampling")
    output.mkdir(parents=True, exist_ok=True)
    rank_split = None
    if args.episode_rank_split:
        rank_split = tuple(int(value) for value in args.episode_rank_split.split(","))
        if len(rank_split) not in (3, 4) or min(rank_split) < 1:
            raise SystemExit("episode rank split must have 3/4 positive counts")
    rank_range = None
    if args.episode_rank_range:
        if rank_split is not None:
            raise SystemExit("choose episode-rank-split or episode-rank-range")
        rank_range = tuple(int(value) for value in args.episode_rank_range.split(","))
        if len(rank_range) != 2 or not (0 <= rank_range[0] < rank_range[1]):
            raise SystemExit("episode rank range must be start,end with 0<=start<end")
        selected = selection_rank_range(*rank_range)
    else:
        selected = selection(args.fit_per_scene, args.other_per_scene, rank_split)
    reuse_path = args.reuse_manifest.resolve() if args.reuse_manifest else None
    if reuse_path is not None and (
        ROOT not in reuse_path.parents or not reuse_path.is_file()
        or reuse_path.is_symlink()
    ):
        raise SystemExit("reuse manifest must be a project-local regular file")
    protocol_path = output / "MF3B_ONLINE_DATA_PROTOCOL.json"
    protocol = {
        "schema_version": "revealnav-mf3b-online-protocol/1",
        "status": "SEALED_BEFORE_COLLECTION",
        "seed": SEED,
        "dataset": "RxR train guide en-US/en-IN only",
        "observation_frontend": (
            "frozen_etp_r1_policy_fusion_token"
            if args.policy_fusion_features
            else "frozen_etp_r1_contextual_gmap_token"
            if args.contextual_gmap_features
            else "frozen_etp_r1_12_view_graphmap"
        ),
        "teacher_role": "label_only_after_native_policy_output",
        "labels": {
            "target": "native RxR nDTW teacher current ghost",
            "target_in_set": "teacher ghost is current-local and unvisited",
            "separation": (
                "teacher is deterministic minimum and geodesic margin to "
                "runner-up is at least 0.5m"
            ),
            "evidence_reveal_expiry": "masked; never invented by geometry",
        },
        "scene_partition": (
            "within-scene deterministic episode-rank range shadow"
            if rank_range is not None else
            "65% fit / 15% calibration / 20% shadow"
            if rank_split is None
            else "within-scene deterministic episode-rank fit/calibration/shadow"
        ),
        "records": selected,
        "public_unseen_authorized": False,
        "uses_future_teacher_as_online_input": False,
    }
    if rank_split is None:
        protocol["fit_per_scene"] = args.fit_per_scene
        protocol["other_per_scene"] = args.other_per_scene
    if rank_split is not None:
        protocol["episode_rank_split"] = list(rank_split)
    if rank_range is not None:
        protocol["episode_rank_range"] = list(rank_range)
    if reuse_path is not None:
        protocol["reuse_manifest"] = {
            "path": str(reuse_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(reuse_path.read_bytes()).hexdigest(),
        }
    if args.workers_per_gpu != 1:
        protocol["workers_per_gpu"] = args.workers_per_gpu
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("sealed MF3B online protocol drift")
    else:
        atomic_json(protocol_path, protocol)

    pending = deque()
    complete = []
    failures = []
    reused = {}
    if reuse_path is not None:
        source = json.loads(reuse_path.read_text())
        if source.get("status") != "PASS":
            raise RuntimeError("reuse manifest is not complete")
        reused = {str(row["episode_id"]): row for row in source["records"]}

    def archive_interrupted(run_dir: Path, episode_id: str) -> None:
        parent = output / "interrupted"
        parent.mkdir(parents=True, exist_ok=True)
        suffix = 0
        while True:
            name = f"ep_{episode_id}" if suffix == 0 else f"ep_{episode_id}_{suffix}"
            destination = parent / name
            if not destination.exists():
                os.replace(run_dir, destination)
                return
            suffix += 1

    for row in selected:
        source = reused.get(row["episode_id"])
        if source is not None:
            if source["scene_id"] != row["scene_id"]:
                raise RuntimeError("reused episode scene drift")
            complete.append(row)
            continue
        run_dir = output / "runs" / f"ep_{row['episode_id']}"
        summary_path = run_dir / "RUN_SUMMARY.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            if summary.get("status") == "SHADOW_PASS" and summary.get(
                "online_feature"
            ):
                complete.append(row)
                continue
            if args.retry_failures or str(summary.get("error", "")).startswith((
                "KeyboardInterrupt", "EOFError",
            )):
                archive_interrupted(run_dir, row["episode_id"])
                pending.append(row)
                continue
            failures.append({**row, "reason": "existing_run_not_pass"})
            continue
        if run_dir.exists():
            archive_interrupted(run_dir, row["episode_id"])
        pending.append(row)
    reused_count = len(complete)
    collection_started = time.time()
    active = {}
    attempts = defaultdict(int)
    if args.worker_slots:
        slot_gpus = tuple(int(value) for value in args.worker_slots.split(","))
        if not slot_gpus or any(gpu not in gpus for gpu in slot_gpus):
            raise SystemExit("worker slots must reference an authorized GPU")
        counters = defaultdict(int)
        slots = []
        for gpu in slot_gpus:
            slots.append((gpu, counters[gpu]))
            counters[gpu] += 1
        slots = tuple(slots)
    else:
        slots = tuple(
            (gpu, worker)
            for gpu in gpus for worker in range(args.workers_per_gpu)
        )
    progress_path = output / "MF3B_ONLINE_DATA_PROGRESS.json"

    def write_progress(status: str = "RUNNING") -> None:
        elapsed = time.time() - collection_started
        new_complete = len(complete) - reused_count
        new_total = len(selected) - reused_count
        rate = new_complete / elapsed if elapsed > 0 else 0.0
        atomic_json(progress_path, {
            "schema_version": "revealnav-mf3b-online-progress/1",
            "status": status,
            "total": len(selected),
            "completed": len(complete),
            "failed": len(failures),
            "remaining": len(pending),
            "reused": reused_count,
            "new_completed": new_complete,
            "new_total": new_total,
            "elapsed_s": round(elapsed, 1),
            "eta_s": (
                round((new_total - new_complete) / rate, 1)
                if rate > 0 else None
            ),
            "active": {
                f"{slot[0]}:{slot[1]}": {
                    "episode_id": value["row"]["episode_id"],
                    "scene_id": value["row"]["scene_id"],
                    "pid": value["process"].pid,
                }
                for slot, value in active.items()
            },
            "retry_attempts": dict(sorted(attempts.items())),
            "updated_at_unix": time.time(),
        })

    while pending or active:
        for slot in slots:
            if slot in active or not pending:
                continue
            gpu, _ = slot
            row = pending.popleft()
            run_dir = output / "runs" / f"ep_{row['episode_id']}"
            stdout_path = output / "logs" / f"ep_{row['episode_id']}.stdout"
            stderr_path = output / "logs" / f"ep_{row['episode_id']}.stderr"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout = stdout_path.open("w")
            stderr = stderr_path.open("w")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen([
                str(PYTHON), str(WORKER),
                "--episode-id", row["episode_id"],
                "--run-dir", str(run_dir), "--collect-feature",
                *(["--contextual-gmap-features"]
                  if args.contextual_gmap_features else []),
                *(["--policy-fusion-features"]
                  if args.policy_fusion_features else []),
            ], cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
            attempts[row["episode_id"]] += 1
            active[slot] = {
                "row": row,
                "process": process,
                "stdout": stdout,
                "stderr": stderr,
            }
        write_progress()
        time.sleep(1)
        for slot, value in list(active.items()):
            code = value["process"].poll()
            if code is None:
                continue
            value["stdout"].close()
            value["stderr"].close()
            if code == 0:
                complete.append(value["row"])
            elif args.retry_failures and attempts[
                value["row"]["episode_id"]
            ] < 3:
                archive_interrupted(
                    output / "runs" / f"ep_{value['row']['episode_id']}",
                    value["row"]["episode_id"],
                )
                pending.append(value["row"])
            else:
                failures.append({
                    **value["row"], "reason": f"worker_exit_{code}"
                })
            del active[slot]

    records = []
    for row in complete:
        source = reused.get(row["episode_id"])
        if source is not None:
            feature = {
                key: value for key, value in source.items()
                if key not in ("episode_id", "scene_id", "split")
            }
        else:
            summary = json.loads((
                output / "runs" / f"ep_{row['episode_id']}" / "RUN_SUMMARY.json"
            ).read_text())
            feature = summary["online_feature"]
        records.append({**row, **feature})
    records.sort(key=lambda row: int(row["episode_id"]))
    manifest = {
        "schema_version": "revealnav-mf3b-online-manifest/1",
        "status": "PASS" if not failures and len(records) == len(selected) else "FAIL",
        "records": records,
        "failures": failures,
        "counts": {
            split: sum(row["split"] == split for row in records)
            for split in sorted({row["split"] for row in selected})
        },
        "public_unseen_authorized": False,
    }
    atomic_json(output / "MF3B_ONLINE_DATA_MANIFEST.json", manifest)
    write_progress("COMPLETE" if manifest["status"] == "PASS" else "FAIL")
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
