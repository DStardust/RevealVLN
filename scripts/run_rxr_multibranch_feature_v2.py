#!/usr/bin/env python3
"""Extract and validate the frozen-feature manifest for MF2-CR6."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
V2 = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
INDEX = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
TX = V2 / "RXR_MULTIBRANCH_TX_V2_GATE.json"
WORKER = ROOT / "scripts/rxr_multibranch_feature_v2_lane.py"
FEATURES = V2 / "frozen_features"
OUT = V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2.json"
GATE = V2 / "RXR_MULTIBRANCH_FEATURE_GATE_V2.json"
REQUIRED_SPLITS = ("train", "development", "gold")
HUMAN_AUDIT_STATUS = "PENDING_FRESH_FULLSET_AUDIT"
REMAINING_BLOCKER = "fresh independent full-set human audit"
FEATURE_GATE_PASS_STATUS = "FEATURE_GATE_PASS_AUDIT_REQUIRED"
TRAINING_AUTHORIZED_AFTER_FEATURE_GATE = False
RECORD_EXTRA = {}
METADATA_EXTRA = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def run_lane(name, gpu, events):
    lane = V2 / ("feature_lane_%s.json" % name)
    event_list = V2 / ("feature_lane_%s_events.json" % name)
    atomic_json(event_list, events)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    result = subprocess.run([
        sys.executable, str(WORKER), "--event-list", str(event_list),
        "--output-dir", str(FEATURES), "--lane-result", str(lane),
        "--physical-gpu", str(gpu),
    ], cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    lane.with_suffix(".stdout").write_text(result.stdout)
    lane.with_suffix(".stderr").write_text(result.stderr)
    return name, gpu, result.returncode, lane


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--gpu-slots")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("invalid feature GPU list")
    gpu_slots = (
        [int(value) for value in args.gpu_slots.split(",")]
        if args.gpu_slots else gpus
    )
    if not gpu_slots or not set(gpu_slots) <= set(gpus):
        raise SystemExit("invalid feature GPU slot list")
    index = json.loads(INDEX.read_text())
    tx = json.loads(TX.read_text())
    if (index.get("feature_generation_authorized") is not True
            or tx.get("status") != "MULTIBRANCH_TX_PASS"):
        raise RuntimeError("feature generation prerequisites failed")
    events = [row["event_id"] for row in index["records"]]
    names = (
        ["slot%02d_gpu%d" % (index, gpu)
         for index, gpu in enumerate(gpu_slots)]
        if args.gpu_slots else ["gpu%d" % gpu for gpu in gpu_slots]
    )
    lanes = [[] for _ in gpu_slots]
    for offset, event_id in enumerate(events):
        lanes[offset % len(gpu_slots)].append(event_id)
    if args.aggregate_only:
        results = [
            (name, gpu, 0, V2 / ("feature_lane_%s.json" % name))
            for name, gpu in zip(names, gpu_slots)
        ]
        if any(not path.is_file() or path.is_symlink()
               for _, _, _, path in results):
            raise RuntimeError("aggregate-only lane evidence is incomplete")
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(gpu_slots)) as pool:
            results = [
                future.result()
                for future in [
                    pool.submit(run_lane, name, gpu, events)
                    for name, gpu, events in zip(names, gpu_slots, lanes)
                ]
            ]
    failed = [(name, gpu, code)
              for name, gpu, code, _ in results if code]
    if failed:
        raise SystemExit("feature lane failures: " + repr(failed))
    extracted = {}
    for _, _, _, path in results:
        value = json.loads(path.read_text())
        if value["network_attempts"] != 0 or value["future_frames_used"] != 0:
            raise RuntimeError("feature lane boundary failure")
        for row in value["records"]:
            if row["event_id"] in extracted:
                raise RuntimeError("duplicate extracted event")
            extracted[row["event_id"]] = row
    if set(extracted) != set(events):
        raise RuntimeError("feature population closure failure")
    records = []
    for source in index["records"]:
        feature = extracted[source["event_id"]]
        path = ROOT / feature["path"]
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != feature["bytes"]
                or sha256_file(path) != feature["sha256"]):
            raise RuntimeError("feature shard provenance failure")
        records.append({
            "event_id": source["event_id"],
            "scene_id": source["scene_id"],
            "split": source["split"],
            "path": os.path.relpath(path, V2),
            "bytes": feature["bytes"],
            "sha256": feature["sha256"],
            "steps": feature["steps"],
            "candidate_count": feature["candidate_count"],
            "feature_dim": feature["feature_dim"],
            **RECORD_EXTRA,
        })
    manifest = {
        "schema_version": "revealnav-mf2-feature-manifest/1",
        "records": records,
        "metadata": {
            "synthetic": False,
            "training_authorized": False,
            "causal_prefix_verified": True,
            "future_frames_used": 0,
            "full_candidate_sets": True,
            "normalized_budgets": [1.5, 2.0, 3.0, 4.0],
            "feature_contract": (
                "frozen ETP-R1 XLM-R instruction mean, causal 63-degree "
                "panorama mean, and persistent branch token embeddings"
            ),
            "training_index_sha256": sha256_file(INDEX),
            "multibranch_tx_sha256": sha256_file(TX),
            "human_audit_status": HUMAN_AUDIT_STATUS,
            "paper_result": False,
            **METADATA_EXTRA,
        },
    }
    atomic_json(OUT, manifest)
    sys.path.insert(0, str(ROOT))
    from revealnav_mf2 import RevealFeatureDataset
    split_counts = {
        split: sum(row["split"] == split for row in records)
        for split in ("train", "development", "gold")
    }
    loaded_splits = {
        split: RevealFeatureDataset(OUT, split)
        for split in REQUIRED_SPLITS
    }
    gates = {
        "all_events_extracted": len(records) == len(events),
        "train_manifest_loads": (
            "train" in loaded_splits
            and len(loaded_splits["train"]) == split_counts["train"] > 0
        ),
        "development_manifest_loads": (
            len(loaded_splits["development"])
            == split_counts["development"] > 0
            if "development" in REQUIRED_SPLITS
            else split_counts["development"] == 0
        ),
        "gold_manifest_loads": (
            len(loaded_splits["gold"]) == split_counts["gold"] > 0
            if "gold" in REQUIRED_SPLITS else split_counts["gold"] == 0
        ),
        "all_features_768d": all(row["feature_dim"] == 768 for row in records),
        "candidate_counts_two_to_four": all(
            2 <= row["candidate_count"] <= 4 for row in records
        ),
        "no_future_frames_used": True,
    }
    training_authorized = (
        TRAINING_AUTHORIZED_AFTER_FEATURE_GATE and all(gates.values())
    )
    if training_authorized:
        manifest["metadata"]["training_authorized"] = True
        atomic_json(OUT, manifest)
    value = {
        "schema_version": "revealnav-mf2-feature-gate/2",
        "status": FEATURE_GATE_PASS_STATUS if all(gates.values())
                  else "FEATURE_GATE_FAIL",
        "manifest": {"path": str(OUT.relative_to(ROOT)),
                     "sha256": sha256_file(OUT)},
        "counts": {
            "events": len(records),
            "train": split_counts["train"],
            "development": split_counts["development"],
            "gold": split_counts["gold"],
            "three_or_four_branch": sum(row["candidate_count"] >= 3
                                        for row in records),
        },
        "gates": gates,
        "training_authorized": training_authorized,
        "remaining_blocker": (
            None if training_authorized else REMAINING_BLOCKER
        ),
    }
    atomic_json(GATE, value)
    print(json.dumps({"status": value["status"], "counts": value["counts"],
                      "gates": gates, "output": str(GATE.relative_to(ROOT))},
                     indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
