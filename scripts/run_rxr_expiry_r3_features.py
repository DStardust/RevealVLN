#!/usr/bin/env python3
"""Seal, generate and validate train/development R3 expiry features."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SOURCE = BASE / "RXR_SECONDARY_AUGMENTED_FEATURE_MANIFEST_V1.json"
AUTHORIZATION = BASE / "RXR_SECONDARY_AUGMENTED_TRAINING_AUTHORIZATION_V1.json"
PRIMARY_TX = BASE / "multibranch_v2/RXR_MULTIBRANCH_TX_V2_GATE.json"
SECONDARY_TX = BASE / (
    "secondary_expansion_v1/multibranch/RXR_SECONDARY_TX_GATE.json"
)
REVISION = ROOT / "artifacts/design/MF2_IMPLEMENTATION_CORRECTNESS_REVISION_R3.md"
OUT = BASE / "expiry_r3"
PROTOCOL = OUT / "RXR_EXPIRY_R3_FEATURE_PROTOCOL.json"
FEATURES = OUT / "features"
MANIFEST = OUT / "RXR_EXPIRY_R3_FEATURE_MANIFEST.json"
GATE = OUT / "RXR_EXPIRY_R3_FEATURE_GATE.json"
WORKER = ROOT / "scripts/rxr_expiry_r3_feature_lane.py"


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


def build_protocol() -> dict:
    source = json.loads(SOURCE.read_text())
    authorization = json.loads(AUTHORIZATION.read_text())
    primary_tx = json.loads(PRIMARY_TX.read_text())
    secondary_tx = json.loads(SECONDARY_TX.read_text())
    records = [
        {
            "event_id": row["event_id"],
            "scene_id": row["scene_id"],
            "split": row["split"],
            "label_source": row["label_source"],
            "source_feature_sha256": row["sha256"],
        }
        for row in source["records"] if row["split"] in ("train", "development")
    ]
    counts = Counter(row["split"] for row in records)
    if not (
        authorization.get("status") == "AUGMENTED_DEVELOPMENT_EXPERIMENT_AUTHORIZED"
        and authorization["manifest"]["sha256"] == sha256_file(SOURCE)
        and primary_tx.get("status") == "MULTIBRANCH_TX_PASS"
        and secondary_tx.get("status") == "MULTIBRANCH_TX_PASS"
        and counts == {"train": 424, "development": 68}
        and len(records) == len({row["event_id"] for row in records})
    ):
        raise RuntimeError("R3 feature protocol preconditions failed")
    return {
        "schema_version": "revealnav-mf2-expiry-feature-protocol/3",
        "status": "SEALED_BEFORE_EXPIRY_FEATURE_GENERATION",
        "records": records,
        "counts": dict(counts),
        "sources": {
            str(SOURCE.relative_to(ROOT)): sha256_file(SOURCE),
            str(AUTHORIZATION.relative_to(ROOT)): sha256_file(AUTHORIZATION),
            str(PRIMARY_TX.relative_to(ROOT)): sha256_file(PRIMARY_TX),
            str(SECONDARY_TX.relative_to(ROOT)): sha256_file(SECONDARY_TX),
            str(REVISION.relative_to(ROOT)): sha256_file(REVISION),
        },
        "horizon": "Q through observed T_X; right-censored through trace end",
        "post_D_candidate_rule": (
            "freeze the complete persistent branch tokens from D while each "
            "new egocentric history embedding remains strictly causal"
        ),
        "expiry_label": (
            "1 at observed last-safe prefix, 0 while at risk, -1 after event; "
            "right-censored trajectories contain only 0"
        ),
        "gold_split_included": False,
        "gold_payload_read": False,
        "future_frames_used_for_online_input": 0,
        "training_authorized": False,
        "paper_result": False,
    }


def seal() -> int:
    value = build_protocol()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != value:
            raise RuntimeError("sealed R3 feature protocol drift")
    else:
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "counts": value["counts"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def run_lane(gpu: int, event_ids: list[str]) -> tuple[int, int, Path]:
    event_list = OUT / f"feature_lane_gpu{gpu}_events.json"
    lane_result = OUT / f"feature_lane_gpu{gpu}.json"
    atomic_json(event_list, event_ids)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    completed = subprocess.run([
        sys.executable, str(WORKER),
        "--event-list", str(event_list),
        "--output-dir", str(FEATURES),
        "--lane-result", str(lane_result),
        "--physical-gpu", str(gpu),
    ], cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    lane_result.with_suffix(".stdout").write_text(completed.stdout)
    lane_result.with_suffix(".stderr").write_text(completed.stderr)
    return gpu, completed.returncode, lane_result


def aggregate(gpus: list[int]) -> int:
    protocol = json.loads(PROTOCOL.read_text())
    expected = {row["event_id"]: row for row in protocol["records"]}
    extracted = {}
    for gpu in gpus:
        path = OUT / f"feature_lane_gpu{gpu}.json"
        value = json.loads(path.read_text())
        if (
            value.get("network_attempts") != 0
            or value.get("future_frames_used_for_online_input") != 0
            or value.get("gold_payload_read") is not False
        ):
            raise RuntimeError("R3 lane boundary failure")
        for row in value["records"]:
            if row["event_id"] in extracted:
                raise RuntimeError("duplicate R3 event")
            extracted[row["event_id"]] = row
    if set(extracted) != set(expected):
        raise RuntimeError("R3 feature population closure failure")
    records = []
    for event_id in expected:
        row = extracted[event_id]
        path = ROOT / row["path"]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise RuntimeError("R3 feature shard provenance failure")
        records.append({
            **row,
            "path": os.path.relpath(path, OUT),
        })
    counts = Counter(row["split"] for row in records)
    observed = sum(row["expiry_observed"] for row in records)
    manifest = {
        "schema_version": "revealnav-mf2-expiry-feature-manifest/3",
        "records": records,
        "metadata": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "source_manifest_sha256": sha256_file(SOURCE),
            "normalized_budgets": [1.5, 2.0, 3.0, 4.0],
            "future_frames_used_for_online_input": 0,
            "gold_payload_read": False,
            "training_authorized": True,
            "paper_result": False,
        },
    }
    atomic_json(MANIFEST, manifest)
    sys.path.insert(0, str(ROOT))
    from revealnav_mf2r3 import RevealExpiryFeatureDataset
    train = RevealExpiryFeatureDataset(MANIFEST, "train")
    development = RevealExpiryFeatureDataset(MANIFEST, "development")
    gates = {
        "all_protocol_events_extracted": len(records) == len(expected),
        "train_loads": len(train) == 424,
        "development_loads": len(development) == 68,
        "observed_and_censored_events_present": 0 < observed < len(records),
        "no_gold_records": not any(row["split"] == "gold" for row in records),
        "no_part_files": not list(OUT.rglob("*.part")),
    }
    value = {
        "schema_version": "revealnav-mf2-expiry-feature-gate/3",
        "status": "EXPIRY_R3_FEATURE_GATE_PASS" if all(gates.values()) else
                  "EXPIRY_R3_FEATURE_GATE_FAIL",
        "counts": {
            **dict(counts),
            "events": len(records),
            "expiry_observed": observed,
            "right_censored": len(records) - observed,
            "prefixes": sum(row["steps"] for row in records),
        },
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(MANIFEST),
        },
        "gates": gates,
        "gold_payload_read": False,
        "training_authorized": all(gates.values()),
        "paper_result": False,
    }
    atomic_json(GATE, value)
    print(json.dumps({
        "status": value["status"],
        "counts": value["counts"],
        "gates": gates,
        "manifest": str(MANIFEST.relative_to(ROOT)),
    }, indent=2))
    return 0 if all(gates.values()) else 1


def generate(gpus: list[int]) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != build_protocol():
        raise RuntimeError("R3 feature protocol must be sealed without drift")
    records = json.loads(PROTOCOL.read_text())["records"]
    lanes = {gpu: [] for gpu in gpus}
    for index, row in enumerate(records):
        lanes[gpus[index % len(gpus)]].append(row["event_id"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = list(pool.map(lambda item: run_lane(*item), lanes.items()))
    failures = [(gpu, code) for gpu, code, _ in results if code]
    if failures:
        raise RuntimeError(f"R3 feature lane failures: {failures}")
    return aggregate(gpus)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--generate", action="store_true")
    group.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("invalid R3 GPU list")
    if args.seal:
        return seal()
    if args.aggregate_only:
        return aggregate(gpus)
    return generate(gpus)


if __name__ == "__main__":
    raise SystemExit(main())
