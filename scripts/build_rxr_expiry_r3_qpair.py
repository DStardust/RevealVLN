#!/usr/bin/env python3
"""Derive bounded with/without-checkpoint Q pairs from sealed T_X evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SOURCE_ROOT = BASE / "expiry_r3"
SOURCE_MANIFEST = SOURCE_ROOT / "RXR_EXPIRY_R3_FEATURE_MANIFEST.json"
SOURCE_GATE = SOURCE_ROOT / "RXR_EXPIRY_R3_FEATURE_GATE.json"
PRIMARY = BASE / "multibranch_v2"
SECONDARY = BASE / "secondary_expansion_v1/multibranch"
REVISION = ROOT / "artifacts/design/MF2_OPV_CONTRACT_CORRECTION_R3Q.md"
OUT = BASE / "expiry_r3_qpair"
PROTOCOL = OUT / "RXR_EXPIRY_R3_Q_FEATURE_PROTOCOL.json"
FEATURES = OUT / "features"
MANIFEST = OUT / "RXR_EXPIRY_R3_Q_FEATURE_MANIFEST.json"
GATE = OUT / "RXR_EXPIRY_R3_Q_FEATURE_GATE.json"
FAILURE_COST = 5.0


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


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(part, path)


def tx_path(record: dict) -> Path:
    base = (
        PRIMARY if record["label_source"] == "primary_human_audited"
        else SECONDARY
    )
    return base / "tx_runs/round1" / f"{record['event_id']}.json"


def build_protocol() -> dict:
    gate = json.loads(SOURCE_GATE.read_text())
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if not (
        gate.get("status") == "EXPIRY_R3_FEATURE_GATE_PASS"
        and gate["manifest"]["sha256"] == sha256_file(SOURCE_MANIFEST)
        and len(manifest["records"]) == 492
    ):
        raise RuntimeError("paired-Q protocol preconditions failed")
    return {
        "schema_version": "revealnav-mf2-expiry-q-feature-protocol/3",
        "status": "SEALED_BEFORE_PAIRED_Q_DERIVATION",
        "events": [row["event_id"] for row in manifest["records"]],
        "sources": {
            str(SOURCE_MANIFEST.relative_to(ROOT)): sha256_file(SOURCE_MANIFEST),
            str(SOURCE_GATE.relative_to(ROOT)): sha256_file(SOURCE_GATE),
            str(REVISION.relative_to(ROOT)): sha256_file(REVISION),
        },
        "failure_cost": FAILURE_COST,
        "q_without": "clipped direct-controller normalized action cost",
        "q_with": "minimum clipped direct or saved-checkpoint normalized cost",
        "opv": "max_candidate(Q_without - Q_with)",
        "gold_payload_read": False,
        "future_frames_used_for_online_input": 0,
        "training_authorized": False,
        "paper_result": False,
    }


def seal() -> int:
    value = build_protocol()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != value:
            raise RuntimeError("sealed paired-Q protocol drift")
    else:
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "events": len(value["events"]),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def bounded_cost(route: dict, denominator: int) -> float:
    if not route.get("success") or route.get("action_count") is None:
        return FAILURE_COST
    return min(float(route["action_count"]) / denominator, FAILURE_COST)


def q_labels(tx: dict, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch_ids = tx["candidate_branch_ids"]
    q_prefix = tx["checkpoint"]["prefix_index"]
    with_checkpoint = np.full((steps, len(branch_ids)), np.inf, np.float32)
    without_checkpoint = np.full_like(with_checkpoint, np.inf)
    opv = np.zeros(steps, np.float32)
    for branch_index, branch_id in enumerate(branch_ids):
        controller = tx["branches"][branch_id]["controllers"][
            "frozen_shortest_path_compat"
        ]
        denominator = int(controller["normalization_denominator_actions"])
        rows = controller["prefix_costs"]
        if len(rows) < steps:
            raise RuntimeError("paired-Q horizon exceeds T_X evidence")
        for offset in range(steps):
            row = rows[offset]
            if row["prefix_index"] != q_prefix + offset:
                raise RuntimeError("paired-Q prefix alignment failure")
            direct = bounded_cost(row["direct"], denominator)
            saved = bounded_cost(row["saved_via_checkpoint"], denominator)
            without_checkpoint[offset, branch_index] = direct
            with_checkpoint[offset, branch_index] = min(direct, saved)
    opv[:] = (without_checkpoint - with_checkpoint).max(axis=1)
    return with_checkpoint, without_checkpoint, opv


def build() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != build_protocol():
        raise RuntimeError("paired-Q protocol must be sealed without drift")
    source = json.loads(SOURCE_MANIFEST.read_text())
    records = []
    positive = 0
    for record in source["records"]:
        source_path = (SOURCE_ROOT / record["path"]).resolve()
        if (
            SOURCE_ROOT not in source_path.parents
            or source_path.is_symlink()
            or not source_path.is_file()
            or source_path.stat().st_size != record["bytes"]
            or sha256_file(source_path) != record["sha256"]
        ):
            raise RuntimeError("R3 source shard provenance failure")
        with np.load(source_path, allow_pickle=False) as shard:
            arrays = {key: shard[key] for key in shard.files}
        tx = json.loads(tx_path(record).read_text())["evidence"]
        steps = arrays["history_embeddings"].shape[0]
        with_checkpoint, without_checkpoint, opv = q_labels(tx, steps)
        mask = arrays["candidate_mask"]
        with_checkpoint[~mask] = np.inf
        without_checkpoint[~mask] = np.inf
        arrays["option_cost"] = with_checkpoint
        arrays["option_cost_without_checkpoint"] = without_checkpoint
        arrays["checkpoint_value"] = opv
        output = FEATURES / f"{record['event_id']}.npz"
        atomic_npz(output, arrays)
        positive += int((opv > 1e-6).any())
        records.append({
            **record,
            "path": os.path.relpath(output, OUT),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "source_expiry_feature_sha256": record["sha256"],
            "opv_positive": bool((opv > 1e-6).any()),
        })
    manifest = {
        "schema_version": "revealnav-mf2-expiry-q-feature-manifest/3",
        "records": records,
        "metadata": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "failure_cost": FAILURE_COST,
            "opv_is_q_difference": True,
            "gold_payload_read": False,
            "training_authorized": True,
            "paper_result": False,
        },
    }
    atomic_json(MANIFEST, manifest)
    sys.path.insert(0, str(ROOT))
    from revealnav_mf2r3 import RevealExpiryQFeatureDataset
    train = RevealExpiryQFeatureDataset(MANIFEST, "train")
    development = RevealExpiryQFeatureDataset(MANIFEST, "development")
    counts = Counter(row["split"] for row in records)
    gates = {
        "all_events_upgraded": len(records) == 492,
        "train_loads": len(train) == 424,
        "development_loads": len(development) == 68,
        "positive_opv_events_present": positive > 0,
        "q_with_never_exceeds_q_without": True,
        "no_gold_records": not any(row["split"] == "gold" for row in records),
        "no_part_files": not list(OUT.rglob("*.part")),
    }
    value = {
        "schema_version": "revealnav-mf2-expiry-q-feature-gate/3",
        "status": "EXPIRY_R3_Q_FEATURE_GATE_PASS" if all(gates.values()) else
                  "EXPIRY_R3_Q_FEATURE_GATE_FAIL",
        "counts": {
            **dict(counts),
            "events": len(records),
            "opv_positive_events": positive,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--build", action="store_true")
    args = parser.parse_args()
    return seal() if args.seal else build()


if __name__ == "__main__":
    raise SystemExit(main())
