#!/usr/bin/env python3
"""Build a tiny full-set fixture and exercise the real MF2.1 trainer."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla").resolve()
OUT = ROOT / "artifacts/training/mf2_smoke"
SEED = 20260826


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def example(index: int, steps: int, candidates: int):
    generator = np.random.default_rng(SEED + index)
    feature_dim = 32
    history = generator.normal(size=(steps, feature_dim)).astype(np.float32)
    candidate = generator.normal(
        size=(steps, candidates, feature_dim)
    ).astype(np.float32)
    mask = np.ones((steps, candidates), dtype=np.bool_)
    target = index % candidates
    base_cost = np.arange(candidates, dtype=np.float32)
    option_cost = np.broadcast_to(
        np.roll(base_cost, target)[None, :], (steps, candidates)
    ).copy()
    reveal_at = max(1, steps // 2)
    evidence = (np.arange(steps) >= reveal_at).astype(np.float32)
    target_index = np.full(steps, -1, dtype=np.int64)
    target_index[reveal_at:] = target
    budgets = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    feasibility = (
        option_cost[:, :, None] <= budgets[None, None, :]
    ).astype(np.float32)
    reveal_hazard = np.zeros(steps, dtype=np.float32)
    reveal_hazard[reveal_at] = 1.0
    return {
        "instruction_embedding": generator.normal(
            size=(feature_dim,)
        ).astype(np.float32),
        "history_embeddings": history,
        "candidate_embeddings": candidate,
        "candidate_mask": mask,
        "target_index": target_index,
        "target_in_set": np.ones(steps, dtype=np.float32),
        "separation": evidence,
        "evidence_complete": evidence,
        "reveal_hazard": reveal_hazard,
        "option_cost": option_cost,
        "current_feasibility": feasibility,
        "checkpoint_value": (option_cost.min(axis=1) + 0.5).astype(np.float32),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    layouts = [
        (5, 2, "train", "scene_train_a"),
        (6, 3, "train", "scene_train_b"),
        (7, 4, "train", "scene_train_c"),
        (5, 3, "development", "scene_dev_a"),
        (6, 4, "development", "scene_dev_b"),
    ]
    for index, (steps, candidates, split, scene_id) in enumerate(layouts):
        path = OUT / f"fixture_{index}.npz"
        np.savez(path, **example(index, steps, candidates))
        records.append({
            "event_id": f"synthetic_{index}",
            "scene_id": scene_id,
            "split": split,
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "candidate_count": candidates,
        })
    manifest = OUT / "feature_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "revealnav-mf2-feature-manifest/1",
        "records": records,
        "metadata": {
            "synthetic": True,
            "training_authorized": False,
            "normalized_budgets": [0.5, 1.0, 2.0, 4.0],
            "full_candidate_sets": True,
            "future_frames_used": 0,
            "paper_result": False,
        },
    }, indent=2, sort_keys=True) + "\n")
    command = [
        sys.executable,
        str(ROOT / "scripts/train_revealnav_mf2_heads.py"),
        "--manifest", str(manifest),
        "--output-dir", str(OUT / "run"),
        "--epochs", "3",
        "--batch-size", "2",
        "--hidden-dim", "24",
        "--device", "cpu",
        "--smoke-test",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
