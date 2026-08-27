"""Validated variable-candidate examples for MF2.1 head training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


ARRAY_KEYS = {
    "instruction_embedding",
    "history_embeddings",
    "candidate_embeddings",
    "candidate_mask",
    "target_index",
    "target_in_set",
    "separation",
    "evidence_complete",
    "reveal_hazard",
    "option_cost",
    "current_feasibility",
    "checkpoint_value",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Tensor]:
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("feature shard keys do not match the frozen schema")
    history = arrays["history_embeddings"]
    candidates = arrays["candidate_embeddings"]
    mask = arrays["candidate_mask"]
    if history.ndim != 2 or candidates.ndim != 3:
        raise ValueError("history/candidate embeddings have invalid rank")
    steps, feature_dim = history.shape
    if candidates.shape[0] != steps or candidates.shape[2] != feature_dim:
        raise ValueError("history/candidate feature axes do not match")
    candidate_count = candidates.shape[1]
    if candidate_count < 2:
        raise ValueError("a multi-branch example needs at least two candidates")
    expected = {
        "instruction_embedding": (feature_dim,),
        "candidate_mask": (steps, candidate_count),
        "target_index": (steps,),
        "target_in_set": (steps,),
        "separation": (steps,),
        "evidence_complete": (steps,),
        "reveal_hazard": (steps,),
        "option_cost": (steps, candidate_count),
        "checkpoint_value": (steps,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key} has shape {arrays[key].shape}, expected {shape}")
    feasibility = arrays["current_feasibility"]
    if feasibility.ndim != 3 or feasibility.shape[:2] != (
        steps, candidate_count
    ):
        raise ValueError("current_feasibility has invalid shape")
    if feasibility.shape[2] < 1:
        raise ValueError("at least one resource budget is required")
    bool_mask = mask.astype(np.bool_, copy=False)
    target = arrays["target_index"].astype(np.int64, copy=False)
    valid_target = target >= 0
    if np.any(target[valid_target] >= candidate_count):
        raise ValueError("target_index is outside the candidate set")
    if np.any(valid_target & ~bool_mask[np.arange(steps), target.clip(min=0)]):
        raise ValueError("target_index points to a masked candidate")
    result = {
        "instruction_embedding": torch.from_numpy(
            arrays["instruction_embedding"].astype(np.float32, copy=False)
        ),
        "history_embeddings": torch.from_numpy(
            history.astype(np.float32, copy=False)
        ),
        "candidate_embeddings": torch.from_numpy(
            candidates.astype(np.float32, copy=False)
        ),
        "candidate_mask": torch.from_numpy(bool_mask),
        "target_index": torch.from_numpy(target),
    }
    for key in ARRAY_KEYS - {
        "instruction_embedding",
        "history_embeddings", "candidate_embeddings", "candidate_mask",
        "target_index",
    }:
        result[key] = torch.from_numpy(
            arrays[key].astype(np.float32, copy=False)
        )
    return result


class RevealFeatureDataset(Dataset[dict[str, Tensor]]):
    """Read checksummed, non-pickle NPZ shards listed by a manifest."""

    def __init__(self, manifest_path: str | Path, split: str) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != "revealnav-mf2-feature-manifest/1":
            raise ValueError("unsupported feature manifest")
        records = manifest.get("records")
        if not isinstance(records, list):
            raise ValueError("manifest records must be a list")
        self.records = [record for record in records if record.get("split") == split]
        if not self.records:
            raise ValueError(f"manifest has no {split!r} records")
        root = self.manifest_path.parent
        self.paths = []
        for record in self.records:
            path = (root / record["path"]).resolve()
            if root not in path.parents or path.is_symlink() or not path.is_file():
                raise ValueError("unsafe feature shard path")
            if path.stat().st_size != record["bytes"]:
                raise ValueError("feature shard size drift")
            if sha256_file(path) != record["sha256"]:
                raise ValueError("feature shard hash drift")
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as shard:
            arrays = {key: shard[key] for key in shard.files}
        return _validated_arrays(arrays)


def collate_reveal_examples(
    examples: Sequence[dict[str, Tensor]],
) -> dict[str, Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(examples)
    max_steps = max(row["history_embeddings"].shape[0] for row in examples)
    max_candidates = max(row["candidate_embeddings"].shape[1] for row in examples)
    feature_dim = examples[0]["history_embeddings"].shape[1]
    budget_count = examples[0]["current_feasibility"].shape[2]
    output: dict[str, Tensor] = {
        "instruction_embedding": torch.zeros(batch_size, feature_dim),
        "history_embeddings": torch.zeros(batch_size, max_steps, feature_dim),
        "candidate_embeddings": torch.zeros(
            batch_size, max_steps, max_candidates, feature_dim
        ),
        "candidate_mask": torch.zeros(
            batch_size, max_steps, max_candidates, dtype=torch.bool
        ),
        "target_index": torch.full((batch_size, max_steps), -1, dtype=torch.long),
        "target_in_set": torch.full((batch_size, max_steps), -1.0),
        "separation": torch.full((batch_size, max_steps), -1.0),
        "evidence_complete": torch.full((batch_size, max_steps), -1.0),
        "reveal_hazard": torch.full((batch_size, max_steps), -1.0),
        "option_cost": torch.full(
            (batch_size, max_steps, max_candidates), torch.inf
        ),
        "current_feasibility": torch.full(
            (batch_size, max_steps, max_candidates, budget_count), -1.0
        ),
        "checkpoint_value": torch.full((batch_size, max_steps), torch.nan),
        "step_mask": torch.zeros(batch_size, max_steps, dtype=torch.bool),
    }
    for batch_index, row in enumerate(examples):
        steps, candidates, row_feature_dim = row["candidate_embeddings"].shape
        if row_feature_dim != feature_dim:
            raise ValueError("feature dimensions differ within a batch")
        if row["current_feasibility"].shape[2] != budget_count:
            raise ValueError("budget counts differ within a batch")
        output["instruction_embedding"][batch_index] = row[
            "instruction_embedding"
        ]
        output["history_embeddings"][batch_index, :steps] = row[
            "history_embeddings"
        ]
        output["candidate_embeddings"][batch_index, :steps, :candidates] = row[
            "candidate_embeddings"
        ]
        for key in (
            "candidate_mask", "option_cost", "current_feasibility"
        ):
            output[key][batch_index, :steps, :candidates] = row[key]
        for key in (
            "target_index", "target_in_set", "separation",
            "evidence_complete", "reveal_hazard", "checkpoint_value",
        ):
            output[key][batch_index, :steps] = row[key]
        output["step_mask"][batch_index, :steps] = True
    return output


def write_feature_manifest(
    path: Path,
    records: Sequence[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    value = {
        "schema_version": "revealnav-mf2-feature-manifest/1",
        "records": list(records),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
