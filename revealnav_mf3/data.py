"""Checksummed exact-online ETP feature episodes for MF3 UAD training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

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
    "native_index",
    "native_scores",
    "outside_score",
    "target_in_set",
    "separation",
    "evidence_complete",
    "reveal_hazard",
    "expiry_hazard",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_shard(path: Path) -> dict[str, Tensor]:
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != ARRAY_KEYS:
            raise ValueError("MF3 online feature keys drift")
        arrays = {name: source[name] for name in source.files}
    history = arrays["history_embeddings"]
    candidates = arrays["candidate_embeddings"]
    mask = arrays["candidate_mask"]
    if history.ndim != 2 or candidates.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("MF3 online feature ranks or mask dtype drift")
    steps, feature_dim = history.shape
    candidate_feature_dim = candidates.shape[2]
    if (
        feature_dim != 768
        or candidates.shape[0] != steps
        or candidate_feature_dim not in (768, 1536)
        or mask.shape != candidates.shape[:2]
        or candidates.shape[1] < 2
    ):
        raise ValueError("MF3 online feature axes drift")
    candidates_count = candidates.shape[1]
    expected = {
        "instruction_embedding": (feature_dim,),
        "target_index": (steps,),
        "native_index": (steps,),
        "native_scores": (steps, candidates_count),
        "outside_score": (steps,),
        "target_in_set": (steps,),
        "separation": (steps,),
        "evidence_complete": (steps,),
        "reveal_hazard": (steps,),
        "expiry_hazard": (steps,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape drift")
    for name in (
        "target_in_set", "separation", "evidence_complete",
        "reveal_hazard", "expiry_hazard",
    ):
        if not np.isin(arrays[name], (-1.0, 0.0, 1.0)).all():
            raise ValueError(f"{name} label drift")
    for name in ("target_index", "native_index"):
        index = arrays[name].astype(np.int64, copy=False)
        valid = index >= 0
        if np.any(index[valid] >= candidates_count):
            raise ValueError(f"{name} outside candidate tensor")
        if np.any(valid & ~mask[np.arange(steps), index.clip(min=0)]):
            raise ValueError(f"{name} points to a masked candidate")
    if not (
        np.isfinite(history).all()
        and np.isfinite(candidates).all()
        and np.isfinite(arrays["instruction_embedding"]).all()
    ):
        raise ValueError("MF3 embeddings must be finite")
    return {
        "instruction_embedding": torch.from_numpy(
            arrays["instruction_embedding"].astype(np.float32, copy=False)
        ),
        "history_embeddings": torch.from_numpy(
            history.astype(np.float32, copy=False)
        ),
        "candidate_embeddings": torch.from_numpy(
            candidates.astype(np.float32, copy=False)
        ),
        "candidate_mask": torch.from_numpy(mask),
        "target_index": torch.from_numpy(
            arrays["target_index"].astype(np.int64, copy=False)
        ),
        "native_index": torch.from_numpy(
            arrays["native_index"].astype(np.int64, copy=False)
        ),
        "native_scores": torch.from_numpy(
            arrays["native_scores"].astype(np.float32, copy=False)
        ),
        "outside_score": torch.from_numpy(
            arrays["outside_score"].astype(np.float32, copy=False)
        ),
        **{
            name: torch.from_numpy(arrays[name].astype(np.float32, copy=False))
            for name in (
                "target_in_set", "separation", "evidence_complete",
                "reveal_hazard", "expiry_hazard",
            )
        },
    }


class OnlineUADFeatureDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, manifest_path: str | Path, split: str) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != "revealnav-mf3b-online-manifest/1":
            raise ValueError("unsupported MF3 online manifest")
        root = self.manifest_path.parents[4]
        self.records = [
            row for row in manifest.get("records", []) if row.get("split") == split
        ]
        if not self.records:
            raise ValueError(f"MF3 online manifest has no {split!r} records")
        self.paths = []
        for row in self.records:
            path = (root / row["path"]).resolve()
            if root not in path.parents or path.is_symlink() or not path.is_file():
                raise ValueError("unsafe MF3 online feature path")
            if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
                raise ValueError("MF3 online feature provenance drift")
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return _load_shard(self.paths[index])


def collate_online_uad(
    examples: Sequence[dict[str, Tensor]],
) -> dict[str, Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty MF3 batch")
    batch_size = len(examples)
    max_steps = max(row["history_embeddings"].shape[0] for row in examples)
    max_candidates = max(
        row["candidate_embeddings"].shape[1] for row in examples
    )
    candidate_feature_dim = examples[0]["candidate_embeddings"].shape[2]
    if any(
        row["candidate_embeddings"].shape[2] != candidate_feature_dim
        for row in examples
    ):
        raise ValueError("mixed MF3 candidate feature dimensions")
    output = {
        "instruction_embedding": torch.zeros(batch_size, 768),
        "history_embeddings": torch.zeros(batch_size, max_steps, 768),
        "candidate_embeddings": torch.zeros(
            batch_size, max_steps, max_candidates, candidate_feature_dim
        ),
        "candidate_mask": torch.zeros(
            batch_size, max_steps, max_candidates, dtype=torch.bool
        ),
        "target_index": torch.full(
            (batch_size, max_steps), -1, dtype=torch.long
        ),
        "native_index": torch.full(
            (batch_size, max_steps), -1, dtype=torch.long
        ),
        "native_scores": torch.full(
            (batch_size, max_steps, max_candidates), -torch.inf
        ),
        "outside_score": torch.full((batch_size, max_steps), -torch.inf),
        "step_mask": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        **{
            name: torch.full((batch_size, max_steps), -1.0)
            for name in (
                "target_in_set", "separation", "evidence_complete",
                "reveal_hazard", "expiry_hazard",
            )
        },
    }
    for batch_index, row in enumerate(examples):
        steps, candidates, _ = row["candidate_embeddings"].shape
        output["instruction_embedding"][batch_index] = row[
            "instruction_embedding"
        ]
        output["history_embeddings"][batch_index, :steps] = row[
            "history_embeddings"
        ]
        output["candidate_embeddings"][batch_index, :steps, :candidates] = row[
            "candidate_embeddings"
        ]
        for name in ("candidate_mask", "native_scores"):
            output[name][batch_index, :steps, :candidates] = row[name]
        for name in (
            "target_index", "native_index", "outside_score",
            "target_in_set", "separation",
            "evidence_complete", "reveal_hazard", "expiry_hazard",
        ):
            output[name][batch_index, :steps] = row[name]
        output["step_mask"][batch_index, :steps] = True
    return output
