"""R3 data contract with paired checkpoint/no-checkpoint Q labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from revealnav_mf2.data import ARRAY_KEYS, _validated_arrays, sha256_file
from .data import collate_reveal_expiry_examples


Q_ARRAY_KEY = "option_cost_without_checkpoint"
Q_ARRAY_KEYS = ARRAY_KEYS | {"expiry_hazard", Q_ARRAY_KEY}


class RevealExpiryQFeatureDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, manifest_path: str | Path, split: str) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != "revealnav-mf2-expiry-q-feature-manifest/3":
            raise ValueError("unsupported paired-Q feature manifest")
        self.records = [
            row for row in manifest.get("records", []) if row.get("split") == split
        ]
        if not self.records:
            raise ValueError(f"manifest has no {split!r} records")
        root = self.manifest_path.parent
        self.paths = []
        for record in self.records:
            path = (root / record["path"]).resolve()
            if root not in path.parents or path.is_symlink() or not path.is_file():
                raise ValueError("unsafe paired-Q feature path")
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record[
                "sha256"
            ]:
                raise ValueError("paired-Q feature provenance drift")
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as shard:
            if set(shard.files) != Q_ARRAY_KEYS:
                raise ValueError("paired-Q feature keys do not match R3")
            expiry = shard["expiry_hazard"]
            without = shard[Q_ARRAY_KEY]
            arrays = {key: shard[key] for key in ARRAY_KEYS}
        example = _validated_arrays(arrays)
        steps, candidates = example["candidate_mask"].shape
        if expiry.shape != (steps,) or without.shape != (steps, candidates):
            raise ValueError("paired-Q target shape mismatch")
        if not np.isin(expiry, (-1.0, 0.0, 1.0)).all():
            raise ValueError("expiry target contains an invalid label")
        finite = np.isfinite(without) & np.isfinite(arrays["option_cost"])
        if np.any(without[finite] + 1e-6 < arrays["option_cost"][finite]):
            raise ValueError("Q_without is lower than Q_with")
        example["expiry_hazard"] = torch.from_numpy(
            expiry.astype(np.float32, copy=False)
        )
        example[Q_ARRAY_KEY] = torch.from_numpy(
            without.astype(np.float32, copy=False)
        )
        return example


def collate_reveal_expiry_q_examples(
    examples: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    expiry_examples = [
        {key: value for key, value in example.items() if key != Q_ARRAY_KEY}
        for example in examples
    ]
    output = collate_reveal_expiry_examples(expiry_examples)
    without = torch.full_like(output["option_cost"], torch.inf)
    for batch_index, example in enumerate(examples):
        steps, candidates = example[Q_ARRAY_KEY].shape
        without[batch_index, :steps, :candidates] = example[Q_ARRAY_KEY]
    output[Q_ARRAY_KEY] = without
    return output
