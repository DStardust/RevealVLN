"""R3 feature reader with a discrete-time expiry target."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from revealnav_mf2.data import (
    ARRAY_KEYS,
    _validated_arrays,
    collate_reveal_examples,
    sha256_file,
)


R3_ARRAY_KEYS = ARRAY_KEYS | {"expiry_hazard"}


class RevealExpiryFeatureDataset(Dataset[dict[str, Tensor]]):
    """Read checksummed R3 shards without permitting pickle payloads."""

    def __init__(self, manifest_path: str | Path, split: str) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != "revealnav-mf2-expiry-feature-manifest/3":
            raise ValueError("unsupported expiry feature manifest")
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
                raise ValueError("unsafe expiry feature shard path")
            if path.stat().st_size != record["bytes"]:
                raise ValueError("expiry feature shard size drift")
            if sha256_file(path) != record["sha256"]:
                raise ValueError("expiry feature shard hash drift")
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as shard:
            if set(shard.files) != R3_ARRAY_KEYS:
                raise ValueError("expiry feature shard keys do not match R3")
            expiry = shard["expiry_hazard"]
            arrays = {key: shard[key] for key in ARRAY_KEYS}
        example = _validated_arrays(arrays)
        steps = example["history_embeddings"].shape[0]
        if expiry.shape != (steps,):
            raise ValueError("expiry_hazard shape mismatch")
        if not np.isin(expiry, (-1.0, 0.0, 1.0)).all():
            raise ValueError("expiry_hazard contains an invalid label")
        if int((expiry == 1.0).sum()) > 1:
            raise ValueError("expiry_hazard has more than one event")
        example["expiry_hazard"] = torch.from_numpy(
            expiry.astype(np.float32, copy=False)
        )
        return example


def collate_reveal_expiry_examples(
    examples: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    base_examples = [
        {key: value for key, value in example.items() if key != "expiry_hazard"}
        for example in examples
    ]
    output = collate_reveal_examples(base_examples)
    expiry = torch.full(output["step_mask"].shape, -1.0)
    for index, example in enumerate(examples):
        steps = example["expiry_hazard"].shape[0]
        expiry[index, :steps] = example["expiry_hazard"]
    output["expiry_hazard"] = expiry
    return output
