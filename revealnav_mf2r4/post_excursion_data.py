"""Validated reached-branch examples for state-conditioned BACKTRACK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from revealnav_mf2.data import sha256_file


PROJECT = Path("/mnt/daiyang/vla").resolve()


class PostExcursionDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, manifest_path: str | Path, split: str) -> None:
        if split not in ("train", "development"):
            raise ValueError("post-excursion split must be train or development")
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != (
            "revealnav-mf2-post-excursion-full-manifest/4.7"
        ) or manifest.get("metadata", {}).get("training_authorized") is not True:
            raise ValueError("unsupported or unauthorized post-excursion manifest")
        self.examples: list[tuple[dict, Path, Path, int, dict]] = []
        for record in manifest.get("records", []):
            if record["split"] != split:
                continue
            feature = (PROJECT / record["feature_path"]).resolve()
            label = (PROJECT / record["label_path"]).resolve()
            if any(PROJECT not in path.parents for path in (feature, label)):
                raise ValueError("post-excursion path outside project")
            if (
                feature.is_symlink() or label.is_symlink()
                or not feature.is_file() or not label.is_file()
                or feature.stat().st_size != record["feature_bytes"]
                or label.stat().st_size != record["label_bytes"]
                or sha256_file(feature) != record["feature_sha256"]
                or sha256_file(label) != record["label_sha256"]
            ):
                raise ValueError("post-excursion provenance drift")
            payload = json.loads(label.read_text())
            for branch in payload["branches"]:
                if branch["trainable"]:
                    self.examples.append((
                        record, feature, label, int(branch["branch_index"]), branch,
                    ))
        if not self.examples:
            raise ValueError("post-excursion selection is empty")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        _, feature_path, _, branch_index, branch = self.examples[index]
        with np.load(feature_path, allow_pickle=False) as shard:
            if any("target" in key.lower() for key in shard.files):
                raise ValueError("target truth field found in model input")
            reachable = shard["reachable_mask"].astype(np.bool_, copy=False)
            if branch_index >= len(reachable) or not reachable[branch_index]:
                raise ValueError("trainable branch lacks reached-state input")
            pre = shard["pre_history_embeddings"].astype(np.float32, copy=False)
            post = shard["post_history_embeddings"][branch_index].astype(
                np.float32, copy=False
            )
            history = np.concatenate((pre, post[None]), axis=0)
            values = {
                "history_embeddings": torch.from_numpy(history),
                "instruction_embedding": torch.from_numpy(
                    shard["instruction_embedding"].astype(np.float32, copy=False)
                ),
                "selected_branch_embedding": torch.from_numpy(
                    shard["selected_branch_embeddings"][branch_index].astype(
                        np.float32, copy=False
                    )
                ),
                "checkpoint_embedding": torch.from_numpy(
                    shard["checkpoint_embedding"].astype(np.float32, copy=False)
                ),
                "post_candidate_embedding": torch.from_numpy(
                    shard["post_candidate_embeddings"][branch_index].astype(
                        np.float32, copy=False
                    )
                ),
                "normalized_excursion_elapsed": torch.tensor(
                    float(shard["normalized_excursion_elapsed"][branch_index]),
                    dtype=torch.float32,
                ),
            }
        values["history_length"] = torch.tensor(history.shape[0], dtype=torch.long)
        values["continue_cost"] = torch.tensor(
            float(branch["continue_cost"]), dtype=torch.float32
        )
        values["backtrack_cost"] = torch.tensor(
            float(branch["backtrack_cost"]), dtype=torch.float32
        )
        return values


def collate_post_excursion_examples(
    examples: Sequence[dict[str, Tensor]],
) -> dict[str, Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty post-excursion batch")
    batch = len(examples)
    steps = max(int(row["history_length"]) for row in examples)
    feature_dim = examples[0]["history_embeddings"].shape[-1]
    output = {
        "history_embeddings": torch.zeros(batch, steps, feature_dim),
        "history_length": torch.zeros(batch, dtype=torch.long),
    }
    vector_keys = (
        "instruction_embedding", "selected_branch_embedding",
        "checkpoint_embedding", "post_candidate_embedding",
    )
    scalar_keys = (
        "normalized_excursion_elapsed", "continue_cost", "backtrack_cost",
    )
    output.update({key: torch.zeros(batch, feature_dim) for key in vector_keys})
    output.update({key: torch.zeros(batch) for key in scalar_keys})
    for index, row in enumerate(examples):
        length, dimension = row["history_embeddings"].shape
        if dimension != feature_dim:
            raise ValueError("post-excursion feature dimensions differ")
        output["history_embeddings"][index, :length] = row["history_embeddings"]
        output["history_length"][index] = length
        for key in vector_keys + scalar_keys:
            output[key][index] = row[key]
    return output
