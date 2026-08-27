"""Validated event-level data for checkpointed branch-excursion costs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from revealnav_mf2.data import sha256_file


class BranchExcursionDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self, manifest_path: str | Path, event_ids: set[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema_version") != (
            "revealnav-mf2-branch-excursion-label-manifest/4"
        ):
            raise ValueError("unsupported branch-excursion manifest")
        rows = manifest.get("records", [])
        self.records = [
            row for row in rows
            if event_ids is None or row["event_id"] in event_ids
        ]
        if not self.records:
            raise ValueError("branch-excursion selection is empty")
        root = self.manifest_path.parent
        self.label_paths = []
        self.feature_paths = []
        for row in self.records:
            label = (root / row["path"]).resolve()
            feature = (Path("/mnt/daiyang/vla") / row["online_feature_path"]).resolve()
            if (
                root not in label.parents or label.is_symlink() or not label.is_file()
                or label.stat().st_size != row["bytes"]
                or sha256_file(label) != row["sha256"]
            ):
                raise ValueError("branch-excursion label provenance drift")
            project = Path("/mnt/daiyang/vla").resolve()
            if (
                project not in feature.parents or feature.is_symlink()
                or not feature.is_file()
                or sha256_file(feature) != row["online_feature_sha256"]
            ):
                raise ValueError("online feature provenance drift")
            self.label_paths.append(label)
            self.feature_paths.append(feature)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        label = json.loads(self.label_paths[index].read_text())
        step = int(label["online_feature_relative_step"])
        with np.load(self.feature_paths[index], allow_pickle=False) as shard:
            instruction = shard["instruction_embedding"].astype(np.float32, copy=False)
            history = shard["history_embeddings"][:step + 1].astype(
                np.float32, copy=False
            )
            candidates = shard["candidate_embeddings"][:step + 1].astype(
                np.float32, copy=False
            )
            mask = shard["candidate_mask"][:step + 1].astype(np.bool_, copy=False)
        rows = sorted(label["labels"], key=lambda row: row["branch_index"])
        count = len(rows)
        if (
            candidates.shape[1] != count
            or [row["branch_index"] for row in rows] != list(range(count))
            or not mask[-1].all()
        ):
            raise ValueError("decision-time branch/feature alignment failure")
        commit = np.asarray([row["commit_cost"] for row in rows], np.float32)
        excursion = np.asarray(
            [row["checkpointed_excursion_cost"] for row in rows], np.float32
        )
        if not np.isfinite(commit).all() or not np.isfinite(excursion).all():
            raise ValueError("action costs must be finite")
        return {
            "instruction_embedding": torch.from_numpy(instruction),
            "history_embeddings": torch.from_numpy(history),
            "candidate_embeddings": torch.from_numpy(candidates),
            "candidate_mask": torch.from_numpy(mask),
            "commit_cost": torch.from_numpy(commit),
            "excursion_cost": torch.from_numpy(excursion),
            "decision_index": torch.tensor(step, dtype=torch.long),
        }


def collate_branch_excursion_examples(
    examples: Sequence[dict[str, Tensor]],
) -> dict[str, Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty branch-excursion batch")
    batch = len(examples)
    steps = max(row["history_embeddings"].shape[0] for row in examples)
    candidates = max(row["candidate_embeddings"].shape[1] for row in examples)
    feature_dim = examples[0]["history_embeddings"].shape[1]
    output = {
        "instruction_embedding": torch.zeros(batch, feature_dim),
        "history_embeddings": torch.zeros(batch, steps, feature_dim),
        "candidate_embeddings": torch.zeros(
            batch, steps, candidates, feature_dim
        ),
        "candidate_mask": torch.zeros(
            batch, steps, candidates, dtype=torch.bool
        ),
        "commit_cost": torch.full((batch, candidates), torch.inf),
        "excursion_cost": torch.full((batch, candidates), torch.inf),
        "decision_index": torch.zeros(batch, dtype=torch.long),
    }
    for index, row in enumerate(examples):
        row_steps, row_candidates, row_dim = row["candidate_embeddings"].shape
        if row_dim != feature_dim:
            raise ValueError("feature dimensions differ within a batch")
        output["instruction_embedding"][index] = row["instruction_embedding"]
        output["history_embeddings"][index, :row_steps] = row[
            "history_embeddings"
        ]
        output["candidate_embeddings"][index, :row_steps, :row_candidates] = row[
            "candidate_embeddings"
        ]
        output["candidate_mask"][index, :row_steps, :row_candidates] = row[
            "candidate_mask"
        ]
        output["commit_cost"][index, :row_candidates] = row["commit_cost"]
        output["excursion_cost"][index, :row_candidates] = row["excursion_cost"]
        output["decision_index"][index] = row_steps - 1
    return output
