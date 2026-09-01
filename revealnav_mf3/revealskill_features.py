"""Causal feature construction for the fixed RevealSkill representation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
from torch import Tensor, nn

from .evidence_memory import EvidenceMemory
from .option_graph import OptionGraph, OptionStatus


FEATURE_NAMES = (
    "candidate_count",
    "active_option_count",
    "preserved_option_count",
    "resolved_evidence_count",
    "stale_evidence_count",
    "frontier_count",
    "mean_semantic_score",
    "max_semantic_score",
    "step",
)


def build_revealskill_features(
    etp_state: Mapping[str, object],
    evidence_memory: EvidenceMemory,
    option_graph: OptionGraph,
    *,
    active_frontier: Sequence[str] = (),
) -> np.ndarray:
    """Build only snapshot/memory features; no outcome or future fields."""

    forbidden = {"target", "delta_utility", "reward", "success", "spl", "ndtw", "sdtw", "outcome", "future", "oracle", "pose", "navmesh"}
    for key in etp_state:
        lowered = str(key).casefold()
        if lowered in forbidden or lowered.startswith(("future_", "outcome_", "oracle_", "treatment_")):
            raise ValueError(f"forbidden ETP state field: {key}")
    candidates = etp_state.get("executable_candidates", ())
    if not isinstance(candidates, (list, tuple)):
        raise TypeError("executable_candidates must be a sequence")
    items = evidence_memory.items()
    scores = [item.semantic_score for item in items]
    values = np.asarray([
        float(len(candidates)),
        float(len(option_graph.active_options())),
        float(sum(node.status is OptionStatus.PRESERVED for node in option_graph.nodes())),
        float(len(evidence_memory.resolved_constraints())),
        float(sum(item.status.value == "STALE" for item in items)),
        float(len(tuple(active_frontier))),
        float(np.mean(scores)) if scores else 0.0,
        float(np.max(scores)) if scores else 0.0,
        float(etp_state.get("step", 0)),
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("RevealSkill features are non-finite")
    values.flags.writeable = False
    return values


class ConstraintConditionedReadout(nn.Module):
    """The single fixed width-64 readout used by the REE revision."""

    def __init__(self, state_dim: int = 64, constraint_dim: int = 64, width: int = 64):
        super().__init__()
        if (state_dim, constraint_dim, width) != (64, 64, 64):
            raise ValueError("RevealSkill readout dimensions are fixed at 64")
        self.net = nn.Sequential(
            nn.Linear(state_dim + constraint_dim + state_dim, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, state: Tensor, constraint_embedding: Tensor) -> Tensor:
        if state.shape != constraint_embedding.shape or state.ndim < 2 or state.shape[-1] != 64:
            raise ValueError("state and constraint embeddings must align at width 64")
        return self.net(torch.cat((state, constraint_embedding, state * constraint_embedding), dim=-1)).squeeze(-1)


__all__ = ["FEATURE_NAMES", "ConstraintConditionedReadout", "build_revealskill_features"]
