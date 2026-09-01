"""Fixed residual evidence-memory reranker for MF3ZU.

The module deliberately has no ETP backbone parameters.  Frozen ETP candidate
embeddings and scores enter as detached tensors; only the two small projections
and the residual interaction MLP are trainable.  The shuffled-memory control is
constructed with train-fold donors, so a held decision can never donate memory
to another held decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CANDIDATE_DIM = 768
PROJECTED_DIM = 64
K_MEM = 8
FIXED_SEED = 20_260_901


class EvidenceMemoryRerankerError(ValueError):
    """Raised when a reranker input violates the sealed MF3ZU contract."""


@dataclass(frozen=True)
class FeatureNormalizer:
    """A train-fold-only population normalizer."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if mean.ndim != 1 or scale.shape != mean.shape or mean.size == 0:
            raise EvidenceMemoryRerankerError("normalizer vectors have invalid shapes")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise EvidenceMemoryRerankerError("normalizer contains non-finite values")
        if np.any(scale <= 0.0):
            raise EvidenceMemoryRerankerError("normalizer scale must be positive")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "scale", scale.copy())

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def transform(self, values: np.ndarray) -> np.ndarray:
        value = np.asarray(values, dtype=np.float32)
        if value.shape[-1:] != (self.dimension,):
            raise EvidenceMemoryRerankerError("normalizer input dimension drift")
        if not np.isfinite(value).all():
            raise EvidenceMemoryRerankerError("normalizer input contains non-finite values")
        return ((value - self.mean) / self.scale).astype(np.float32, copy=False)


def fit_feature_normalizer(values: np.ndarray) -> FeatureNormalizer:
    """Fit one normalizer from an explicitly supplied training-fold matrix."""

    value = np.asarray(values, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise EvidenceMemoryRerankerError("normalizer fit needs a non-empty matrix")
    if not np.isfinite(value).all():
        raise EvidenceMemoryRerankerError("normalizer fit matrix is non-finite")
    mean = value.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = value.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-6] = 1.0
    return FeatureNormalizer(mean=mean, scale=scale)


def mean_pool_evidence(
    evidence_features: Tensor,
    evidence_mask: Tensor,
) -> Tensor:
    """Mean-pool at most eight records separately for every candidate.

    MF3ZU's last evidence coordinate is an exact current-candidate binding.
    Consequently a global ``[batch,D]`` memory vector is not a legal model
    input: the same semantic record must be expanded and pooled as
    ``[batch,candidates,K_MEM,D]``.
    """

    if evidence_features.ndim != 4:
        raise EvidenceMemoryRerankerError(
            "evidence_features must have shape [batch,candidates,K,memory_dim]"
        )
    if evidence_mask.dtype is not torch.bool or evidence_mask.shape != evidence_features.shape[:3]:
        raise EvidenceMemoryRerankerError("evidence_mask shape or dtype is invalid")
    if evidence_features.shape[2] != K_MEM:
        raise EvidenceMemoryRerankerError("MF3ZU evidence tensors must use K_MEM=8")
    if not bool(torch.isfinite(evidence_features[evidence_mask]).all()):
        raise EvidenceMemoryRerankerError("active evidence contains non-finite values")
    weights = evidence_mask.to(evidence_features.dtype).unsqueeze(-1)
    count = weights.sum(dim=2).clamp_min(1.0)
    return (evidence_features * weights).sum(dim=2) / count


class EvidenceMemoryResidualReranker(nn.Module):
    """Candidate-specific memory residual added directly to frozen ETP scores."""

    def __init__(
        self,
        memory_dim: int,
        *,
        candidate_dim: int = CANDIDATE_DIM,
        projected_dim: int = PROJECTED_DIM,
    ) -> None:
        super().__init__()
        if int(candidate_dim) != CANDIDATE_DIM:
            raise EvidenceMemoryRerankerError("MF3ZU fixes candidate width at 768")
        if int(projected_dim) != PROJECTED_DIM:
            raise EvidenceMemoryRerankerError("MF3ZU fixes projected width at 64")
        if isinstance(memory_dim, bool) or int(memory_dim) < 1:
            raise EvidenceMemoryRerankerError("memory_dim must be positive")
        self.memory_dim = int(memory_dim)
        self.candidate_projection = nn.Linear(CANDIDATE_DIM, PROJECTED_DIM)
        self.memory_projection = nn.Linear(self.memory_dim, PROJECTED_DIM)
        self.interaction = nn.Sequential(
            nn.Linear(3 * PROJECTED_DIM, PROJECTED_DIM),
            nn.GELU(),
            nn.Linear(PROJECTED_DIM, 1),
        )

    def forward(
        self,
        candidate_features: Tensor,
        base_scores: Tensor,
        candidate_mask: Tensor,
        pooled_memory: Tensor,
    ) -> Tensor:
        if (
            candidate_features.ndim != 3
            or candidate_features.shape[-1] != CANDIDATE_DIM
            or base_scores.shape != candidate_features.shape[:2]
            or candidate_mask.shape != candidate_features.shape[:2]
            or candidate_mask.dtype is not torch.bool
        ):
            raise EvidenceMemoryRerankerError("candidate reranker tensors have invalid shapes")
        if pooled_memory.ndim != 3 or pooled_memory.shape != (
            candidate_features.shape[0], candidate_features.shape[1], self.memory_dim
        ):
            raise EvidenceMemoryRerankerError(
                "pooled_memory must be candidate-specific [batch,candidates,memory_dim]"
            )
        if bool((candidate_mask.sum(dim=1) < 2).any()):
            raise EvidenceMemoryRerankerError("each decision needs at least two candidates")
        if not bool(torch.isfinite(candidate_features[candidate_mask]).all()):
            raise EvidenceMemoryRerankerError("active candidate features are non-finite")
        if not bool(torch.isfinite(base_scores[candidate_mask]).all()):
            raise EvidenceMemoryRerankerError("active base scores are non-finite")
        if not bool(torch.isfinite(pooled_memory).all()):
            raise EvidenceMemoryRerankerError("pooled memory is non-finite")

        # The ETP representation is observational input, never a trainable
        # upstream graph in this decision probe.
        candidate = candidate_features.detach()
        base = base_scores.detach()
        memory = pooled_memory.detach()
        candidate_state = self.candidate_projection(candidate)
        memory_state = self.memory_projection(memory)
        joined = torch.cat(
            (candidate_state, memory_state, candidate_state * memory_state), dim=-1
        )
        residual = self.interaction(joined).squeeze(-1)
        return (base + residual).masked_fill(~candidate_mask, float("-inf"))


def masked_candidate_cross_entropy(
    scores: Tensor,
    target_index: Tensor,
    candidate_mask: Tensor,
) -> Tensor:
    if scores.ndim != 2 or scores.shape != candidate_mask.shape:
        raise EvidenceMemoryRerankerError("ranking score/mask shapes do not match")
    if candidate_mask.dtype is not torch.bool:
        raise EvidenceMemoryRerankerError("candidate mask must be boolean")
    if target_index.dtype != torch.long or target_index.shape != (scores.shape[0],):
        raise EvidenceMemoryRerankerError("target_index must be int64 [batch]")
    if bool(((target_index < 0) | (target_index >= scores.shape[1])).any()):
        raise EvidenceMemoryRerankerError("candidate target is out of bounds")
    if bool((~candidate_mask.gather(1, target_index[:, None]).squeeze(1)).any()):
        raise EvidenceMemoryRerankerError("candidate target is not executable")
    masked = scores.masked_fill(~candidate_mask, float("-inf"))
    if not bool(torch.isfinite(masked[candidate_mask]).all()):
        raise EvidenceMemoryRerankerError("active ranking scores are non-finite")
    return F.cross_entropy(masked, target_index)


def parameter_sha256(model: nn.Module) -> str:
    """Canonical hash used to prove common B/C initialization."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def common_initialized_rerankers(
    memory_dim: int,
    *,
    seed: int = FIXED_SEED,
) -> tuple[EvidenceMemoryResidualReranker, EvidenceMemoryResidualReranker, str]:
    """Create the true/shuffled models with byte-identical parameters."""

    if int(seed) != FIXED_SEED:
        raise EvidenceMemoryRerankerError("MF3ZU uses one fixed seed")
    torch.manual_seed(int(seed))
    true_model = EvidenceMemoryResidualReranker(memory_dim)
    shuffled_model = EvidenceMemoryResidualReranker(memory_dim)
    shuffled_model.load_state_dict(true_model.state_dict())
    true_hash = parameter_sha256(true_model)
    if parameter_sha256(shuffled_model) != true_hash:
        raise RuntimeError("B/C common initialization failed")
    return true_model, shuffled_model, true_hash


def _hash_order(seed: int, event_id: str, donor_id: str, context: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{context}\0{event_id}\0{donor_id}".encode("utf-8")
    ).hexdigest()


def shuffled_memory_donor_indices(
    event_ids: Sequence[object],
    memory_counts: Sequence[int],
    train_indices: Sequence[int],
    held_indices: Sequence[int],
    *,
    candidate_counts: Sequence[int] | None = None,
    seed: int = FIXED_SEED,
) -> tuple[np.ndarray, dict[str, object]]:
    """Assign count-matched train donors without target or held-fold access.

    Training events are deranged within each count stratum when possible.
    Singleton strata fall back to the closest available count, while held
    events always draw from the training pool.  No correctness label, score, or
    outcome participates in this mapping.
    """

    if int(seed) != FIXED_SEED:
        raise EvidenceMemoryRerankerError("MF3ZU uses one fixed shuffle seed")
    ids = np.asarray([str(value) for value in event_ids])
    counts = np.asarray(memory_counts, dtype=np.int64)
    if ids.ndim != 1 or counts.shape != ids.shape or len(set(ids.tolist())) != len(ids):
        raise EvidenceMemoryRerankerError("invalid or repeated event identities")
    if np.any((counts < 0) | (counts > K_MEM)):
        raise EvidenceMemoryRerankerError("memory count is outside [0,8]")
    candidates = (
        np.zeros_like(counts)
        if candidate_counts is None
        else np.asarray(candidate_counts, dtype=np.int64)
    )
    if candidates.shape != counts.shape or np.any(candidates < 0):
        raise EvidenceMemoryRerankerError("candidate counts are invalid")
    train = np.asarray(train_indices, dtype=np.int64)
    held = np.asarray(held_indices, dtype=np.int64)
    if train.ndim != 1 or held.ndim != 1 or len(train) < 2 or len(held) == 0:
        raise EvidenceMemoryRerankerError("shuffle needs >=2 train and >=1 held events")
    if np.any(train < 0) or np.any(train >= len(ids)) or np.any(held < 0) or np.any(held >= len(ids)):
        raise EvidenceMemoryRerankerError("shuffle index is out of bounds")
    if set(train.tolist()) & set(held.tolist()):
        raise EvidenceMemoryRerankerError("train and held events overlap")

    donor = np.full(len(ids), -1, dtype=np.int64)
    groups: dict[tuple[int, int], list[int]] = {}
    for index in train.tolist():
        groups.setdefault(
            (int(counts[index]), int(candidates[index])), []
        ).append(index)
    # A stable hashed cyclic shift is a strict derangement for non-singletons.
    for stratum, members in groups.items():
        ordered = sorted(
            members,
            key=lambda index: _hash_order(
                seed, ids[index], str(stratum), "train-group"
            ),
        )
        if len(ordered) >= 2:
            for position, index in enumerate(ordered):
                donor[index] = ordered[(position + 1) % len(ordered)]

    train_list = train.tolist()
    for index in train_list:
        if donor[index] >= 0:
            continue
        choices = [value for value in train_list if value != index]
        choices.sort(key=lambda value: (
            abs(int(counts[value]) - int(counts[index])),
            abs(int(candidates[value]) - int(candidates[index])),
            _hash_order(seed, ids[index], ids[value], "train-fallback"),
        ))
        donor[index] = choices[0]

    for index in held.tolist():
        choices = list(train_list)
        choices.sort(key=lambda value: (
            abs(int(counts[value]) - int(counts[index])),
            abs(int(candidates[value]) - int(candidates[index])),
            _hash_order(seed, ids[index], ids[value], "held-from-train"),
        ))
        donor[index] = choices[0]

    relevant = np.concatenate((train, held))
    if np.any(donor[relevant] < 0):
        raise RuntimeError("shuffled-memory donor assignment is incomplete")
    if np.any(donor[train] == train):
        raise RuntimeError("training shuffled-memory mapping is not a derangement")
    train_set = set(train.tolist())
    if any(int(value) not in train_set for value in donor[held]):
        raise RuntimeError("held memory donor escaped the training fold")
    matched = counts[relevant] == counts[donor[relevant]]
    candidate_matched = candidates[relevant] == candidates[donor[relevant]]
    return donor, {
        "train_event_count": int(len(train)),
        "held_event_count": int(len(held)),
        "train_derangement": True,
        "held_donors_train_only": True,
        "count_matched_events": int(matched.sum()),
        "count_matched_rate": float(matched.mean()),
        "count_match_preferred": True,
        "candidate_count_matched_events": int(candidate_matched.sum()),
        "candidate_count_matched_rate": float(candidate_matched.mean()),
        "candidate_count_match_preferred": candidate_counts is not None,
        "outcome_or_target_used": False,
        "seed": int(seed),
    }


__all__ = [
    "CANDIDATE_DIM",
    "PROJECTED_DIM",
    "K_MEM",
    "FIXED_SEED",
    "EvidenceMemoryRerankerError",
    "FeatureNormalizer",
    "fit_feature_normalizer",
    "mean_pool_evidence",
    "EvidenceMemoryResidualReranker",
    "masked_candidate_cross_entropy",
    "parameter_sha256",
    "common_initialized_rerankers",
    "shuffled_memory_donor_indices",
]
