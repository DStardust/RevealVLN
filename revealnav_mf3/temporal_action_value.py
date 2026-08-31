"""Native-anchored exact action value head for MF3ZN-TUAD v1."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .temporal_uad_model import TEMPORAL_HIDDEN_DIM


ACTION_VALUE_HIDDEN_DIM = 64
HUBER_DELTA = 1.0


class NativeAnchoredActionValue(nn.Module):
    """Predict one-switch utility while keeping the native value exactly zero.

    Native rows are removed before the MLP call.  Consequently their returned
    value is a literal zero and cannot acquire calibration drift or gradients
    through the action-value network.
    """

    def __init__(
        self,
        action_embedding_dim: int,
        action_feature_dim: int,
        temporal_dim: int = TEMPORAL_HIDDEN_DIM,
        hidden_dim: int = ACTION_VALUE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if int(temporal_dim) != TEMPORAL_HIDDEN_DIM:
            raise ValueError("MF3ZN-TUAD v1 fixes temporal state size at 64")
        if int(hidden_dim) != ACTION_VALUE_HIDDEN_DIM:
            raise ValueError("MF3ZN-TUAD v1 fixes action-head hidden size at 64")
        if int(action_embedding_dim) < 1 or int(action_feature_dim) < 1:
            raise ValueError("action dimensions must be positive")
        self.temporal_dim = int(temporal_dim)
        self.action_embedding_dim = int(action_embedding_dim)
        self.action_feature_dim = int(action_feature_dim)
        input_dim = (
            self.temporal_dim
            + 2 * self.action_embedding_dim
            + self.action_feature_dim
        )
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ACTION_VALUE_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(ACTION_VALUE_HIDDEN_DIM, 1),
        )

    def _validate(
        self,
        temporal_state: Tensor,
        native_embedding: Tensor,
        action_embedding: Tensor,
        action_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
        if temporal_state.ndim != 2 or temporal_state.shape[-1] != self.temporal_dim:
            raise ValueError("temporal_state must have shape [batch,64]")
        if native_embedding.ndim != 2 or native_embedding.shape != (
            temporal_state.shape[0], self.action_embedding_dim
        ):
            raise ValueError("native_embedding has the wrong shape")
        set_input = action_embedding.ndim == 3
        if not set_input:
            if action_embedding.ndim != 2:
                raise ValueError("action_embedding must have rank 2 or 3")
            action_embedding = action_embedding.unsqueeze(1)
            action_features = action_features.unsqueeze(1)
        if (
            action_embedding.shape[0] != temporal_state.shape[0]
            or action_embedding.shape[-1] != self.action_embedding_dim
            or action_features.ndim != 3
            or action_features.shape[:2] != action_embedding.shape[:2]
            or action_features.shape[-1] != self.action_feature_dim
        ):
            raise ValueError("action embeddings/features have incompatible shapes")
        tensors = (temporal_state, native_embedding, action_embedding, action_features)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("action-value inputs contain non-finite values")
        return (*tensors, set_input)

    def forward(
        self,
        temporal_state: Tensor,
        native_embedding: Tensor,
        action_embedding: Tensor,
        action_features: Tensor,
        *,
        is_native: Tensor | None = None,
    ) -> Tensor:
        (
            temporal_state,
            native_embedding,
            action_embedding,
            action_features,
            set_input,
        ) = self._validate(
            temporal_state, native_embedding, action_embedding, action_features
        )
        batch, actions, _ = action_embedding.shape
        expanded_native = native_embedding[:, None, :].expand(-1, actions, -1)
        expanded_state = temporal_state[:, None, :].expand(-1, actions, -1)
        if is_native is None:
            native_mask = torch.eq(action_embedding, expanded_native).all(dim=-1)
        else:
            native_mask = torch.as_tensor(is_native, device=action_embedding.device)
            if not set_input and native_mask.ndim == 1:
                native_mask = native_mask.unsqueeze(1)
            if native_mask.dtype is not torch.bool or native_mask.shape != (batch, actions):
                raise ValueError("is_native must be a boolean [batch,actions] mask")

        flat_mask = native_mask.reshape(-1)
        value = action_embedding.new_zeros(batch * actions)
        alternative = ~flat_mask
        if bool(alternative.any()):
            joined = torch.cat(
                (
                    expanded_state,
                    action_embedding,
                    action_embedding - expanded_native,
                    action_features,
                ),
                dim=-1,
            ).reshape(batch * actions, -1)
            predicted = self.network(joined[alternative]).squeeze(-1)
            value = value.masked_scatter(alternative, predicted)
        value = value.reshape(batch, actions)
        return value if set_input else value[:, 0]


def native_anchored_huber_loss(
    predicted: Tensor,
    exact_delta_utility: Tensor,
    action_mask: Tensor,
    is_native: Tensor,
) -> Tensor:
    """Fixed Huber regression over executable non-native treatments only."""

    if not (
        predicted.shape
        == exact_delta_utility.shape
        == action_mask.shape
        == is_native.shape
    ):
        raise ValueError("action-value loss shapes do not match")
    if action_mask.dtype is not torch.bool or is_native.dtype is not torch.bool:
        raise TypeError("action masks must be boolean")
    if bool((is_native & ~action_mask).any()):
        raise ValueError("native action cannot be padding")
    if bool((is_native.sum(dim=-1) != 1).any()):
        raise ValueError("each action set must contain exactly one native action")
    if not bool(torch.isfinite(exact_delta_utility[action_mask]).all()):
        raise ValueError("exact utility targets contain non-finite values")
    if not bool(torch.equal(
        exact_delta_utility[is_native],
        torch.zeros_like(exact_delta_utility[is_native]),
    )):
        raise ValueError("native exact utility must be identically zero")
    if not bool(torch.equal(
        predicted[is_native], torch.zeros_like(predicted[is_native])
    )):
        raise ValueError("native predicted utility drifted away from zero")
    alternatives = action_mask & ~is_native
    if not bool(alternatives.any()):
        raise ValueError("action-value training has no non-native treatment")
    return F.huber_loss(
        predicted[alternatives],
        exact_delta_utility[alternatives],
        delta=HUBER_DELTA,
        reduction="mean",
    )


def choose_native_inclusive_action(
    values: Tensor,
    action_mask: Tensor,
    is_native: Tensor,
) -> Tensor:
    """Argmax over the sealed support; ties deterministically keep native."""

    if values.ndim != 2 or values.shape != action_mask.shape or values.shape != is_native.shape:
        raise ValueError("invalid native-inclusive selection tensors")
    if action_mask.dtype is not torch.bool or is_native.dtype is not torch.bool:
        raise TypeError("selection masks must be boolean")
    if bool((is_native.sum(dim=1) != 1).any()) or bool((is_native & ~action_mask).any()):
        raise ValueError("each support must contain one executable native action")
    if not bool(torch.isfinite(values[action_mask]).all()):
        raise ValueError("non-finite action values")
    # Put native first in the stable comparison.  An alternative must be
    # strictly greater than zero to replace the anchored native action.
    native_index = is_native.to(torch.int64).argmax(dim=1)
    chosen = native_index.clone()
    best = values.gather(1, native_index[:, None]).squeeze(1)
    for index in range(values.shape[1]):
        improve = action_mask[:, index] & (values[:, index] > best)
        chosen = torch.where(improve, torch.full_like(chosen, index), chosen)
        best = torch.where(improve, values[:, index], best)
    return chosen


__all__ = [
    "ACTION_VALUE_HIDDEN_DIM",
    "HUBER_DELTA",
    "NativeAnchoredActionValue",
    "choose_native_inclusive_action",
    "native_anchored_huber_loss",
]
