"""Causal pairwise net-advantage head for sparse topology interventions."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


ONLINE_SCORE_DEFINITION = (
    "p_better*positive_gain-(1-p_better)*"
    "2*online_checkpoint_to_alternative_euclidean_distance"
)


class PairwiseNetAdvantageHead(nn.Module):
    """Predict whether one alternative beats the frozen policy's native branch."""

    def __init__(self, input_dim: int = 768, projection_dim: int = 96) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(input_dim)
        self.project = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(projection_dim * 6 + 2),
            nn.Linear(projection_dim * 6 + 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.better_logit = nn.Linear(128, 1)
        self.positive_gain = nn.Sequential(nn.Linear(128, 1), nn.Softplus())

    def forward(
        self,
        instruction: torch.Tensor,
        current_history: torch.Tensor,
        temporal_history: torch.Tensor,
        native: torch.Tensor,
        alternative: torch.Tensor,
        immediate_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = (
            instruction,
            current_history,
            temporal_history,
            native,
            alternative,
            alternative - native,
        )
        encoded = [self.project(self.normalizer(value)) for value in raw]
        fused = self.fusion(torch.cat([*encoded, immediate_costs], dim=-1))
        return self.better_logit(fused).squeeze(-1), self.positive_gain(fused).squeeze(-1)


class OnlineNetAdvantageScorer:
    """Rank causal alternatives and expose a deployable sparse veto threshold."""

    def __init__(
        self, models: list[PairwiseNetAdvantageHead], device: torch.device,
        gain_scale_m: float, threshold: float | None,
        checkpoint_seeds: tuple[int, ...],
    ) -> None:
        self.models = [model.eval() for model in models]
        self.device = device
        self.gain_scale_m = gain_scale_m
        self.threshold = threshold
        self.checkpoint_seeds = checkpoint_seeds

    @classmethod
    def from_checkpoint(
        cls, path: Path, device: str | torch.device = "cpu",
        require_online_threshold: bool = True,
        expected_member_seeds: tuple[int, ...] | None = None,
    ) -> "OnlineNetAdvantageScorer":
        device = torch.device(device)
        payload = torch.load(path, map_location=device, weights_only=False)
        schema = payload.get("schema_version")
        if schema not in (
            "revealnav-pairwise-net-advantage-checkpoint/1",
            "revealnav-pairwise-net-advantage-ensemble/1",
        ):
            raise RuntimeError("unsupported net-advantage checkpoint schema")
        if payload.get("input_dim") != 768 or payload.get("projection_dim") != 96:
            raise RuntimeError("net-advantage checkpoint architecture drift")
        if schema.endswith("ensemble/1"):
            checkpoint_seeds = tuple(payload.get("member_seeds", ()))
            states = payload.get("model_state_dicts", ())
            if (
                checkpoint_seeds != (20260826, 20260827, 20260828)
                or len(states) != len(checkpoint_seeds)
                or payload.get("aggregation")
                != "mean_probability_and_mean_positive_gain"
            ):
                raise RuntimeError("net-advantage ensemble contract drift")
        else:
            checkpoint_seeds = (payload.get("seed"),)
            states = (payload["model_state_dict"],)
        if (
            expected_member_seeds is not None
            and checkpoint_seeds != expected_member_seeds
        ):
            raise RuntimeError("net-advantage checkpoint seed mismatch")
        models = []
        for state in states:
            model = PairwiseNetAdvantageHead(
                payload["input_dim"], payload["projection_dim"]
            ).to(device)
            model.load_state_dict(state, strict=True)
            models.append(model)
        online = payload.get("score_definition") == ONLINE_SCORE_DEFINITION
        if require_online_threshold and not online:
            raise RuntimeError(
                "checkpoint threshold used an offline-only penalty and is not deployable"
            )
        threshold = (
            float(payload["calibrated_score_threshold"]) if online else None
        )
        return cls(
            models, device, float(payload["immediate_cost_scale_m"]), threshold,
            checkpoint_seeds,
        )

    def score_candidates(
        self, instruction: torch.Tensor, current_history: torch.Tensor,
        temporal_history: torch.Tensor, native: torch.Tensor,
        alternatives: dict[str, torch.Tensor], native_distance_m: float,
        alternative_distances_m: dict[str, float],
    ) -> list[dict]:
        branch_ids = sorted(alternatives)
        if not branch_ids or set(branch_ids) != set(alternative_distances_m):
            raise ValueError("candidate embeddings and causal distances must align")
        values = (instruction, current_history, temporal_history, native)
        if any(tuple(value.shape) != (768,) for value in values):
            raise ValueError("online net-advantage embeddings must each have shape [768]")
        alternative = torch.stack([alternatives[key] for key in branch_ids])
        size = len(branch_ids)
        immediate = torch.tensor([
            [native_distance_m, alternative_distances_m[key]]
            for key in branch_ids
        ], dtype=torch.float32, device=self.device) / self.gain_scale_m
        repeated = [value.to(self.device).float().unsqueeze(0).expand(size, -1) for value in values]
        probabilities = []
        gains = []
        with torch.no_grad():
            for model in self.models:
                logit, scaled_gain = model(
                    repeated[0], repeated[1], repeated[2], repeated[3],
                    alternative.to(self.device).float(), immediate,
                )
                probabilities.append(torch.sigmoid(logit))
                gains.append(scaled_gain * self.gain_scale_m)
        probability = torch.stack(probabilities).mean(0)
        gain_m = torch.stack(gains).mean(0)
        penalty_m = torch.tensor([
            2.0 * alternative_distances_m[key] for key in branch_ids
        ], dtype=torch.float32, device=self.device)
        scores = probability * gain_m - (1.0 - probability) * penalty_m
        return [{
            "branch_id": key,
            "p_better": float(probability[index].cpu()),
            "positive_gain_m": float(gain_m[index].cpu()),
            "online_wrong_trial_cost_m": float(penalty_m[index].cpu()),
            "net_advantage_score_m": float(scores[index].cpu()),
        } for index, key in enumerate(branch_ids)]

    def approve(self, proposed_branch: str, rows: list[dict]) -> bool:
        if self.threshold is None:
            raise RuntimeError("checkpoint has no deployable online threshold")
        best = max(rows, key=lambda row: (
            row["net_advantage_score_m"], row["branch_id"]
        ))
        return (
            best["branch_id"] == proposed_branch
            and best["net_advantage_score_m"] > self.threshold
        )
