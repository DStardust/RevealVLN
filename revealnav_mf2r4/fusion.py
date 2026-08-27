"""Fixed composition of REE branch identity and V4 macro-action costs."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

from .controller import BranchExcursionMacroController, BranchMacroDecision


class ReeQFusionController:
    def __init__(
        self, persistence_k: int = 3, wrong_commitment_weight: float = 5.0,
    ) -> None:
        if not isfinite(wrong_commitment_weight) or wrong_commitment_weight < 0.0:
            raise ValueError("wrong_commitment_weight must be finite and non-negative")
        self.macro = BranchExcursionMacroController(persistence_k)
        self.wrong_commitment_weight = float(wrong_commitment_weight)

    @property
    def persistence_k(self) -> int:
        return self.macro.persistence_k

    def decide(
        self,
        branch_ids: Sequence[str],
        target_probabilities: Sequence[float],
        commit_costs: Sequence[float],
        excursion_costs: Sequence[float],
        stable_observations: int,
    ) -> BranchMacroDecision:
        if not (
            len(branch_ids) == len(target_probabilities)
            == len(commit_costs) == len(excursion_costs)
        ):
            raise ValueError("branch probabilities and costs must align")
        penalties = []
        for probability in target_probabilities:
            if not isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("target probabilities must be finite and in [0, 1]")
            penalties.append(self.wrong_commitment_weight * (1.0 - probability))
        return self.macro.decide(
            branch_ids,
            [cost + penalty for cost, penalty in zip(commit_costs, penalties)],
            [cost + penalty for cost, penalty in zip(excursion_costs, penalties)],
            stable_observations,
        )
