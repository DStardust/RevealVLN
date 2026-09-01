"""Bounded, outcome-blind option memory for MF3ZQ."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .oracle_revealskill_schema import (
    OPTION_MEMORY_BUDGET,
    OracleOption,
    OracleOptionStatus,
)


class OracleOptionMemory:
    """Mutable option store with a fixed eight-option capacity.

    Eviction is solely based on executable state and sealed observation order;
    no reward, metric, or correctness label is consulted.
    """

    def __init__(self, options: Iterable[OracleOption] = (), *, budget: int = OPTION_MEMORY_BUDGET) -> None:
        if budget != OPTION_MEMORY_BUDGET:
            raise ValueError("MF3ZQ option memory budget is fixed at 8")
        self.budget = budget
        self._options: dict[str, OracleOption] = {}
        for option in options:
            self.observe(option)

    def observe(self, option: OracleOption) -> tuple[str, ...]:
        if not isinstance(option, OracleOption):
            raise TypeError("option memory accepts OracleOption")
        prior = self._options.get(option.option_id)
        if prior is not None:
            if prior.branch_candidate_id != option.branch_candidate_id:
                raise ValueError("option identity/candidate conflict")
            # Keep the earliest sealed observation and newest causal state.
            option = replace(
                option,
                first_seen_step=min(prior.first_seen_step, option.first_seen_step),
                last_seen_step=max(prior.last_seen_step, option.last_seen_step),
            )
        self._options[option.option_id] = option
        return self._enforce_budget()

    def get(self, option_id: str) -> OracleOption:
        try:
            return self._options[str(option_id)]
        except KeyError as error:
            raise KeyError(f"unknown option: {option_id}") from error

    def options(self) -> tuple[OracleOption, ...]:
        return tuple(self._options[key] for key in sorted(self._options))

    def active(self) -> tuple[OracleOption, ...]:
        terminal = {OracleOptionStatus.EXHAUSTED, OracleOptionStatus.COMMITTED, OracleOptionStatus.INVALID}
        return tuple(option for option in self.options() if option.status not in terminal)

    def preserve(self, option_id: str) -> None:
        option = self.get(option_id)
        if option.status in (OracleOptionStatus.EXHAUSTED, OracleOptionStatus.COMMITTED, OracleOptionStatus.INVALID):
            raise ValueError("terminal option cannot be preserved")
        self._options[option.option_id] = replace(option, status=OracleOptionStatus.PRESERVED)

    def mark_active(self, option_id: str) -> None:
        option = self.get(option_id)
        if option.status is OracleOptionStatus.UNTRIED:
            self._options[option.option_id] = replace(option, status=OracleOptionStatus.ACTIVE)

    def commit(self, option_id: str) -> None:
        option = self.get(option_id)
        if option.readiness.value != "D":
            raise ValueError("commit requires D readiness")
        self._options[option.option_id] = replace(option, status=OracleOptionStatus.COMMITTED)

    def exhaust(self, option_id: str) -> None:
        option = self.get(option_id)
        if option.status is OracleOptionStatus.COMMITTED:
            raise ValueError("committed option cannot be exhausted")
        self._options[option.option_id] = replace(option, status=OracleOptionStatus.EXHAUSTED)

    def max_usage(self) -> int:
        return len(self._options)

    def _eviction_key(self, option: OracleOption) -> tuple[int, int, int, str]:
        # Returnable first, then earliest expiry, first seen, stable ID.  A
        # missing expiry is sorted after a known expiry.
        expiry = option.expiry_step if option.expiry_step is not None else 2**31 - 1
        return (0 if option.returnable else 1, expiry, option.first_seen_step, option.option_id)

    def _enforce_budget(self) -> tuple[str, ...]:
        if len(self._options) <= self.budget:
            return ()
        ordered = sorted(self._options.values(), key=self._eviction_key)
        keep = {item.option_id for item in ordered[: self.budget]}
        evicted = tuple(sorted(set(self._options) - keep))
        for option_id in evicted:
            del self._options[option_id]
        return evicted

    def snapshot(self) -> tuple[dict[str, object], ...]:
        return tuple({
            "option_id": option.option_id,
            "branch_candidate_id": option.branch_candidate_id,
            "anchor_checkpoint_id": option.anchor_checkpoint_id,
            "frozen_candidate_rank": option.frozen_candidate_rank,
            "first_seen_step": option.first_seen_step,
            "last_seen_step": option.last_seen_step,
            "prerequisite_ids": list(option.prerequisite_ids),
            "decisive_constraint_ids": list(option.decisive_constraint_ids),
            "readiness": option.readiness.value,
            "reveal_step": option.reveal_step,
            "expiry_step": option.expiry_step,
            "returnable": option.returnable,
            "status": option.status.value,
        } for option in self.options())


__all__ = ["OracleOptionMemory"]
