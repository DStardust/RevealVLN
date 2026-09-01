"""Deterministic cognitive-only Oracle RevealSkill policies.

This module never chooses an action from an outcome oracle.  It only maps a
sealed cognitive state to one of the six fixed high-level skills.  A caller
must provide the frozen ETP executor for movement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping

from .oracle_option_memory import OracleOptionMemory
from .oracle_revealskill_schema import (
    OracleOption,
    OracleOptionStatus,
    OraclePolicyState,
    OracleReadiness,
    OracleSkill,
    OPTION_MEMORY_BUDGET,
    reject_forbidden_oracle_mapping,
)


class OracleArm(str, Enum):
    BASELINE = "A_BASELINE"
    DEC = "B_ORACLE_DEC"
    DEC_OPTION = "C_DEC_OPTION_MEMORY"
    FULL = "D_FULL_REVEALSKILL"


@dataclass(frozen=True)
class SkillDecision:
    skill: OracleSkill
    option_id: str | None
    reason: str
    memory_usage: int


def deterministic_option_key(option: OracleOption) -> tuple[int, int, str]:
    """Fixed ordering: frozen rank, first observation, stable ID."""

    return (int(option.frozen_candidate_rank), int(option.first_seen_step), option.option_id)


def ordered_options(options: Iterable[OracleOption]) -> tuple[OracleOption, ...]:
    return tuple(sorted(options, key=deterministic_option_key))


def _option_by_id(options: Iterable[OracleOption]) -> dict[str, OracleOption]:
    result = {option.option_id: option for option in options}
    if len(result) != len(tuple(options)):
        raise ValueError("duplicate option identity")
    return result


class OracleRevealSkillPolicy:
    """The four pre-registered arms share one deterministic policy skeleton."""

    def __init__(self, arm: OracleArm | str, *, memory_budget: int = OPTION_MEMORY_BUDGET) -> None:
        if memory_budget != OPTION_MEMORY_BUDGET:
            raise ValueError("MF3ZQ memory budget is fixed at 8")
        self.arm = arm if isinstance(arm, OracleArm) else OracleArm(arm)
        self.memory = OracleOptionMemory(budget=memory_budget)
        self.decisions: list[SkillDecision] = []
        self.max_memory_usage = 0
        self.return_attempts = 0
        self.return_successes = 0

    def observe_options(self, options: Iterable[OracleOption]) -> tuple[str, ...]:
        evicted: list[str] = []
        for option in options:
            evicted.extend(self.memory.observe(option))
        self.max_memory_usage = max(self.max_memory_usage, self.memory.max_usage())
        return tuple(evicted)

    def _choose(self, skill: OracleSkill, option_id: str | None, reason: str) -> SkillDecision:
        decision = SkillDecision(skill, option_id, reason, self.memory.max_usage())
        self.decisions.append(decision)
        return decision

    def step(self, state: OraclePolicyState) -> SkillDecision:
        """Return a skill without touching a simulator or reading outcomes."""

        reject_forbidden_oracle_mapping({
            "step": state.step,
            "checkpoint_id": state.checkpoint_id,
            "executable_option_ids": state.executable_option_ids,
            "reveal_after_expiry": state.reveal_after_expiry,
        })
        self.observe_options(state.options)
        by_id = {option.option_id: option for option in self.memory.options()}
        executable = [by_id[item] for item in state.executable_option_ids if item in by_id]
        executable = list(ordered_options(executable))

        if self.arm is OracleArm.BASELINE:
            return self._choose(OracleSkill.FOLLOW, None, "frozen_native_baseline")

        decisive = [option for option in executable if option.readiness is OracleReadiness.D]
        if decisive:
            chosen = decisive[0]
            if chosen.status is not OracleOptionStatus.COMMITTED:
                self.memory.commit(chosen.option_id)
            return self._choose(OracleSkill.COMMIT, chosen.option_id, "decisive_option")

        if self.arm in (OracleArm.DEC_OPTION, OracleArm.FULL):
            if self.arm is OracleArm.FULL:
                expiring = [
                    option for option in executable
                    if option.readiness in (OracleReadiness.U, OracleReadiness.A)
                    and option.returnable
                    and option.expiry_step is not None
                    and option.expiry_step - state.step <= 1
                ]
                if expiring:
                    chosen = sorted(expiring, key=lambda option: (
                        option.expiry_step if option.expiry_step is not None else 2**31 - 1,
                        *deterministic_option_key(option),
                    ))[0]
                    self.memory.preserve(chosen.option_id)
                    self.return_attempts += 1
                    self.return_successes += int(chosen.returnable)
                    return self._choose(OracleSkill.BACKTRACK, chosen.option_id, "expiry_aware_preservation")
            unresolved = [option for option in executable if option.readiness in (OracleReadiness.U, OracleReadiness.A)]
            if unresolved and state.unresolved_decisive_evidence:
                chosen = unresolved[0]
                self.memory.preserve(chosen.option_id)
                return self._choose(OracleSkill.INSPECT, None, "preserve_unresolved_competing_option")

        if state.unresolved_decisive_evidence:
            return self._choose(OracleSkill.INSPECT, None, "inspect_current_decisive_evidence")
        if executable:
            chosen = executable[0]
            self.memory.mark_active(chosen.option_id)
            return self._choose(OracleSkill.EXPLORE, chosen.option_id, "fixed_rank_untried_option")
        return self._choose(OracleSkill.FOLLOW, None, "no_legal_option")


def execute_skill_with_frozen_controller(
    decision: SkillDecision,
    *,
    frozen_executor,
):
    """Movement boundary used by integration workers.

    The executor is intentionally injected; this function never accepts a
    pose, waypoint, navmesh path, or direct simulator mutation.
    """

    if not callable(frozen_executor):
        raise TypeError("frozen_executor must be callable")
    return frozen_executor(decision.skill, decision.option_id)


__all__ = [
    "OracleArm", "SkillDecision", "OracleRevealSkillPolicy",
    "deterministic_option_key", "ordered_options",
    "execute_skill_with_frozen_controller",
]
