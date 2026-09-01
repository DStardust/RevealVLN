"""Minimal evidence-preserving high-level policy shell.

The shell delegates movement to a caller-provided frozen ETP executor.  It
does not implement a second simulator or permit teleportation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .evidence_constraints import InstructionEvidenceGraph
from .evidence_memory import EvidenceMemory
from .evidence_uad import option_readiness
from .option_graph import OptionGraph
from .revealskill_q import legal_skill_actions
from .revealskill_schema import Readiness, RevealSkillAction, RevealSkillState, reject_forbidden_state_mapping


def revealskill_step(
    instruction_graph: InstructionEvidenceGraph,
    etp_state: Mapping[str, object],
    evidence_memory: EvidenceMemory,
    option_graph: OptionGraph,
    *,
    constraint_states: Mapping[str, object],
    q_values: Mapping[tuple[RevealSkillAction, str | None], float],
    execute: Callable[[RevealSkillAction, str | None], object],
) -> tuple[RevealSkillAction, str | None, object]:
    reject_forbidden_state_mapping(etp_state)
    candidates = tuple(str(value) for value in etp_state.get("executable_candidates", ()))
    readiness = {
        node.branch_candidate_id: option_readiness(instruction_graph, node.branch_candidate_id, constraint_states)
        for node in option_graph.active_options()
        if node.branch_candidate_id in candidates
    }
    legal = legal_skill_actions(readiness, executable_options=candidates, returnable_options=())
    if bool(etp_state.get("reveal_after_expiry", False)):
        legal = tuple(action for action in legal if action[0] not in (RevealSkillAction.COMMIT, RevealSkillAction.EXPLORE))
    if not legal:
        raise RuntimeError("no legal RevealSkill action")
    chosen = min(legal, key=lambda action: float(q_values.get(action, 0.0)))
    # The executor is the only movement boundary; this function never edits
    # simulator poses or creates an alternative action trajectory itself.
    return chosen[0], chosen[1], execute(*chosen)


__all__ = ["revealskill_step"]
