"""Fail-closed execution state for checkpointed exploration and return."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from revealnav_mf2r3 import OptionStatus


class ExecutorPhase(str, Enum):
    AT_CHECKPOINT = "at_checkpoint"
    EXPLORING = "exploring"
    RETURNING = "returning"
    RETURN_FAILED = "return_failed"
    COMMITTED = "committed"


@dataclass(frozen=True)
class ReturnCommand:
    checkpoint_id: str
    controller_ref: str
    branch_id: str


class CheckpointReturnExecutor:
    """Execute a policy decision without deciding when to return."""

    def __init__(
        self, checkpoint_id: str, controller_ref: str,
        branch_ids: tuple[str, ...],
    ) -> None:
        if not checkpoint_id or not controller_ref:
            raise ValueError("checkpoint and controller references must be non-empty")
        if len(branch_ids) < 2 or len(set(branch_ids)) != len(branch_ids):
            raise ValueError("a checkpoint needs at least two unique branches")
        if any(not branch_id for branch_id in branch_ids):
            raise ValueError("branch ids must be non-empty")
        self.checkpoint_id = checkpoint_id
        self.controller_ref = controller_ref
        self.branch_status = {
            branch_id: OptionStatus.UNTRIED for branch_id in branch_ids
        }
        self.phase = ExecutorPhase.AT_CHECKPOINT
        self.active_branch: str | None = None

    def start_excursion(self, branch_id: str) -> None:
        if self.phase is not ExecutorPhase.AT_CHECKPOINT:
            raise RuntimeError("excursion may start only at the checkpoint")
        if self.branch_status.get(branch_id) is not OptionStatus.UNTRIED:
            raise ValueError("excursion branch must be known and untried")
        self.branch_status[branch_id] = OptionStatus.ACTIVE
        self.active_branch = branch_id
        self.phase = ExecutorPhase.EXPLORING

    def request_backtrack(self) -> ReturnCommand:
        if self.phase is not ExecutorPhase.EXPLORING or self.active_branch is None:
            raise RuntimeError("backtrack requires an active excursion")
        self.phase = ExecutorPhase.RETURNING
        return self._return_command()

    def report_return(self, succeeded: bool) -> None:
        if self.phase is not ExecutorPhase.RETURNING or self.active_branch is None:
            raise RuntimeError("return result requires an in-flight return")
        if not isinstance(succeeded, bool):
            raise TypeError("return result must be boolean")
        if succeeded:
            self.branch_status[self.active_branch] = OptionStatus.EXHAUSTED
            self.active_branch = None
            self.phase = ExecutorPhase.AT_CHECKPOINT
        else:
            self.phase = ExecutorPhase.RETURN_FAILED

    def retry_return(self) -> ReturnCommand:
        if self.phase is not ExecutorPhase.RETURN_FAILED or self.active_branch is None:
            raise RuntimeError("retry requires a failed return")
        self.phase = ExecutorPhase.RETURNING
        return self._return_command()

    def commit(self, branch_id: str) -> None:
        if self.phase is not ExecutorPhase.AT_CHECKPOINT:
            raise RuntimeError("commit may occur only at the checkpoint")
        if self.branch_status.get(branch_id) is not OptionStatus.UNTRIED:
            raise ValueError("commit branch must be known and untried")
        self.branch_status[branch_id] = OptionStatus.COMMITTED
        self.active_branch = branch_id
        self.phase = ExecutorPhase.COMMITTED

    def _return_command(self) -> ReturnCommand:
        if self.active_branch is None:
            raise RuntimeError("return command lacks an active branch")
        return ReturnCommand(
            self.checkpoint_id, self.controller_ref, self.active_branch
        )
