"""Small topological memory for checkpoints and recoverable branch options."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import inf
from typing import Callable, Iterable, Optional

from .types import BranchCandidate, BranchStatus


@dataclass
class Checkpoint:
    checkpoint_id: str
    branches: dict[str, BranchCandidate]
    neighbors: dict[str, float] = field(default_factory=dict)


class TopologicalMemory:
    """Stores only decision checkpoints, not every robot pose.

    Localization and low-level path execution remain responsibilities of the
    surrounding navigation system.  Edges here are traversable return routes
    between checkpoints.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    def __contains__(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self._checkpoints

    def __len__(self) -> int:
        return len(self._checkpoints)

    def checkpoint(self, checkpoint_id: str) -> Checkpoint:
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as error:
            raise KeyError(f"unknown checkpoint: {checkpoint_id}") from error

    def add_checkpoint(
        self,
        checkpoint_id: str,
        branches: Iterable[BranchCandidate],
        connect_to: Optional[str] = None,
        travel_cost: float = 0.0,
    ) -> None:
        if not checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if checkpoint_id in self:
            raise ValueError(f"checkpoint already exists: {checkpoint_id}")
        if travel_cost < 0.0:
            raise ValueError("travel_cost must be non-negative")
        branch_rows = list(branches)
        branch_map = {branch.branch_id: branch for branch in branch_rows}
        if not branch_map:
            raise ValueError("a checkpoint must retain at least one branch")
        if len(branch_map) != len(branch_rows):
            raise ValueError("branch ids must be unique within a checkpoint")
        if connect_to is not None and connect_to not in self:
            raise KeyError(f"unknown checkpoint: {connect_to}")

        self._checkpoints[checkpoint_id] = Checkpoint(checkpoint_id, branch_map)
        if connect_to is not None:
            self.connect(checkpoint_id, connect_to, travel_cost)

    def connect(self, first: str, second: str, travel_cost: float) -> None:
        if travel_cost < 0.0:
            raise ValueError("travel_cost must be non-negative")
        first_node = self.checkpoint(first)
        second_node = self.checkpoint(second)
        first_node.neighbors[second] = travel_cost
        second_node.neighbors[first] = travel_cost

    def set_branch_status(
        self, checkpoint_id: str, branch_id: str, status: BranchStatus
    ) -> None:
        branch = self._branch(checkpoint_id, branch_id)
        branch.status = status
        if status is BranchStatus.ACTIVE:
            branch.visits += 1

    def pending_branches(self) -> Iterable[tuple[str, BranchCandidate]]:
        for checkpoint_id, checkpoint in self._checkpoints.items():
            for branch in checkpoint.branches.values():
                if branch.status is BranchStatus.UNTRIED:
                    yield checkpoint_id, branch

    def best_pending_branch(
        self,
        score: Callable[[BranchCandidate], float],
        exclude_checkpoint: Optional[str] = None,
    ) -> Optional[tuple[str, BranchCandidate, float]]:
        choices = (
            (score(branch), checkpoint_id, branch)
            for checkpoint_id, branch in self.pending_branches()
            if checkpoint_id != exclude_checkpoint
        )
        try:
            utility, checkpoint_id, branch = max(
                choices, key=lambda item: (item[0], item[1], item[2].branch_id)
            )
        except ValueError:
            return None
        return checkpoint_id, branch, utility

    def shortest_path(self, start: str, goal: str) -> tuple[str, ...]:
        self.checkpoint(start)
        self.checkpoint(goal)
        distances = {start: 0.0}
        previous: dict[str, str] = {}
        queue = [(0.0, start)]

        while queue:
            distance, checkpoint_id = heappop(queue)
            if checkpoint_id == goal:
                break
            if distance != distances[checkpoint_id]:
                continue
            for neighbor, edge_cost in self.checkpoint(checkpoint_id).neighbors.items():
                candidate_distance = distance + edge_cost
                if candidate_distance < distances.get(neighbor, inf):
                    distances[neighbor] = candidate_distance
                    previous[neighbor] = checkpoint_id
                    heappush(queue, (candidate_distance, neighbor))

        if goal not in distances:
            raise ValueError(f"no return path from {start} to {goal}")
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        return tuple(reversed(path))

    def _branch(self, checkpoint_id: str, branch_id: str) -> BranchCandidate:
        checkpoint = self.checkpoint(checkpoint_id)
        try:
            return checkpoint.branches[branch_id]
        except KeyError as error:
            raise KeyError(
                f"unknown branch {branch_id!r} at checkpoint {checkpoint_id!r}"
            ) from error
