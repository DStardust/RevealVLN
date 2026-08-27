"""Consecutive-prefix persistence for evolving candidate identities."""

from __future__ import annotations

from collections.abc import Iterable


class ConsecutiveCandidateTracker:
    """Return candidates present in at least K consecutive prefixes."""

    def __init__(self, persistence_k: int = 3) -> None:
        if persistence_k < 1:
            raise ValueError("persistence_k must be positive")
        self.persistence_k = persistence_k
        self.previous: set[str] = set()
        self.streaks: dict[str, int] = {}

    def update(self, candidate_ids: Iterable[str]) -> tuple[str, ...]:
        current = set(candidate_ids)
        if any(not isinstance(value, str) or not value for value in current):
            raise ValueError("candidate ids must be non-empty strings")
        self.streaks = {
            candidate_id: (
                self.streaks.get(candidate_id, 0) + 1
                if candidate_id in self.previous else 1
            )
            for candidate_id in current
        }
        self.previous = current
        return tuple(sorted(
            candidate_id for candidate_id in current
            if self.streaks[candidate_id] >= self.persistence_k
        ))
