"""Control-backed returnability contract for MF3ZR.

This module deliberately does not provide a geometry-only or teleport
fallback.  A caller must inject a real frozen ETP-R1/controller callback.  In
the current 80-event population no such callback is sealed, so the audit
records ``EXECUTION_UNAVAILABLE`` for every candidate and stops downstream
support computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from .option_binding_schema import is_sha256


RETURN_HORIZON = 8


class ReturnabilityStatus(str, Enum):
    RETURNABLE = "RETURNABLE"
    NOT_RETURNABLE = "NOT_RETURNABLE"
    EXECUTION_UNAVAILABLE = "EXECUTION_UNAVAILABLE"
    INVALID_ANCHOR = "INVALID_ANCHOR"


@dataclass(frozen=True)
class ReturnabilityResult:
    event_id: str
    option_id: str
    from_step: int
    anchor_checkpoint_id: str
    status: ReturnabilityStatus | str
    attempted: bool
    success: bool
    high_level_steps: int | None
    low_level_steps: int | None
    controller_sha256: str | None
    reason: str

    def __post_init__(self) -> None:
        for value, name in ((self.event_id, "event_id"), (self.option_id, "option_id"), (self.anchor_checkpoint_id, "anchor_checkpoint_id"), (self.reason, "reason")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.from_step, bool) or not isinstance(self.from_step, int) or self.from_step < 0:
            raise ValueError("from_step must be a non-negative integer")
        status = self.status if isinstance(self.status, ReturnabilityStatus) else ReturnabilityStatus(self.status)
        if type(self.attempted) is not bool or type(self.success) is not bool:
            raise TypeError("attempted/success must be Boolean")
        if self.success and not self.attempted:
            raise ValueError("an unattempted return cannot succeed")
        if self.high_level_steps is not None and (isinstance(self.high_level_steps, bool) or not isinstance(self.high_level_steps, int) or self.high_level_steps < 0 or self.high_level_steps > RETURN_HORIZON):
            raise ValueError("high_level_steps exceeds the fixed return horizon")
        if self.low_level_steps is not None and (isinstance(self.low_level_steps, bool) or not isinstance(self.low_level_steps, int) or self.low_level_steps < 0):
            raise ValueError("low_level_steps must be non-negative")
        if self.controller_sha256 is not None and not is_sha256(self.controller_sha256):
            raise ValueError("controller_sha256 must be a lowercase SHA-256")
        if status is ReturnabilityStatus.RETURNABLE and not self.success:
            raise ValueError("RETURNABLE requires a successful callback")
        if status is ReturnabilityStatus.EXECUTION_UNAVAILABLE and (self.attempted or self.success):
            raise ValueError("unavailable returnability cannot be attempted")
        object.__setattr__(self, "status", status)

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": "revealnav-mf3zr-returnability/1",
            "event_id": self.event_id,
            "option_id": self.option_id,
            "from_step": self.from_step,
            "anchor_checkpoint_id": self.anchor_checkpoint_id,
            "status": self.status.value,
            "attempted": self.attempted,
            "success": self.success,
            "high_level_steps": self.high_level_steps,
            "low_level_steps": self.low_level_steps,
            "controller_sha256": self.controller_sha256,
            "reason": self.reason,
        }


ReturnabilityCallback = Callable[[Mapping[str, object], Mapping[str, object], int], Mapping[str, object]]


@dataclass(frozen=True)
class FrozenReturnabilityAdapter:
    """A strict adapter around a real frozen-controller callback."""

    callback: ReturnabilityCallback | None = None
    controller_sha256: str | None = None
    horizon: int = RETURN_HORIZON

    def __post_init__(self) -> None:
        if self.horizon != RETURN_HORIZON:
            raise ValueError("MF3ZR return horizon is fixed at 8")
        if self.callback is not None and not callable(self.callback):
            raise TypeError("returnability callback must be callable")
        if self.callback is not None and not is_sha256(self.controller_sha256):
            raise ValueError("a callback requires a sealed controller SHA-256")

    @property
    def available(self) -> bool:
        return self.callback is not None

    def audit(
        self,
        *,
        event_id: str,
        option_id: str,
        from_step: int,
        anchor_checkpoint_id: str,
        state: Mapping[str, object],
        option: Mapping[str, object],
    ) -> ReturnabilityResult:
        if from_step < 0:
            raise ValueError("from_step must be non-negative")
        if not self.available:
            return ReturnabilityResult(
                event_id=event_id,
                option_id=option_id,
                from_step=from_step,
                anchor_checkpoint_id=anchor_checkpoint_id,
                status=ReturnabilityStatus.EXECUTION_UNAVAILABLE,
                attempted=False,
                success=False,
                high_level_steps=None,
                low_level_steps=None,
                controller_sha256=None,
                reason="NO_SEALED_FROZEN_CONTROLLER_CALLBACK_FOR_MF3ZR",
            )
        # The injected callback owns execution.  It must return a small,
        # outcome-free control witness; metrics/rewards are rejected here.
        raw = self.callback(state, option, self.horizon)
        if not isinstance(raw, Mapping):
            raise ValueError("returnability callback must return a mapping")
        forbidden = {"reward", "utility", "delta_utility", "success_metric", "route_truth", "pose", "teleport", "navmesh_shortest_path"}
        if any(str(key).casefold() in forbidden for key in raw):
            raise ValueError("returnability callback returned forbidden outcome/control shortcut")
        status = ReturnabilityStatus(raw.get("status", ""))
        attempted = raw.get("attempted")
        success = raw.get("success")
        steps = raw.get("high_level_steps")
        low = raw.get("low_level_steps")
        return ReturnabilityResult(
            event_id=event_id,
            option_id=option_id,
            from_step=from_step,
            anchor_checkpoint_id=anchor_checkpoint_id,
            status=status,
            attempted=attempted,
            success=success,
            high_level_steps=steps,
            low_level_steps=low,
            controller_sha256=self.controller_sha256,
            reason=str(raw.get("reason", "callback")),
        )


def unavailable_adapter() -> FrozenReturnabilityAdapter:
    """Return the explicit no-callback adapter used by the current audit."""

    return FrozenReturnabilityAdapter(callback=None, controller_sha256=None, horizon=RETURN_HORIZON)


def reject_snapshot_as_skill(value: Mapping[str, object]) -> None:
    """Snapshots may initialise an audit, never count as a return action."""

    if value.get("used_as_return") is True or value.get("teleport") is True or value.get("pose_reset") is True:
        raise ValueError("simulator snapshot/restore cannot count as returnability")


__all__ = [
    "RETURN_HORIZON", "ReturnabilityStatus", "ReturnabilityResult",
    "FrozenReturnabilityAdapter", "ReturnabilityCallback", "unavailable_adapter",
    "reject_snapshot_as_skill",
]
