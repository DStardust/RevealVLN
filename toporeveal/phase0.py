"""Auditable go/no-go decision for the frozen Phase 0 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt


@dataclass(frozen=True)
class Phase0Evidence:
    project_self_contained: bool
    mp3d_scene_count: int
    mp3d_access_authorized: bool
    official_metadata_verified: bool
    habitat_ready: bool
    waypoint_frontend_reproduced: bool
    etpr1_reproduced: bool
    screened_instructions: int
    candidate_trajectories: int
    reviewed_candidates: int
    valid_candidates: int
    validated_events: int
    unique_expiry_events: int

    def __post_init__(self) -> None:
        for name in (
            "project_self_contained",
            "mp3d_access_authorized",
            "official_metadata_verified",
            "habitat_ready",
            "waypoint_frontend_reproduced",
            "etpr1_reproduced",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in (
            "mp3d_scene_count",
            "screened_instructions",
            "candidate_trajectories",
            "reviewed_candidates",
            "valid_candidates",
            "validated_events",
            "unique_expiry_events",
        ):
            if not isinstance(getattr(self, name), int) or isinstance(
                getattr(self, name), bool
            ):
                raise TypeError(f"{name} must be an integer")
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.reviewed_candidates > self.screened_instructions:
            raise ValueError("reviewed_candidates cannot exceed screened_instructions")
        if self.candidate_trajectories > self.screened_instructions:
            raise ValueError(
                "candidate_trajectories cannot exceed screened_instructions"
            )
        if self.valid_candidates > self.reviewed_candidates:
            raise ValueError("valid_candidates cannot exceed reviewed_candidates")
        if self.reviewed_candidates > self.candidate_trajectories:
            raise ValueError(
                "reviewed_candidates cannot exceed candidate_trajectories"
            )
        if self.unique_expiry_events > self.validated_events:
            raise ValueError("unique_expiry_events cannot exceed validated_events")

    @property
    def valid_rate(self) -> float:
        if self.reviewed_candidates == 0:
            return 0.0
        return self.valid_candidates / self.reviewed_candidates

    @property
    def estimated_valid_events(self) -> int:
        """One-event-per-trajectory projection using a 95% Wilson lower bound."""

        return floor(self.candidate_trajectories * self.projected_valid_rate)

    @property
    def projected_valid_rate(self) -> float:
        if self.reviewed_candidates == 0:
            return 0.0
        z = 1.959963984540054
        count = self.reviewed_candidates
        rate = self.valid_rate
        denominator = 1.0 + z * z / count
        center = rate + z * z / (2.0 * count)
        radius = z * sqrt(
            rate * (1.0 - rate) / count + z * z / (4.0 * count * count)
        )
        return max(0.0, (center - radius) / denominator)

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.project_self_contained:
            blockers.append("runtime dependencies are not self-contained under the project")
        if self.mp3d_scene_count != 90:
            blockers.append("project-local MP3D must contain exactly 90 scenes")
        if not self.mp3d_access_authorized:
            blockers.append("MP3D access provenance is not authorized for this project")
        if not self.official_metadata_verified:
            blockers.append("official R2R-CE/RxR-CE metadata is not verified")
        if not self.habitat_ready:
            blockers.append("Habitat/VLN-CE is not runnable")
        if not self.waypoint_frontend_reproduced:
            blockers.append("the frozen waypoint frontend is not reproduced")
        if not self.etpr1_reproduced:
            blockers.append("the frozen ETP-R1 checkpoint is not reproduced")
        if self.reviewed_candidates < 50:
            blockers.append("fewer than 50 screened candidates were manually reviewed")
        if self.valid_rate < 0.25:
            blockers.append("manual valid-event rate is below 25%")
        if self.estimated_valid_events < 300:
            blockers.append("fewer than 300 valid Reveal Events are estimated")
        if self.validated_events == 0:
            blockers.append("no Reveal Event has passed full artifact validation")
        if self.validated_events < self.valid_candidates:
            blockers.append(
                "not every valid candidate produced a validated Reveal Event"
            )
        if self.unique_expiry_events != self.validated_events:
            blockers.append("not every valid event has a unique reproducible expiry")
        return tuple(blockers)

    @property
    def go(self) -> bool:
        return not self.blockers
