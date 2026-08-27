"""Load a Phase 0 snapshot whose positive claims cite hashed local artifacts."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from .phase0 import Phase0Evidence
from .provenance import (
    CANONICAL_PHASE0_MANIFEST_SHA256,
    canonical_phase0_asset,
    regular_project_file,
    sha256_file,
)
from .screening import (
    iter_vlnce_episodes,
    pilot_sample,
    screen_vlnce,
    screening_summary,
)


_BOOLEAN_CLAIMS = {
    "project_self_contained",
    "mp3d_access_authorized",
    "official_metadata_verified",
    "habitat_ready",
    "waypoint_frontend_reproduced",
    "etpr1_reproduced",
}


def _required_evidence_keys(claims: Phase0Evidence) -> set[str]:
    required = {
        name for name in _BOOLEAN_CLAIMS if getattr(claims, name) is True
    }
    if claims.mp3d_scene_count:
        required.add("mp3d_scene_count")
    if claims.screened_instructions or claims.candidate_trajectories:
        required.add("screening_counts")
    if claims.reviewed_candidates or claims.valid_candidates:
        required.add("manual_review")
    if claims.validated_events or claims.unique_expiry_events:
        required.add("event_validation")
    return required


def _verify_artifact_records(
    evidence_payload: dict[str, object], project_root: Path
) -> dict[str, tuple[Path, ...]]:
    verified: dict[str, tuple[Path, ...]] = {}
    for claim_name, records in evidence_payload.items():
        if not isinstance(claim_name, str) or not isinstance(records, list) or not records:
            raise ValueError("each evidence entry must be a non-empty artifact list")
        artifacts: list[Path] = []
        for record in records:
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise ValueError("artifact evidence requires exactly path and sha256")
            artifact_path = record["path"]
            expected_hash = record["sha256"]
            if not isinstance(artifact_path, str):
                raise ValueError("artifact evidence path must be a string")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_hash
                )
            ):
                raise ValueError("artifact evidence requires a lowercase SHA-256")
            artifact = regular_project_file(Path(artifact_path), project_root)
            if sha256_file(artifact) != expected_hash:
                raise ValueError(f"artifact hash mismatch: {artifact_path}")
            artifacts.append(artifact)
        if len(artifacts) != len(set(artifacts)):
            raise ValueError(f"duplicate artifact evidence for {claim_name}")
        verified[claim_name] = tuple(artifacts)
    return verified


def _verify_official_metadata(artifacts: tuple[Path, ...], project_root: Path) -> None:
    root = project_root.resolve()
    relative_paths = {str(path.relative_to(root)) for path in artifacts}
    expected_paths = {
        "data/phase0/manifest.json",
        "data/phase0/raw/rxr_vlnce_v0/train/train_guide.json.gz",
        "data/phase0/raw/rxr_vlnce_v0/val_seen/val_seen_guide.json.gz",
        "data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz",
        "data/phase0/raw/r2r_vlnce_v1-3/val_seen/val_seen.json.gz",
    }
    if relative_paths != expected_paths:
        raise ValueError("official metadata evidence must contain the canonical five files")
    manifest = root / "data/phase0/manifest.json"
    if sha256_file(manifest) != CANONICAL_PHASE0_MANIFEST_SHA256:
        raise ValueError("canonical Phase 0 manifest hash mismatch")
    identities = set()
    for path in expected_paths:
        if path.endswith(".json.gz"):
            asset = canonical_phase0_asset(root / path, root)
            identities.add((asset.dataset, asset.split))
    if identities != {
        ("rxr-ce", "train"),
        ("rxr-ce", "val_seen"),
        ("r2r-ce", "train"),
        ("r2r-ce", "val_seen"),
    }:
        raise ValueError("canonical metadata identities are incomplete")


def _verify_screening_counts(
    artifacts: tuple[Path, ...], claims: Phase0Evidence, project_root: Path
) -> None:
    if len(artifacts) != 1:
        raise ValueError("screening_counts requires exactly one canonical artifact")
    artifact = artifacts[0]
    if artifact != (
        project_root.resolve()
        / "artifacts/phase0/rxr_train_screening_seed20260822.json"
    ):
        raise ValueError("screening_counts must use the frozen RxR train artifact")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid screening artifact: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("screening artifact must be a JSON object")
    source = payload.get("source")
    sampling = payload.get("sampling")
    if not isinstance(source, dict) or not isinstance(sampling, dict):
        raise ValueError("screening artifact lacks source or sampling metadata")
    if source.get("dataset") != "rxr-ce" or source.get("split") != "train":
        raise ValueError("screening artifact must be canonical RxR-CE train")
    if source.get("language_filter") != ["en-IN", "en-US"]:
        raise ValueError("formal RxR-CE screening must include both English tags")
    if source.get("manifest_sha256") != CANONICAL_PHASE0_MANIFEST_SHA256:
        raise ValueError("screening artifact uses a noncanonical manifest")
    source_path = source.get("path")
    if not isinstance(source_path, str):
        raise ValueError("screening source path must be a string")
    asset = canonical_phase0_asset(Path(source_path), project_root)
    if source.get("sha256") != asset.sha256 or source.get("bytes") != asset.byte_count:
        raise ValueError("screening source metadata does not match its canonical asset")
    if sampling != {
        "actual": 50,
        "requested": 50,
        "seed": 20260822,
        "design": "uniform_unique_trajectory_v1",
    }:
        raise ValueError("screening artifact does not use the frozen pilot sample")
    candidates = list(
        screen_vlnce(
            iter_vlnce_episodes(asset.path),
            dataset="rxr-ce",
            split="train",
            languages={"en-US", "en-IN"},
        )
    )
    expected_payload = screening_summary(candidates)
    expected_payload["source"] = {
        "dataset": asset.dataset,
        "split": asset.split,
        "role": asset.role,
        "path": str(asset.path.relative_to(project_root.resolve())),
        "bytes": asset.byte_count,
        "sha256": asset.sha256,
        "manifest_sha256": asset.manifest_sha256,
        "language_filter": ["en-IN", "en-US"],
    }
    expected_payload["sampling"] = {
        "actual": 50,
        "requested": 50,
        "seed": 20260822,
        "design": "uniform_unique_trajectory_v1",
    }
    expected_payload["samples"] = [
        {
            "dataset": candidate.dataset,
            "split": candidate.split,
            "episode_id": candidate.episode_id,
            "instruction_id": candidate.instruction_id,
            "trajectory_id": candidate.trajectory_id,
            "scene_id": candidate.scene_id,
            "language": candidate.language,
            "triggers": list(candidate.triggers),
            "instruction": candidate.instruction,
        }
        for candidate in pilot_sample(candidates, 50, seed=20260822)
    ]
    if payload != expected_payload:
        raise ValueError("screening artifact does not reproduce from canonical source")
    if payload.get("candidate_instructions") != claims.screened_instructions:
        raise ValueError("screened_instructions is not derived from the artifact")
    if payload.get("unique_trajectories") != claims.candidate_trajectories:
        raise ValueError("candidate_trajectories is not derived from the artifact")


def load_phase0_snapshot(path: Path, project_root: Path) -> Phase0Evidence:
    snapshot = regular_project_file(path, project_root)
    try:
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid evidence snapshot: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"claims", "evidence"}:
        raise ValueError("snapshot must contain exactly claims and evidence")
    claims_payload = payload["claims"]
    evidence_payload = payload["evidence"]
    if not isinstance(claims_payload, dict):
        raise ValueError("snapshot claims must be a JSON object")
    expected_claims = {field.name for field in fields(Phase0Evidence)}
    if set(claims_payload) != expected_claims:
        raise ValueError("snapshot claims do not match the Phase0Evidence schema")
    if not isinstance(evidence_payload, dict):
        raise ValueError("snapshot evidence must be a JSON object")
    try:
        claims = Phase0Evidence(**claims_payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Phase 0 claims: {error}") from error

    required_keys = _required_evidence_keys(claims)
    if set(evidence_payload) != required_keys:
        missing = sorted(required_keys.difference(evidence_payload))
        extra = sorted(set(evidence_payload).difference(required_keys))
        raise ValueError(f"evidence keys mismatch; missing={missing}, extra={extra}")
    verified = _verify_artifact_records(evidence_payload, project_root)
    supported = {"official_metadata_verified", "screening_counts"}
    unsupported = sorted(required_keys.difference(supported))
    if unsupported:
        raise ValueError(
            "semantic evidence verifier is not implemented for: "
            + ", ".join(unsupported)
        )
    if "official_metadata_verified" in required_keys:
        _verify_official_metadata(
            verified["official_metadata_verified"], project_root
        )
    if "screening_counts" in required_keys:
        _verify_screening_counts(
            verified["screening_counts"], claims, project_root
        )
    return claims
