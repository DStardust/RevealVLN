"""Project-local, hash-bound input provenance for Phase 0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


CANONICAL_PHASE0_MANIFEST_SHA256 = (
    "a5f0a9abdf48156b3e3336951399640f8c3d347f16048f398ff6fd37d8a60476"
)

@dataclass(frozen=True)
class DatasetAsset:
    dataset: str
    split: str
    role: str
    path: Path
    byte_count: int
    sha256: str
    source_url: str
    manifest_sha256: str


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def regular_project_file(path: Path, project_root: Path) -> Path:
    root = project_root.resolve()
    lexical = path if path.is_absolute() else root / path
    current = root
    try:
        relative_lexical = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError(f"file is outside the project: {path}") from error
    for part in relative_lexical.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinked project input is forbidden: {path}")
    resolved = lexical.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"not a regular project-local file: {path}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_dataset_asset(
    manifest_path: Path,
    annotation_path: Path,
    project_root: Path,
) -> DatasetAsset:
    """Resolve source identity from canonical path and content, never CLI labels."""

    manifest_file = regular_project_file(manifest_path, project_root)
    annotation_file = regular_project_file(annotation_path, project_root)
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid asset manifest: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("asset manifest must use schema_version 1")
    records = payload.get("assets")
    if not isinstance(records, list):
        raise ValueError("asset manifest must contain an assets list")

    matches: list[dict[str, object]] = []
    root = project_root.resolve()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("every asset record must contain a string path")
        record_path = Path(record["path"])
        if record_path.is_absolute():
            raise ValueError("manifest asset paths must be project-relative")
        resolved_record = (root / record_path).resolve()
        if not resolved_record.is_relative_to(root):
            raise ValueError("manifest asset path escapes the project")
        if resolved_record == annotation_file:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError("annotation must match exactly one canonical manifest asset")

    record = matches[0]
    dataset = record.get("dataset")
    split = record.get("split")
    role = record.get("role")
    byte_count = record.get("bytes")
    expected_hash = record.get("sha256")
    source_url = record.get("source_url")
    if dataset not in {"rxr-ce", "r2r-ce"}:
        raise ValueError("manifest contains an unsupported dataset")
    if split not in {"train", "val_seen"}:
        raise ValueError("manifest contains a forbidden split")
    if not isinstance(role, str) or not role:
        raise ValueError("manifest role must be a non-empty string")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool):
        raise ValueError("manifest bytes must be an integer")
    if not _is_lower_sha256(expected_hash):
        raise ValueError("manifest sha256 must be a lowercase digest")
    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        raise ValueError("manifest source_url must use HTTPS")
    if annotation_file.stat().st_size != byte_count:
        raise ValueError("annotation byte count does not match the manifest")
    actual_hash = sha256_file(annotation_file)
    if actual_hash != expected_hash:
        raise ValueError("annotation SHA-256 does not match the manifest")
    return DatasetAsset(
        dataset=dataset,
        split=split,
        role=role,
        path=annotation_file,
        byte_count=byte_count,
        sha256=actual_hash,
        source_url=source_url,
        manifest_sha256=sha256_file(manifest_file),
    )


def canonical_phase0_asset(
    annotation_path: Path, project_root: Path
) -> DatasetAsset:
    manifest = project_root.resolve() / "data/phase0/manifest.json"
    manifest_file = regular_project_file(manifest, project_root)
    if sha256_file(manifest_file) != CANONICAL_PHASE0_MANIFEST_SHA256:
        raise ValueError("canonical Phase 0 manifest hash is not frozen")
    return verified_dataset_asset(manifest_file, annotation_path, project_root)
