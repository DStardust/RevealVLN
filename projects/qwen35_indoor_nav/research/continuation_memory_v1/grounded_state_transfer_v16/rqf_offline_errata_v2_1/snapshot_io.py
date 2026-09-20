"""Byte-bound input sealing and read-only content-addressed access."""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Any, Iterable

from common import digest_bytes, load_json_bytes, require_within


class SnapshotError(RuntimeError):
    pass


class EvidenceMissing(SnapshotError):
    pass


class EvidenceInvalid(SnapshotError):
    pass


def _stable_read(path: Path, allowed_root: Path) -> tuple[bytes, os.stat_result, os.stat_result]:
    resolved = require_within(path, allowed_root)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise EvidenceInvalid(f"SOURCE_NOT_REGULAR_FILE:{path}")
    data = resolved.read_bytes()
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(data) != after.st_size:
        raise EvidenceInvalid(f"SOURCE_CHANGED_DURING_READ:{path}")
    return data, before, after


def _store_object(object_root: Path, expected_sha256: str, data: bytes) -> None:
    if digest_bytes(data) != expected_sha256:
        raise EvidenceInvalid(f"OBJECT_PAYLOAD_HASH:{expected_sha256}")
    object_root.mkdir(parents=True, exist_ok=True)
    destination = object_root / expected_sha256
    if destination.exists():
        if destination.is_symlink() or digest_bytes(destination.read_bytes()) != expected_sha256:
            raise EvidenceInvalid(f"EXISTING_OBJECT_CORRUPT:{destination}")
        return
    temporary = object_root / f".{expected_sha256}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _log_prefix_metadata(data: bytes) -> dict[str, Any]:
    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        complete, tail = b"", data
        last_line = 0
    else:
        complete, tail = data[: last_newline + 1], data[last_newline + 1 :]
        last_line = complete.count(b"\n")
    return {
        "byte_start": 0,
        "byte_end": len(data),
        "prefix_sha256": digest_bytes(data),
        "last_complete_line": last_line,
        "incomplete_tail_bytes": len(tail),
    }


def seal_snapshot(
    old_snapshot: dict[str, Any],
    source_roots: Iterable[Path],
    object_root: Path,
    supplemental: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Recover every declared old object by exact hash, then seal fixed context."""
    roots = [Path(root).resolve(strict=True) for root in source_roots]
    started = time.time()
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for old_record in old_snapshot.get("file_records", []):
        relative = old_record.get("path")
        expected = old_record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise EvidenceInvalid("OLD_SNAPSHOT_RECORD_SCHEMA")
        found: tuple[bytes, Path, os.stat_result] | None = None
        mismatches: list[dict[str, Any]] = []
        for root in roots:
            candidate = root / relative
            if not candidate.exists():
                continue
            try:
                data, before, _ = _stable_read(candidate, root)
            except (OSError, ValueError, SnapshotError) as exc:
                mismatches.append({"root": str(root), "error": repr(exc)})
                continue
            actual = digest_bytes(data)
            if actual == expected:
                found = data, candidate, before
                break
            mismatches.append({"root": str(root), "actual_sha256": actual})
        if found is None:
            missing.append(
                {
                    "source_path": relative,
                    "expected_sha256": expected,
                    "status": "UNRECOVERABLE_ORIGINAL_BYTES",
                    "observed": mismatches,
                }
            )
            records.append(missing[-1])
            continue
        data, candidate, source_stat = found
        _store_object(object_root, expected, data)
        record: dict[str, Any] = {
            "kind": "original_v2_input",
            "source_path": relative,
            "expected_sha256": expected,
            "object_sha256": expected,
            "bytes": len(data),
            "source_used": str(candidate),
            "source_mtime_ns_at_capture": source_stat.st_mtime_ns,
            "equivalence": "EXACT_ORIGINAL_BYTES",
        }
        if candidate.suffix in {".jsonl", ".log"} or candidate.name.endswith(".log"):
            record["log_prefix"] = _log_prefix_metadata(data)
        records.append(record)

    for item in supplemental:
        path = Path(item["path"])
        root = Path(item["allowed_root"])
        data, source_stat, _ = _stable_read(path, root)
        actual = digest_bytes(data)
        expected = item.get("expected_sha256")
        if expected is not None and actual != expected:
            raise EvidenceInvalid(f"SUPPLEMENTAL_IDENTITY:{path}:{actual}:{expected}")
        _store_object(object_root, actual, data)
        records.append(
            {
                "kind": item["kind"],
                "source_path": item["source_path"],
                "object_sha256": actual,
                "expected_sha256": expected,
                "bytes": len(data),
                "source_used": str(path),
                "source_mtime_ns_at_capture": source_stat.st_mtime_ns,
                "equivalence": item["equivalence"],
            }
        )
    finished = time.time()
    unique_objects = {row.get("object_sha256") for row in records if row.get("object_sha256")}
    return {
        "schema_version": "q35n.snapshot.v2.1",
        "capture_started_at_unix": started,
        "capture_finished_at_unix": finished,
        "object_store": "inputs/objects",
        "records": records,
        "record_count": len(records),
        "unique_object_count": len(unique_objects),
        "unrecoverable_original_count": len(missing),
        "read_policy": "all audit reads resolve through object_sha256; no live fallback",
    }


class SnapshotReader:
    def __init__(self, manifest: dict[str, Any], object_root: Path):
        self.manifest = manifest
        self.object_root = Path(object_root).resolve(strict=True)
        self._by_path: dict[str, dict[str, Any]] = {}
        self._by_hash: dict[str, dict[str, Any]] = {}
        for record in manifest.get("records", []):
            source_path = record.get("source_path")
            object_hash = record.get("object_sha256")
            if object_hash is None:
                continue
            if not isinstance(source_path, str) or not isinstance(object_hash, str):
                raise EvidenceInvalid("SNAPSHOT_MANIFEST_RECORD")
            if source_path in self._by_path:
                previous = self._by_path[source_path]
                if previous["object_sha256"] != object_hash:
                    raise EvidenceInvalid(f"DUPLICATE_SOURCE_IDENTITY:{source_path}")
            self._by_path[source_path] = record
            self._by_hash[object_hash] = record
        self.verify()

    @classmethod
    def from_paths(cls, manifest_path: Path, object_root: Path) -> "SnapshotReader":
        manifest_path = Path(manifest_path)
        if manifest_path.is_symlink():
            raise EvidenceInvalid(f"SNAPSHOT_MANIFEST_SYMLINK:{manifest_path}")
        return cls(load_json_bytes(manifest_path.read_bytes(), str(manifest_path)), object_root)

    def verify(self) -> None:
        for object_hash in self._by_hash:
            self.read_bytes(object_hash)

    def list_records(self, kind: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.manifest.get("records", []) if row.get("kind") == kind]

    def record_for_path(self, source_path: str) -> dict[str, Any]:
        try:
            return dict(self._by_path[source_path])
        except KeyError as exc:
            raise EvidenceMissing(f"SNAPSHOT_PATH_MISSING:{source_path}") from exc

    def read_path_bytes(self, source_path: str) -> bytes:
        return self.read_bytes(self.record_for_path(source_path)["object_sha256"])

    def read_path_json(self, source_path: str) -> Any:
        record = self.record_for_path(source_path)
        return self.read_json(record["object_sha256"])

    def read_bytes(self, object_sha256: str) -> bytes:
        if not isinstance(object_sha256, str) or len(object_sha256) != 64:
            raise EvidenceInvalid(f"OBJECT_ID:{object_sha256!r}")
        path = self.object_root / object_sha256
        if not path.exists():
            raise EvidenceMissing(f"OBJECT_MISSING:{object_sha256}")
        if path.is_symlink():
            raise EvidenceInvalid(f"OBJECT_SYMLINK:{object_sha256}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.object_root):
            raise EvidenceInvalid(f"OBJECT_ESCAPE:{object_sha256}")
        data = resolved.read_bytes()
        actual = digest_bytes(data)
        if actual != object_sha256:
            raise EvidenceInvalid(f"OBJECT_HASH_MISMATCH:{object_sha256}:{actual}")
        return data

    def read_json(self, object_sha256: str) -> Any:
        return load_json_bytes(self.read_bytes(object_sha256), object_sha256)

