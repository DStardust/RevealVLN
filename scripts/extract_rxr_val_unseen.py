#!/usr/bin/env python3
"""Extract only the locked RxR-CE guide val_unseen evaluation payloads."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import zlib
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts/upstream/ETP-R1-extra-files-86cacf29.zip"
DESTINATION = ROOT / "third_party/ETP-R1"
MANIFEST = ROOT / "artifacts/upstream/RXR_VAL_UNSEEN_EXTRACTION.json"
EXPECTED_ARCHIVE_BYTES = 15_238_627_709
EXPECTED_ARCHIVE_SHA256 = (
    "f3de48e9184eeff380b4fdc83769131a358e6b7e0ea37e057f4d431cfb027fa0"
)
MEMBERS = {
    "extra_files/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/"
    "val_unseen_guide.json.gz": (4_551_924, 0x9C8AEAEC),
    "extra_files/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/"
    "val_unseen_guide_gt.json.gz": (13_167_787, 0xF2D905E5),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_not_symlink(path: Path) -> bool:
    mode = path.lstat().st_mode
    return stat.S_ISREG(mode) and not path.is_symlink()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def destination_for(member: str) -> Path:
    prefix = "extra_files/"
    if not member.startswith(prefix):
        raise RuntimeError("archive member lacks the fixed prefix")
    destination = DESTINATION / member.removeprefix(prefix)
    if DESTINATION not in destination.resolve().parents:
        raise RuntimeError("archive destination escapes ETP-R1")
    return destination


def extract_member(archive: ZipFile, member: str, size: int, crc: int) -> dict:
    destination = destination_for(member)
    part = destination.with_name(destination.name + ".part")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to overwrite {destination}")
    if part.exists() or part.is_symlink():
        raise RuntimeError(f"stale extraction part: {part}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cursor = DESTINATION
    for component in destination.relative_to(DESTINATION).parts:
        cursor = cursor / component
        if cursor.exists() and cursor.is_symlink():
            raise RuntimeError(f"symlink in extraction path: {cursor}")
    digest = hashlib.sha256()
    computed_crc = 0
    written = 0
    try:
        with archive.open(member) as source, part.open("xb") as output:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                computed_crc = zlib.crc32(block, computed_crc)
                written += len(block)
            output.flush()
            os.fsync(output.fileno())
        if written != size or computed_crc & 0xFFFFFFFF != crc:
            raise RuntimeError(f"extracted member integrity mismatch: {member}")
        os.replace(part, destination)
    except BaseException:
        if part.exists() and regular_not_symlink(part):
            part.unlink()
        raise
    if not regular_not_symlink(destination):
        raise RuntimeError(f"extracted destination is not a regular file: {destination}")
    return {
        "archive_member": member,
        "path": str(destination.relative_to(ROOT)),
        "bytes": written,
        "zip_crc32": f"{crc:08x}",
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    if not regular_not_symlink(ARCHIVE):
        raise RuntimeError("ETP-R1 archive is not a regular file")
    if ARCHIVE.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("ETP-R1 archive byte count drift")
    if MANIFEST.exists() or MANIFEST.is_symlink():
        raise RuntimeError(f"refusing to overwrite {MANIFEST}")
    with ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        results = []
        for member, (size, crc) in MEMBERS.items():
            matches = [info for info in infos if info.filename == member]
            if len(matches) != 1:
                raise RuntimeError(f"archive member count is not one: {member}")
            info = matches[0]
            if info.file_size != size or info.CRC != crc:
                raise RuntimeError(f"archive metadata drift: {member}")
            results.append(extract_member(archive, member, size, crc))
    for result in results:
        path = ROOT / result["path"]
        decompressed = 0
        with gzip.open(path, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                decompressed += len(block)
        result["gzip_integrity"] = "PASS"
        result["gzip_uncompressed_bytes"] = decompressed
    value = {
        "schema_version": "revealnav-rxr-val-unseen-extraction/1",
        "status": "RXR_VAL_UNSEEN_EXTRACTION_PASS",
        "authorization": "user-requested locked-model unseen controller integration",
        "archive": {
            "path": str(ARCHIVE.relative_to(ROOT)),
            "bytes": EXPECTED_ARCHIVE_BYTES,
            "sha256_from_independent_phase0_full_reverification": (
                EXPECTED_ARCHIVE_SHA256
            ),
        },
        "files": results,
        "total_extracted_bytes": sum(row["bytes"] for row in results),
        "roles": ["guide"],
        "split": "val_unseen",
        "test_or_test_challenge_extracted": False,
    }
    atomic_json(MANIFEST, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
