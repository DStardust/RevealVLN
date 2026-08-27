#!/usr/bin/env python3
"""Build a collision-safe conversation union from migrated Codex homes."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


SOURCE_NAMES = ("mnt_daiyang_codex", "root_codex", "mnt_root_codex")
SECTIONS = ("sessions", "archived_sessions")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            result.update(chunk)
    return result.hexdigest()


def is_prefix(shorter, longer):
    if shorter.stat().st_size > longer.stat().st_size:
        return False
    with shorter.open("rb") as left, longer.open("rb") as right:
        while True:
            chunk = left.read(8 << 20)
            if not chunk:
                return True
            if chunk != right.read(len(chunk)):
                return False


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == source.stat().st_size:
            if digest(destination) == digest(source):
                return
    part = destination.with_name(destination.name + ".migration-part")
    shutil.copy2(str(source), str(part))
    os.replace(str(part), str(destination))


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".migration-part")
    with part.open("wb") as handle:
        handle.write(content)
    os.replace(str(part), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    target = args.target.resolve()
    snapshots = target / "migration_sources"
    sources = [snapshots / name for name in SOURCE_NAMES]
    if any(not source.is_dir() for source in sources):
        raise SystemExit("missing Codex migration source snapshot")

    variants = {}
    for source in sources:
        for section in SECTIONS:
            section_root = source / section
            if not section_root.is_dir():
                continue
            for path in section_root.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    relative = Path(section) / path.relative_to(section_root)
                    variants.setdefault(str(relative), []).append(path)

    identical_overlaps = 0
    prefix_overlaps = 0
    for relative, candidates in sorted(variants.items()):
        winner = max(candidates, key=lambda item: item.stat().st_size)
        winner_digest = digest(winner)
        for candidate in candidates:
            if candidate == winner:
                continue
            if candidate.stat().st_size == winner.stat().st_size:
                if digest(candidate) != winner_digest:
                    raise SystemExit("divergent equal-length session: " + relative)
                identical_overlaps += 1
            elif is_prefix(candidate, winner):
                prefix_overlaps += 1
            else:
                raise SystemExit("divergent session variants: " + relative)
        atomic_copy(winner, target / relative)

    history = {}
    for source in sources:
        history_path = source / "history.jsonl"
        if not history_path.is_file():
            continue
        for raw in history_path.read_bytes().splitlines():
            record = json.loads(raw.decode("utf-8"))
            key = (record.get("session_id"), record.get("ts"), record.get("text"))
            history[key] = raw
    ordered = sorted(history.items(), key=lambda item: (item[0][1] or 0, item[0][0] or "", item[0][2] or ""))
    history_bytes = b"\n".join(raw for _, raw in ordered)
    if history_bytes:
        history_bytes += b"\n"
    atomic_write(target / "history.jsonl", history_bytes)

    report = {
        "status": "PASS",
        "unique_session_files": len(variants),
        "history_records": len(ordered),
        "identical_overlaps": identical_overlaps,
        "prefix_overlaps": prefix_overlaps,
        "sources": list(SOURCE_NAMES),
    }
    atomic_write(
        target / "CODEX_CONVERSATION_UNION.json",
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
