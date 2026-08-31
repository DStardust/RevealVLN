#!/usr/bin/env python3
"""Seal or verify the result-independent MF3ZN-TUAD v1 protocol.

With no arguments this script only seals the protocol.  It has deliberately no
collection, training, confirmation, result, or public-evaluation command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revealnav_mf3.tuad_protocol import (  # noqa: E402
    ALLOWED_COMMANDS,
    TUADProtocolError,
    build_protocol,
    canonical_json_bytes,
    sha256_file,
    validate_protocol,
    verify_protocol as verify_sealed_protocol,
)


PROTOCOL_PATH = (
    PROJECT_ROOT
    / "artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json"
)


def _atomic_create(path: Path, payload: bytes) -> None:
    """Create ``path`` atomically without overwriting any prior artifact."""

    if path.exists() or path.is_symlink():
        raise TUADProtocolError(f"refusing to overwrite MF3ZN protocol: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise TUADProtocolError(f"stale MF3ZN protocol partial: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def seal_protocol(
    path: Path = PROTOCOL_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Build and atomically seal the deterministic protocol, and nothing else."""

    value = build_protocol(project_root)
    validate_protocol(value, project_root)
    payload = canonical_json_bytes(value)
    _atomic_create(path, payload)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(persisted, project_root)
    if path.read_bytes() != payload:
        raise TUADProtocolError("persisted MF3ZN protocol is not canonical")
    return persisted


def verify_protocol(
    path: Path = PROTOCOL_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Verify exact source hashes and canonical bytes without writing."""

    return verify_sealed_protocol(path, root=project_root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seal or verify the MF3ZN-TUAD train-development protocol."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="seal",
        choices=ALLOWED_COMMANDS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    value = seal_protocol() if args.command == "seal" else verify_protocol()
    print(json.dumps({
        "status": value["status"],
        "revision": value["revision"],
        "dataset_revision": value["dataset_revision"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "family_stopped": value["family_tombstone"]["value"],
        "collection_authorized": value["authorization"][
            "new_treatment_collection"
        ],
        "public_unseen_authorized": value["authorization"]["public_unseen"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
