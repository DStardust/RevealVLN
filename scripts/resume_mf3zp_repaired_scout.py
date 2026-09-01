#!/usr/bin/env python3
"""Resume the repaired MF3ZP scout with a representation-only compatibility shim.

The imported scout and its protocol are immutable.  The shim makes the
reference helper's string U/A/D states expose the enum-style ``.value``
attribute expected by the scout, without changing the state values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCOUT_SCRIPT = ROOT / "scripts/run_mf3zp_repaired_scout.py"
SCOUT_PROTOCOL = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2r1/MF3ZP_REPAIRED_SCOUT_PROTOCOL.json"
CORRECTION_PROTOCOL = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2r1/MF3ZP_REPAIRED_SCOUT_RESUME_PROTOCOL.json"
METHOD = ROOT / "METHOD_REVISION_3ZP_QWEN_UAD_REFERENCE_V2R1_SCOUT_RESUME.md"
SCHEMA = "revealnav-mf3zp-repaired-scout-resume/1"
STATUS = "SEALED_BEFORE_REPAIRED_SCOUT_RESUME"


class ResumeError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ResumeError(f"path escapes project: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise ResumeError(f"invalid project file: {path}")
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise ResumeError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ResumeError(f"stale partial output: {partial}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def load_scout():
    spec = importlib.util.spec_from_file_location("mf3zp_repaired_scout_immutable", SCOUT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ResumeError("cannot load repaired scout")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_protocol() -> dict:
    if not SCOUT_PROTOCOL.is_file() or SCOUT_PROTOCOL.is_symlink():
        raise ResumeError("sealed repaired-scout protocol missing")
    scout = load_scout()
    scout.verify_scout_protocol()
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "parent_scout_protocol": inventory(SCOUT_PROTOCOL),
        "parent_scout_script": inventory(SCOUT_SCRIPT),
        "method": inventory(METHOD),
        "wrapper": inventory(Path(__file__).resolve()),
        "correction": "string U/A/D values are exposed through a read-only .value compatibility property; values and derivation are unchanged",
        "boundary": {
            "target_payload_read": False,
            "outcome_payload_read": False,
            "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
            "checkpoint_generated": False,
        },
    }


def verify_protocol(value: dict | None = None) -> dict:
    protocol = value if value is not None else json.loads(CORRECTION_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA or protocol.get("status") != STATUS:
        raise ResumeError("resume protocol identity/status drift")
    if protocol.get("boundary", {}).get("public_split_access") != {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}:
        raise ResumeError("public split access is not fail-closed")
    if protocol.get("parent_scout_protocol") != inventory(SCOUT_PROTOCOL):
        raise ResumeError("parent scout protocol changed")
    if protocol.get("parent_scout_script") != inventory(SCOUT_SCRIPT):
        raise ResumeError("parent scout script changed")
    if protocol.get("method") != inventory(METHOD) or protocol.get("wrapper") != inventory(Path(__file__).resolve()):
        raise ResumeError("resume implementation changed")
    scout = load_scout()
    scout.verify_scout_protocol()
    return protocol


class _StringState(str):
    @property
    def value(self) -> str:
        return str(self)


def run(protocol: dict) -> dict:
    verify_protocol(protocol)
    scout = load_scout()
    original = scout.m.derive_uad

    def compatible_derive_uad(*args, **kwargs):
        return tuple(_StringState(value) for value in original(*args, **kwargs))

    scout.m.derive_uad = compatible_derive_uad
    # The immutable scout performs all filtering, response validation, label
    # projection, and outcome-boundary checks itself.
    return scout.run_scout(scout.verify_scout_protocol())


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    sub.add_parser("run")
    args = parser.parse_args()
    try:
        if args.command == "seal":
            if CORRECTION_PROTOCOL.exists() or CORRECTION_PROTOCOL.is_symlink():
                raise ResumeError("resume protocol already exists; resealing is forbidden")
            value = build_protocol()
            atomic_json(CORRECTION_PROTOCOL, value, refuse_existing=True)
            print(json.dumps({"status": value["status"], "protocol_sha256": sha256(CORRECTION_PROTOCOL)}, indent=2))
        else:
            print(json.dumps(run(verify_protocol()), indent=2, ensure_ascii=False))
        return 0
    except BaseException as error:
        print(f"MF3ZP_SCOUT_RESUME_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
