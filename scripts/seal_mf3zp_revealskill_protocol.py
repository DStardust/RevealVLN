#!/usr/bin/env python3
"""Seal or verify the MF3ZP RevealSkill protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.revealskill_protocol import PROTOCOL_PATH, seal_protocol, sha256_file, verify_protocol  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "verify"))
    args = parser.parse_args()
    result = seal_protocol() if args.command == "seal" else verify_protocol()
    print(json.dumps({
        "status": result["status"],
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL_PATH),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
