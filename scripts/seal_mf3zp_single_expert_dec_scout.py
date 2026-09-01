#!/usr/bin/env python3
"""Seal the outcome-blind MF3ZP single-expert DEC scout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.single_expert_dec_scout import (  # noqa: E402
    seal_protocols,
    write_selection_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "seal", "all"))
    args = parser.parse_args()
    result: dict[str, object] = {}
    if args.command in {"select", "all"}:
        selection, retest = write_selection_artifacts()
        result["selection"] = {
            "events": selection["event_count"],
            "domains": selection["domain_counts"],
            "scenes": selection["raw_scene_count"],
        }
        result["retest"] = {
            "events": retest["event_count"],
            "domains": retest["domain_counts"],
        }
    if args.command in {"seal", "all"}:
        closure, protocol = seal_protocols()
        result["closure_status"] = closure["status"]
        result["scout_status"] = protocol["status"]
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
