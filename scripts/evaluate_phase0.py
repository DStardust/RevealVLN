#!/usr/bin/env python3
"""Evaluate a recorded Phase 0 evidence snapshot against frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporeveal.evidence import load_phase0_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_json", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    try:
        evidence = load_phase0_snapshot(args.evidence_json, project_root)
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "go": evidence.go,
                "valid_rate": evidence.valid_rate,
                "projected_valid_rate_95pct_lower": evidence.projected_valid_rate,
                "estimated_valid_events": evidence.estimated_valid_events,
                "blockers": evidence.blockers,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not evidence.go:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
