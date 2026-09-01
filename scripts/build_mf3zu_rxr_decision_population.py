#!/usr/bin/env python3
"""Materialize fixed MF3ZU sanitized-population and exact-target artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zu_protocol import (  # noqa: E402
    POPULATION_MANIFEST_PATH,
    ProtocolError,
    write_population,
)


def main() -> int:
    try:
        value = write_population()
    except (OSError, KeyError, TypeError, ValueError, ProtocolError) as error:
        print(f"MF3ZU_RXR_POPULATION_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "path": str(POPULATION_MANIFEST_PATH),
                "status": value["status"],
                "population_rows": value["population_rows"],
                "episodes": value["episodes"],
                "raw_scenes": value["raw_scenes"],
                "exact_target_rows": value["exact_target_rows"],
                "exact_target_accessed_for_support_eligibility": value[
                    "exact_target_accessed_for_support_eligibility"
                ],
                "target_value_in_sanitized_population": value[
                    "target_value_in_sanitized_population"
                ],
                "training_target_access": value["target_access_boundary"][
                    "training_value_access"
                ],
                "next_action": "CAUSAL_OBSERVATION_REPLAY_AND_EVIDENCE_FREEZE",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
