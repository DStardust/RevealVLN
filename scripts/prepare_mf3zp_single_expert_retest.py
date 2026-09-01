#!/usr/bin/env python3
"""Materialize the sealed blank retest only after first-pass validation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.single_expert_dec_scout import prepare_retest  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(prepare_retest(), indent=2, sort_keys=True, ensure_ascii=False))
