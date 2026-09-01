#!/usr/bin/env python3
"""Build the isolated 80-event AI-proxy DEC/S/G/E label artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.codex_proxy_ree import build_proxy_labels  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(build_proxy_labels(), indent=2, sort_keys=True))
