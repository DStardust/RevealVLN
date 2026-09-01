#!/usr/bin/env python3
"""Run the fixed independent-visual-label REE learnability probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.codex_visual_review import train_visual_ree  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train_visual_ree(args.device)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
