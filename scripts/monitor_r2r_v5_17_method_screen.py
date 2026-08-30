#!/usr/bin/env python3
"""Compact one-shot or live monitor for the V5.17 method screen."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_17_method_screen"
STATUS = OUT / "RUN_STATUS.json"
RESULT = OUT / "R2R_V5_17_METHOD_SCREEN_RESULT.json"
EXPECTED = 72


def snapshot() -> dict:
    summaries = list((OUT / "runs").glob("*/RUN_SUMMARY.json"))
    passed = 0
    failed = 0
    for path in summaries:
        try:
            status = json.loads(path.read_text()).get("status")
        except (OSError, json.JSONDecodeError):
            continue
        passed += status == "PASS"
        failed += status != "PASS"
    value = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completed_summaries": len(summaries),
        "passed": passed,
        "failed": failed,
        "expected": EXPECTED,
        "remaining": max(0, EXPECTED - len(summaries)),
    }
    if STATUS.is_file():
        value["executor"] = json.loads(STATUS.read_text())
    if RESULT.is_file():
        result = json.loads(RESULT.read_text())
        value["result"] = {
            "status": result.get("status"),
            "scientific_outcome": result.get("scientific_outcome"),
            "method_screen_pass": result.get("method_screen_pass"),
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    while True:
        value = snapshot()
        print(json.dumps(value, ensure_ascii=False), flush=True)
        if not args.watch or value["remaining"] == 0 or "result" in value:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
