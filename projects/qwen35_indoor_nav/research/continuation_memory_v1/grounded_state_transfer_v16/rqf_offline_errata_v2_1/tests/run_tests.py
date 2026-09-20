"""Run CPU cases and emit per-ID machine-readable results."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
for path in (str(PACKAGE), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)


class RecordingResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.cases: list[dict[str, object]] = []
        self.started: dict[str, float] = {}

    def _id(self, test):
        name = test._testMethodName
        parts = name.split("_", 2)
        return parts[1] if len(parts) > 1 else name

    def startTest(self, test):
        super().startTest(test)
        self.started[test.id()] = time.monotonic()

    def _record(self, test, status, detail=None):
        row = {
            "test_id": self._id(test),
            "test_name": test.id(),
            "status": status,
            "seconds": time.monotonic() - self.started[test.id()],
        }
        if detail:
            row["failure"] = detail
        self.cases.append(row)

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "FAIL", "".join(traceback.format_exception(*err)))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "ERROR", "".join(traceback.format_exception(*err)))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "SKIP", reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py")
    result = RecordingResult()
    began = time.time()
    suite.run(result)
    cases = sorted(result.cases, key=lambda row: (str(row["test_id"]), str(row["test_name"])))
    counts = {status: sum(row["status"] == status for row in cases) for status in ("PASS", "FAIL", "ERROR", "SKIP")}
    payload = {
        "status": "PASS" if counts == {"PASS": len(cases), "FAIL": 0, "ERROR": 0, "SKIP": 0} else "FAIL",
        "cases": cases,
        "executed": len(cases),
        "passed": counts["PASS"],
        "failed": counts["FAIL"] + counts["ERROR"],
        "not_executed": counts["SKIP"],
        "started_at_unix": began,
        "finished_at_unix": time.time(),
        "simulator_used": False,
        "model_loaded": False,
        "new_physical_candidate_executions": 0,
        "new_family_admissions": 0,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "executed", "passed", "failed", "not_executed")}, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
