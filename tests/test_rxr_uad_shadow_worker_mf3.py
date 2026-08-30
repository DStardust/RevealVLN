from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rxr_uad_shadow_worker_mf3 import verify_native_trace


class NativeTraceVerificationTest(unittest.TestCase):
    def test_stop_and_ghost_actions_match(self):
        rows = [
            {"step": 0, "native_action_index": 2, "native_action_id": "g1"},
            {"step": 1, "native_action_index": 0, "native_action_id": None},
        ]
        trace = [
            {"act": 4, "ghost_vp": "g1"},
            {"act": 0, "stop_vp": "v0"},
        ]
        self.assertEqual(
            verify_native_trace(rows, trace),
            {"checked_decisions": 2, "all_equal": True},
        )

    def test_action_mismatch_fails_closed(self):
        rows = [
            {"step": 0, "native_action_index": 2, "native_action_id": "g1"}
        ]
        with self.assertRaisesRegex(RuntimeError, "changed or misreported"):
            verify_native_trace(rows, [{"act": 4, "ghost_vp": "g2"}])


if __name__ == "__main__":
    unittest.main()
