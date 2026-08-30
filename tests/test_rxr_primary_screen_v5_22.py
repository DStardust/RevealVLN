#!/usr/bin/env python3
"""Pure contract tests for the RxR V5.22 primary screen."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rxr_primary_screen_v5_22 as screen  # noqa: E402


class RxRPrimaryScreenV522Test(unittest.TestCase):
    def test_selection_is_scene_balanced_and_deterministic(self):
        first, counts = screen.selection()
        second, _ = screen.selection()
        self.assertEqual(first, second)
        self.assertEqual(counts["selected_episodes"], 24)
        self.assertEqual(len({row["scene_id"] for row in first}), 24)
        self.assertTrue(all(row["language"] in {"en-US", "en-IN"} for row in first))

    def test_protocol_preserves_benchmark_roles(self):
        value = screen.protocol_value()
        self.assertIn("RxR-CE", value["scope"])
        self.assertEqual(value["runs"], {"baseline": 24, "revealnav": 72, "total": 96})
        self.assertIn("R2R val_unseen", value["forbidden"])

    def test_hash_chain_rejects_mutation(self):
        previous = "0" * 64
        rows = []
        for index in range(2):
            row = {"event": "x", "step": index, "previous_hash": previous}
            row["record_hash"] = screen.hashlib.sha256(json.dumps(
                row, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            previous = row["record_hash"]
            rows.append(row)
        self.assertTrue(screen.valid_chain(rows))
        rows[1]["step"] = 3
        self.assertFalse(screen.valid_chain(rows))


if __name__ == "__main__":
    unittest.main()
