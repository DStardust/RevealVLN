#!/usr/bin/env python3
"""Static contract tests for the expanded branch-excursion label pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rxr_branch_excursion_labels_v5 as run  # noqa: E402
import rxr_branch_excursion_label_worker_v5 as worker  # noqa: E402


class BranchExcursionV5ContractTest(unittest.TestCase):
    def test_population_closure(self) -> None:
        records = run.train_records()
        self.assertEqual(len(records), 1830)
        self.assertEqual(sum(row["split"] != "train" for row in records), 0)
        self.assertEqual(len(run.legacy_records()), 424)

    def test_all_source_bundles_exist(self) -> None:
        for source in run.EXPECTED_SOURCES:
            paths = worker.bundle(source)
            self.assertEqual(set(paths), {"geometry", "causal", "shards", "tx"})
            self.assertTrue(paths["geometry"].is_file())
            self.assertTrue(paths["causal"].is_file())
            self.assertTrue(paths["shards"].is_dir())
            self.assertTrue(paths["tx"].is_dir())

    def test_protocol_excludes_evaluation_payloads(self) -> None:
        value = run.protocol_value()
        self.assertFalse(value["development_access_allowed"])
        self.assertFalse(value["gold_access_allowed"])
        self.assertEqual(value["future_information_used_for_online_input"], 0)


if __name__ == "__main__":
    unittest.main()
