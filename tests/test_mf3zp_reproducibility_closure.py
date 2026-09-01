from __future__ import annotations

import subprocess
import unittest

from revealnav_mf3.single_expert_dec_scout import (
    BASE_REVIEW_COMMIT,
    EXPECTED_HISTORICAL_SHA256,
    OUTPUT,
    PUBLIC_CLOSED,
    ROOT,
    _record_inventories,
    sha256_file,
)


class ReproducibilityClosureTests(unittest.TestCase):
    def test_reviewed_base_is_history_ancestor_not_head_equality(self) -> None:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_REVIEW_COMMIT, "HEAD"],
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0)

    def test_historical_scientific_files_are_byte_identical(self) -> None:
        for relative, expected in EXPECTED_HISTORICAL_SHA256.items():
            self.assertEqual(sha256_file(ROOT / relative), expected)

    def test_frozen_qwen_record_inventories_are_complete(self) -> None:
        inventories = _record_inventories()
        self.assertEqual(inventories["instruction_graph_records"]["count"], 141)
        self.assertEqual(inventories["qwen_evidence_records"]["count"], 538)

    def test_scout_cannot_open_public_splits_or_create_checkpoint(self) -> None:
        self.assertEqual(
            PUBLIC_CLOSED,
            {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        )
        self.assertFalse(any(OUTPUT.glob("*.pt")))
        self.assertFalse(any(OUTPUT.glob("*.ckpt")))


if __name__ == "__main__":
    unittest.main()
