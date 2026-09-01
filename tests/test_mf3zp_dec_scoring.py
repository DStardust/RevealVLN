from __future__ import annotations

import unittest

from revealnav_mf3.human_dec_schema import terminal_uad
from revealnav_mf3.single_expert_dec_scout import ScoutError, dec_adequacy_counts


class DecScoringTests(unittest.TestCase):
    def test_manual_mapping_controls_precision_and_recall(self) -> None:
        result = dec_adequacy_counts(
            qwen_proposed_ids=("c1", "c2", "c3", "c4"),
            human_dec_item_count=3,
            mapped_qwen_ids=("c1", "c2"),
        )
        self.assertEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 2 / 3)

    def test_embedding_like_unknown_mapping_is_rejected(self) -> None:
        with self.assertRaises(ScoutError):
            dec_adequacy_counts(
                qwen_proposed_ids=("c1",), human_dec_item_count=1,
                mapped_qwen_ids=("embedding-nearest",),
            )

    def test_uad_is_mechanically_derived_with_exact_k3(self) -> None:
        factors = [
            {"step": index, "instantiated": True, "distinguishable": True, "resolved": True}
            for index in range(3)
        ]
        self.assertEqual(terminal_uad(factors).value, "D")
        factors[0]["resolved"] = False
        self.assertEqual(terminal_uad(factors).value, "A")


if __name__ == "__main__":
    unittest.main()
