from __future__ import annotations

import unittest

from revealnav_mf3.single_expert_dec_scout import false_decisive_summary


class FalseDecisiveTests(unittest.TestCase):
    def test_false_decisive_rate_uses_qwen_d_denominator(self) -> None:
        pairs = [("D", "D")] * 18 + [("A", "D")] * 2 + [("U", "A")] * 10
        result = false_decisive_summary(pairs, minimum_support=20)
        self.assertEqual(result["qwen_D_count"], 20)
        self.assertEqual(result["false_D_count"], 2)
        self.assertAlmostEqual(result["rate"], 0.1)
        self.assertEqual(result["support_status"], "SUFFICIENT_QWEN_D_SUPPORT")

    def test_insufficient_d_support_cannot_be_interpreted_as_pass(self) -> None:
        result = false_decisive_summary([("D", "D")] * 19, minimum_support=20)
        self.assertEqual(result["support_status"], "INSUFFICIENT_QWEN_D_SUPPORT")
        self.assertEqual(result["qwen_D_count"], 19)


if __name__ == "__main__":
    unittest.main()
