from __future__ import annotations

import copy
import inspect
import unittest

from revealnav_mf3.human_dec_schema import (
    HumanDecSchemaError,
    reject_blinded_payload,
    validate_review_row,
)
from revealnav_mf3.single_expert_dec_scout import (
    ScoutError,
    UAD_K,
    build_review_rows,
    build_selection_artifacts,
    validate_frozen_review_fields,
)


class SingleExpertBlindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        selection, _ = build_selection_artifacts()
        cls.rows = build_review_rows(mode="first", selection=selection)

    def test_real_templates_are_blank_strictly_causal_and_k3(self) -> None:
        self.assertEqual(len(self.rows), 80)
        self.assertEqual(UAD_K, 3)
        for row in self.rows:
            validate_review_row(row, require_complete=False, expected_mode="first")
            self.assertLessEqual(len(row["prefixes"]), 5)
            self.assertTrue(all(prefix["step"] <= row["decision_step"] for prefix in row["prefixes"]))
            self.assertTrue(all(item["dec_role"] is None for item in row["constraint_reviews"].values()))

    def test_forbidden_outcome_and_qwen_factor_fields_fail_closed(self) -> None:
        for key in ("delta_utility", "success", "qwen_sge", "native_action", "runner_action"):
            with self.subTest(key=key), self.assertRaises(HumanDecSchemaError):
                reject_blinded_payload({key: 1})

    def test_dec_labels_cannot_mutate_the_frozen_qwen_graph(self) -> None:
        expected = self.rows[0]
        actual = copy.deepcopy(expected)
        actual["constraint_graph"][0]["subject"] = "mutated"
        with self.assertRaises(ScoutError):
            validate_frozen_review_fields(actual, expected)

    def test_scout_source_has_no_qwen_api_client(self) -> None:
        source = inspect.getsource(__import__(
            "revealnav_mf3.single_expert_dec_scout", fromlist=["dummy"]
        ))
        self.assertNotIn("dashscope", source.casefold())
        self.assertNotIn("api_key", source.casefold())


if __name__ == "__main__":
    unittest.main()
