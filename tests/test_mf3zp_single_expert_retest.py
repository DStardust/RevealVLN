from __future__ import annotations

import inspect
import unittest

from revealnav_mf3.human_dec_schema import validate_review_row
from revealnav_mf3.single_expert_dec_scout import (
    build_review_rows,
    build_selection_artifacts,
    prepare_retest,
    select_retest_events,
)


class SingleExpertRetestTests(unittest.TestCase):
    def test_blank_retest_is_twenty_events_and_has_no_first_labels(self) -> None:
        _, selection = build_selection_artifacts()
        rows = build_review_rows(mode="retest", selection=selection)
        self.assertEqual(len(rows), 20)
        for row in rows:
            validate_review_row(row, require_complete=False, expected_mode="retest")
            self.assertEqual(row["reviewer_id"], "")
            self.assertFalse(row["review_complete"])

    def test_retest_selection_api_has_no_first_review_argument(self) -> None:
        self.assertEqual(list(inspect.signature(select_retest_events).parameters), ["selected"])
        source = inspect.getsource(prepare_retest)
        self.assertNotIn("_load_completed_review", source)
        self.assertNotIn("REVIEW_TEMPLATE", source)


if __name__ == "__main__":
    unittest.main()
