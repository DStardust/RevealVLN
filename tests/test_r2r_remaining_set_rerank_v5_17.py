#!/usr/bin/env python3
"""Method-contract tests for V5.17 remaining-set reranking."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r2r_remaining_set_rerank_worker_v5_17 import (  # noqa: E402
    RemainingSetRerankController, _validate_actions,
)


class RemainingSetRerankTests(unittest.TestCase):
    def test_horizon_budget_is_exact(self) -> None:
        self.assertTrue(RemainingSetRerankController._has_switch_budget(11, 15))
        self.assertFalse(RemainingSetRerankController._has_switch_budget(12, 15))

    def test_median_ignores_one_continue_outlier(self) -> None:
        votes, keep, backtrack, decision = (
            RemainingSetRerankController._robust_post_decision([
                (4.59, 4.69), (4.85, 4.44), (5.26, 4.18),
            ])
        )
        self.assertEqual(votes, (False, True, True))
        self.assertLess(backtrack, keep)
        self.assertTrue(decision)

    def test_frozen_etp_ranks_only_unexhausted_options_and_stop(self) -> None:
        index = RemainingSetRerankController._remaining_index(
            [None, "native", "weak", "best", "outside"],
            torch.tensor([9.0, 8.0, 2.0, 7.0, 10.0]),
            ("native", "weak", "best"), {"native"},
        )
        self.assertEqual(index, 0)

    def test_outside_high_logit_cannot_enter_checkpoint_option_set(self) -> None:
        index = RemainingSetRerankController._remaining_index(
            [None, "native", "best", "outside"],
            torch.tensor([1.0, 8.0, 7.0, 10.0]),
            ("native", "best"), {"native"},
        )
        self.assertEqual(index, 2)

    def test_stop_remains_a_valid_frozen_etp_action(self) -> None:
        index = RemainingSetRerankController._remaining_index(
            [None, "native", "other"], torch.tensor([3.0, 2.0, 1.0]),
            ("native", "other"), {"native"},
        )
        self.assertEqual(index, 0)

    def test_exhausted_option_cannot_be_selected_again(self) -> None:
        index = RemainingSetRerankController._remaining_index(
            [None, "first", "second", "third"],
            torch.tensor([0.0, 10.0, 9.0, 8.0]),
            ("first", "second", "third"), {"first", "second"},
        )
        self.assertEqual(index, 3)

    def test_eligible_score_evidence_matches_masked_choice_set(self) -> None:
        rows = RemainingSetRerankController._eligible_score_evidence(
            [None, "first", "second", "outside"],
            torch.tensor([0.5, 9.0, 3.0, 10.0]),
            ("first", "second"), {"first"},
        )
        self.assertEqual(
            rows,
            [
                {"index": 0, "branch_id": None, "score": 0.5, "finite": True},
                {"index": 2, "branch_id": "second", "score": 3.0, "finite": True},
            ],
        )

    def test_stop_validation_requires_a_real_stop_action(self) -> None:
        state = SimpleNamespace(events=[{
            "event": "remaining_set_rerank_committed",
            "branch_id": None,
            "navigation_step": 0,
        }])
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text('{"act": 0, "done": true}\n')
            self.assertTrue(_validate_actions(state, trace)["all_equal"])
            trace.write_text("")
            with self.assertRaises(RuntimeError):
                _validate_actions(state, trace)


if __name__ == "__main__":
    unittest.main()
