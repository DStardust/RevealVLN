#!/usr/bin/env python3
"""Contract tests for the V5.18 failure-risk head and calibration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from revealnav_failure_risk import ETPFailureRiskHead  # noqa: E402
from train_r2r_etp_failure_risk_v5_18 import _group_scores, _threshold  # noqa: E402


class FailureRiskTests(unittest.TestCase):
    def test_head_shape(self) -> None:
        model = ETPFailureRiskHead()
        inputs = [torch.zeros(3, 768) for _ in range(5)]
        logits = model(*inputs, torch.zeros(3, 2))
        self.assertEqual(tuple(logits.shape), (3,))

    def test_group_score_is_max_event_risk(self) -> None:
        records = [
            {"group_id": "a", "failure": False},
            {"group_id": "a", "failure": False},
            {"group_id": "b", "failure": True},
        ]
        labels, scores, ids = _group_scores(
            records, np.asarray([0.1, 0.7, 0.6])
        )
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(labels.tolist(), [0.0, 1.0])
        self.assertEqual(scores.tolist(), [0.7, 0.6])

    def test_threshold_respects_success_false_positive_budget(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.float32)
        scores = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
        threshold, policy = _threshold(labels, scores)
        self.assertEqual(threshold, 0.7)
        self.assertEqual(policy["tpr"], 1.0)
        self.assertEqual(policy["fpr"], 0.0)


if __name__ == "__main__":
    unittest.main()
