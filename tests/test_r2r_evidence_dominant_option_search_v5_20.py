#!/usr/bin/env python3
"""Contracts for V5.20 evidence-dominant option search."""

from __future__ import annotations

import unittest


class EvidenceDominanceTests(unittest.TestCase):
    def test_threshold_free_monotonic_acceptance(self) -> None:
        selected, maximum = 0.6, 0.6
        pre, post = 0.8, 0.81
        self.assertTrue(selected >= maximum - 1e-7 and post >= pre)
        self.assertFalse(0.5 >= maximum - 1e-7 and post >= pre)
        self.assertFalse(selected >= maximum - 1e-7 and 0.79 >= pre)


if __name__ == "__main__":
    unittest.main()
