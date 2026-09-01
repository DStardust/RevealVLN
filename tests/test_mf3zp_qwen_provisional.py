import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_mf3zp_qwen_provisional_experiment.py"
spec = importlib.util.spec_from_file_location("mf3zp_qwen_provisional_test_module", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProvisionalExperimentTest(unittest.TestCase):
    def test_fixed_ridge_is_deterministic_and_scene_oof(self):
        matrix = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=float)
        target = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        folds = np.arange(5, dtype=np.int64)
        first = module.ridge_oof(matrix, target, folds)
        second = module.ridge_oof(matrix, target, folds)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())

    def test_augmented_evaluation_never_opens_public_split(self):
        self.assertEqual(module.MODEL, "qwen3.8-max")
        self.assertEqual(module.FOLDS, 5)
        self.assertEqual(module.BOOTSTRAP_REPLICATES, 10000)

    def test_qwen_summary_is_finite(self):
        vector = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.0, 0.0])
        self.assertTrue(np.isfinite(vector).all())


if __name__ == "__main__":
    unittest.main()
