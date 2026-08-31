import unittest

import numpy as np

from revealnav_mf3.mf3zo_probes import scene_bootstrap_mean


class MF3ZOBootstrapTest(unittest.TestCase):
    def test_bootstrap_is_scene_clustered_and_deterministic(self):
        scenes = np.asarray(["a", "a", "b", "b", "c", "c"])
        values = np.asarray([1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
        mask = np.ones(6, dtype=np.bool_)
        first = scene_bootstrap_mean(values, scenes, mask, seed=123)
        second = scene_bootstrap_mean(values, scenes, mask, seed=123)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed"], values.mean())
        self.assertLessEqual(first["lower_95"], first["observed"])
        self.assertGreaterEqual(first["upper_95"], first["observed"])


if __name__ == "__main__":
    unittest.main()

