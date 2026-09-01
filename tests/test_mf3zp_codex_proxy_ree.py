import unittest

from revealnav_mf3.codex_proxy_ree import infer_proxy_dec_roles, scene_fold


class CodexProxyREETest(unittest.TestCase):
    def test_proxy_dec_uses_current_terminal_and_prior_dependencies(self):
        graph = [
            {"constraint_id": "c1", "dependencies": []},
            {"constraint_id": "c2", "dependencies": ["c1"]},
            {"constraint_id": "c3", "dependencies": ["c2"]},
        ]
        blank = lambda: {  # noqa: E731
            "instantiated": False, "distinguishable": False,
            "resolved": False, "candidate_ids": [],
            "evidence_image_indices": [],
        }
        factors = {
            0: {"c1": {**blank(), "instantiated": True}, "c2": blank(), "c3": blank()},
            1: {"c1": blank(), "c2": {**blank(), "instantiated": True}, "c3": blank()},
            2: {"c1": blank(), "c2": blank(), "c3": {**blank(), "evidence_image_indices": [1]}},
        }
        self.assertEqual(infer_proxy_dec_roles(graph, factors), {
            "c1": "PREREQUISITE_ONLY",
            "c2": "PREREQUISITE_ONLY",
            "c3": "DEC_REQUIRED",
        })

    def test_scene_fold_is_stable(self):
        self.assertEqual(scene_fold("17DRP5sb8fy"), scene_fold("17DRP5sb8fy"))
        self.assertIn(scene_fold("17DRP5sb8fy"), range(5))


if __name__ == "__main__":
    unittest.main()
