import unittest

from revealnav_mf3.oracle_headroom_metrics import scene_cluster_bootstrap


class SceneBootstrapTest(unittest.TestCase):
    def test_clusters_by_scene(self):
        rows = [
            {"scene_id": "s1", "v": 1.0}, {"scene_id": "s1", "v": 3.0},
            {"scene_id": "s2", "v": 5.0}, {"scene_id": "s2", "v": 7.0},
        ]
        value = scene_cluster_bootstrap(rows, lambda values: sum(item["v"] for item in values) / len(values))
        self.assertEqual(value["cluster"], "raw_mp3d_scene")
        self.assertEqual(value["scene_count"], 2)
        self.assertEqual(value["replicates"], 10000)


if __name__ == "__main__":
    unittest.main()
