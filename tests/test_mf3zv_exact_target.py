import unittest

from revealnav_mf3.progress_target_support import exact_target_from_trace_row


class Mf3zvExactTargetTest(unittest.TestCase):
    def test_exact_native_target(self):
        target = exact_target_from_trace_row(
            dataset="RxR",
            episode_id="1",
            scene_id="scene",
            row={
                "step": 3,
                "current_local_action_ids": ["g1", "g2"],
                "native_action_id": "g2",
                "public_unseen_authorized": False,
            },
            source_sha256="a" * 64,
        )
        self.assertEqual(target.native_action_id, "g2")

    def test_target_must_be_in_dynamic_set(self):
        with self.assertRaises(ValueError):
            exact_target_from_trace_row(
                dataset="RxR",
                episode_id="1",
                scene_id="scene",
                row={
                    "step": 3,
                    "current_local_action_ids": ["g1"],
                    "native_action_id": "g2",
                },
                source_sha256="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()

