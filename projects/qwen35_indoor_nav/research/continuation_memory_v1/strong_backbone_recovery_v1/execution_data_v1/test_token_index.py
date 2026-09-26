"""CPU checks for token boundaries, causal transitions and missing features."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_index import action_rows


def trajectory(actions, queries, kind='PRESERVATION', cutoff=0):
    return dict(id=7, partition='FIT', kind=kind, cutoff=cutoff, actions=actions,
                query_steps=queries, rgb_sha256=['rgb' + str(i) for i in range(len(actions))])


class TokenIndexTests(unittest.TestCase):
    def test_late_stop_does_not_copy_first_actor(self):
        rows = list(action_rows(trajectory([1, 2, 1, 0], [0]), 9))
        self.assertEqual([r['action_token_offset'] for r in rows], [0, 1, 2, 3])
        self.assertEqual([r['actor_feature_index'] for r in rows], [0, None, None, None])
        self.assertTrue(rows[-1]['requires_actor_recapture'])
        self.assertIsNone(rows[-1]['next_memory_feature_index'])
        self.assertIsNone(rows[-1]['next_rgb_sha256'])
        self.assertFalse(rows[-1]['has_observed_transition'])

    def test_recovery_cutoff_keeps_failed_prefix_unknown(self):
        rows = list(action_rows(trajectory([1, 2, 1, 1, 2, 0], [0, 4, 5], 'RECOVERY', 4), 0))
        self.assertEqual([r['existing_actor_supervision_known'] for r in rows], [False] * 4 + [True, True])
        self.assertEqual(rows[4]['query_index'], 1)
        self.assertEqual(rows[5]['actor_feature_index'], 2)
        self.assertFalse(rows[5]['requires_actor_recapture'])

    def test_transition_uses_executed_action_and_next_observation(self):
        rows = list(action_rows(trajectory([3, 1, 0], [0]), 2))
        self.assertIsNone(rows[0]['previous_executed_action'])
        self.assertEqual(rows[0]['executed_action'], 3)
        self.assertEqual(rows[0]['next_memory_feature_index'], 1)
        self.assertEqual(rows[1]['previous_executed_action'], 3)
        self.assertEqual(rows[1]['next_rgb_sha256'], 'rgb2')

    def test_interior_stop_is_rejected(self):
        with self.assertRaises(ValueError):
            list(action_rows(trajectory([1, 0, 1, 0], [0]), 0))

    def test_missing_query_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            list(action_rows(trajectory([1, 1, 1, 1, 1, 0], [0]), 0))
        with self.assertRaises(ValueError):
            list(action_rows(trajectory([1, 0], [1]), 0))

    def test_stop_is_inside_original_500_decision_budget(self):
        rows = list(action_rows(trajectory([1] * 499 + [0], list(range(0, 500, 4))), 0))
        self.assertEqual(len(rows), 500)
        self.assertEqual(sum(r['has_observed_transition'] for r in rows), 499)
        with self.assertRaises(ValueError):
            list(action_rows(trajectory([1] * 500 + [0], list(range(0, 501, 4))), 0))


if __name__ == '__main__':
    unittest.main()
