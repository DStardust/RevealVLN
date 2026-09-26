"""A late token must never see later physical observations or promoted labels."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recapture_plan import requests_from_rows
from token_index import action_rows


def rows(kind='PRESERVATION', cutoff=0):
    trace = dict(id=9, partition='FIT', kind=kind, cutoff=cutoff,
                 actions=[1, 2, 1, 3, 1, 2, 0], query_steps=[0, 4],
                 rgb_sha256=['rgb' + str(i) for i in range(7)])
    return list(action_rows(trace, 2))


class RecapturePlanTests(unittest.TestCase):
    def test_later_physical_features_do_not_enter_chunk_actor(self):
        request = requests_from_rows(rows())[0]
        self.assertEqual(request['memory_replay_exclusive_end'], 1)
        self.assertEqual([a['allowed_memory_feature_index'] for a in request['actions']], [0] * 4)
        self.assertEqual([a['physical_feature_index_audit_only'] for a in request['actions']], [0, 1, 2, 3])
        self.assertEqual(request['actions'][3]['autoregressive_prefix_action_ids'], [1, 2, 1])

    def test_new_query_only_uses_its_own_available_prefix(self):
        request = requests_from_rows(rows())[1]
        self.assertEqual(request['query_start_step'], 4)
        self.assertEqual(request['memory_replay_exclusive_end'], 5)
        self.assertTrue(all(a['allowed_memory_feature_index'] == 4 for a in request['actions']))

    def test_stop_has_no_new_observation(self):
        stop = requests_from_rows(rows())[1]['actions'][-1]
        self.assertTrue(stop['terminal_stop'])
        self.assertIsNone(stop['physical_next_feature_index_audit_only'])
        self.assertEqual(stop['next_observation_status'], 'TERMINAL_STOP_NO_NEW_OBSERVATION')

    def test_original_known_masks_are_not_promoted(self):
        source = rows('RECOVERY', 4)
        plans = requests_from_rows(source)
        planned = [action for request in plans for action in request['actions']]
        self.assertEqual([a['original_existing_actor_supervision_known'] for a in planned],
                         [r['existing_actor_supervision_known'] for r in source])
        self.assertEqual([a['original_action_in_registered_supervised_region'] for a in planned],
                         [r['action_in_registered_supervised_region'] for r in source])
        self.assertTrue(planned[5]['original_action_in_registered_supervised_region'])
        self.assertFalse(planned[5]['original_existing_actor_supervision_known'])

    def test_false_later_actor_reuse_is_rejected(self):
        source = rows()
        source[1]['actor_feature_index'] = 0
        with self.assertRaises(ValueError):
            requests_from_rows(source)

    def test_query_with_no_missing_actor_needs_no_capture(self):
        trace = dict(id=1, partition='FIT', kind='RECOVERY', cutoff=0,
                     actions=[1, 0], query_steps=[0, 1], rgb_sha256=['a', 'b'])
        self.assertEqual(requests_from_rows(list(action_rows(trace, 0))), [])


if __name__ == '__main__':
    unittest.main()
