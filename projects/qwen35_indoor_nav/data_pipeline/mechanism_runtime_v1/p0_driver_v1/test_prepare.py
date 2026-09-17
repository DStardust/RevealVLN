import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import prepare_batch as p


class PrepareTests(unittest.TestCase):
    def fixture(self):
        ep = {'scene_id': 'mp3d/house/house.glb', 'start_position': [0, 0, 0],
              'start_rotation': [0, 0, 0, 1], 'reference_path': [[0, 0, 0], [1, 0, 0]],
              'goals': [{'position': [1, 0, 0]}], 'instruction': 'unused'}
        row = {'house_id': 'house', 'source_physical_route_sha256': p.physical_hash(ep)}
        return ep, row

    def test_alias_not_new_candidate(self):
        ep, row = self.fixture()
        second = dict(ep, instruction='different')
        with patch.object(p.planning, 'select_candidates', return_value=[row]):
            result = p.resolve([], [ep, second])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['resolved_instruction_alias_count'], 2)
        configs = result[0]['configurations']
        self.assertEqual(len(configs), 16)
        self.assertEqual([c['u_position'] for c in configs[:2]], [[0., 0., 0.], [1., 0., 0.]])
        self.assertEqual(configs[2]['yaw_bin'], 12)
        self.assertEqual(configs[8]['public_tail'], 'FFFFFFFF')

    def test_changed_physical_route_denied(self):
        ep, row = self.fixture()
        ep['start_position'] = [0, 0, .1]
        with patch.object(p.planning, 'select_candidates', return_value=[row]), self.assertRaises(AssertionError):
            p.resolve([], [ep])

    def test_reserved_metadata_does_not_resolve(self):
        ep, row = self.fixture()
        ep['scene_id'] = 'mp3d/reserved/reserved.glb'
        with patch.object(p.planning, 'select_candidates', return_value=[row]), self.assertRaises(AssertionError):
            p.resolve([], [ep])

    def test_supervisor_denies_unapproved_before_gpu(self):
        import supervisor
        with self.assertRaises(AssertionError):
            supervisor.verify_admission(HERE, {}, {'approved': False})

    def test_real_prepared_order(self):
        config = json.loads((HERE.parent/'p0_batch_v1/CONFIG_DRAFT.json').read_text())
        self.assertFalse(config['runtime_allowed'])
        self.assertEqual([r['candidate_id'] for r in config['candidates']], ['MP5_%02d' % i for i in range(5)])
        self.assertEqual([len(r['configurations']) for r in config['candidates']], [56, 40, 48, 48, 40])
        for row in config['candidates']:
            self.assertEqual(len(row['assets']), 4)
            self.assertTrue(all(Path(k).resolve().is_relative_to(p.ROOT) for k in row['assets']))


if __name__ == '__main__':
    unittest.main()
