import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('scout_next_prepare', Path(__file__).with_name('prepare.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PreparationTests(unittest.TestCase):
    def rows(self):
        return [dict(house_id=h, house_candidate_rank=rank, split='FIT')
                for rank in (0, 1) for h in ['old', 'a', 'b', 'c', 'd', 'e', 'f', 'g']]

    def test_exclusion_first_encounter(self):
        rows = self.rows()
        chosen = m.choose(rows, {r['house_id'] for r in rows}, excluded=['old'])
        self.assertEqual([r['house_id'] for r in chosen], list('abcdef'))

    def test_no_input_mutation(self):
        rows = self.rows()
        old = copy.deepcopy(rows)
        out = m.choose(rows, {r['house_id'] for r in rows}, excluded=['old'])
        out[0]['house_id'] = 'changed'
        self.assertEqual(old, rows)

    def test_shortage_fails(self):
        with self.assertRaises(ValueError):
            m.choose(self.rows(), set('abcdefg') | {'old'}, excluded=['old'], limit=8)

    def test_non_fit_fails_closed(self):
        with self.assertRaises(AssertionError):
            m.choose(self.rows(), set('abcdefg'), excluded=['old'])

    def test_positions_only_requested_house(self):
        e = [dict(scene_id='mp3d/a/a.glb', start_position=[1, 2, 3], reference_path=[[1, 2, 3], [0, 2, 3]]),
             dict(scene_id='mp3d/b/b.glb', start_position=[99, 99, 99], reference_path=[])]
        points, count = m.source_positions(e, 'a')
        self.assertEqual(points, [[0, 2, 3], [1, 2, 3]])
        self.assertEqual(count, 1)

    def test_missing_source_fails(self):
        with self.assertRaises(AssertionError):
            m.source_positions([], 'a')


if __name__ == '__main__':
    unittest.main()
