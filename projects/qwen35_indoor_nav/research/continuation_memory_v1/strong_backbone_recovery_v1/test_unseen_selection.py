import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_unseen import pick
import common as u


class SelectionTests(unittest.TestCase):
    def test_stable_house_split_and_no_route_leakage(self):
        rows=[dict(scene_id=f'mp3d/{h}/{h}.glb',trajectory_id=r,episode_id=f'{h}_{r}_{i}') for h in ('a','b','c') for r in range(10) for i in range(3)]
        chosen,audit=pick(rows,9,1209,{'c'},{('a','0')})
        reverse,_=pick(list(reversed(rows)),9,1209,{'c'},{('a','0')})
        self.assertEqual(chosen,reverse)
        keys={(r['scene_id'].split('/')[-2],str(r['trajectory_id'])) for r in chosen}
        self.assertEqual(len(keys),9);self.assertNotIn(('a','0'),keys)
        self.assertEqual({h for h,_ in keys},{'a','b'})
        self.assertEqual(sum(audit['house_quotas'].values()),9)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelectionTests))
    u.write(u.HERE/'UNSEEN_SELECTION_CPU_TEST_RESULT.json',dict(tests=result.testsRun,successful=result.wasSuccessful()))
    raise SystemExit(not result.wasSuccessful())
