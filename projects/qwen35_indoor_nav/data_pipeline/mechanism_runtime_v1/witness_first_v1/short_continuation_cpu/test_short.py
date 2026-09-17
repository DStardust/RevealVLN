import copy
import json
import unittest
from prepare import WF,protected_compress,equivalent,proposals


class ShortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=WF/'revisit_v1/run_v1'
        cls.row=json.loads((root/'EXECUTION_CONFIG.json').read_text())['candidates'][0]
        cls.trace=json.loads((root/'bundles/WF_REVISIT_004/traces/000001.json').read_text())

    def test_boundary_protection(self):
        a=list('LLRRFLR')
        for t in range(1,len(a)+1):
            c,pair=protected_compress(a,t)
            self.assertEqual(c[pair[0]],a[t-1])
            self.assertTrue(equivalent(a[:t-1],c[:pair[0]]))
            self.assertTrue(equivalent(a[:t],c[:pair[1]]))

    def test_invalid_step(self):
        with self.assertRaises(ValueError):protected_compress(['F'],0)

    def test_source_unmodified(self):
        saved=copy.deepcopy(self.trace); actions=list(self.row['components']['a'])
        proposals(actions,self.trace,[5]);self.assertEqual(saved,self.trace)
        self.assertEqual(actions,self.row['components']['a'])

    def test_real_candidates_bounded_and_short(self):
        rows=proposals(self.row['components']['a'],self.trace,[5])
        self.assertEqual([r['motion_actions'] for r in rows],[34,36,44])
        self.assertEqual(len({tuple(r['actions']) for r in rows}),3)

    def test_real_witness_protected_and_not_physically_claimed(self):
        for row in proposals(self.row['components']['a'],self.trace,[5]):
            self.assertGreaterEqual(row['protected_min_pixels'],256)
            self.assertEqual(row['protected_source_frames'][1]-row['protected_source_frames'][0],1)
            self.assertEqual(row['protected_new_frames'][1]-row['protected_new_frames'][0],1)
            self.assertIsNone(row['actual_event_preserved'])
            self.assertIsNone(row['physical_replay_pass'])

    def test_no_evidence_no_candidate(self):
        self.assertEqual(proposals(self.row['components']['a'],self.trace,[999999]),[])

    def test_unknown_evidence_rejected(self):
        trace=copy.deepcopy(self.trace)
        for o in trace['observations']:o['evidence_complete']=False
        self.assertEqual(proposals(self.row['components']['a'],trace,[5]),[])

    def test_collision_source_denied(self):
        trace=copy.deepcopy(self.trace);trace['collisions']=1
        with self.assertRaises(ValueError):proposals(self.row['components']['a'],trace,[5])

    def test_deterministic(self):
        self.assertEqual(proposals(self.row['components']['a'],self.trace,[5]),proposals(self.row['components']['a'],self.trace,[5]))


if __name__=='__main__':unittest.main()
