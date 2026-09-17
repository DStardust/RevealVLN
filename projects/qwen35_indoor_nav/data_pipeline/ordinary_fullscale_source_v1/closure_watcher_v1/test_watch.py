from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import watch as w


class Tests(unittest.TestCase):
    def fixture(self):
        restore=dict(restored=True,new_identity=dict(pid=123),gpu=dict(processes={'123':29000}))
        return dict(error=None,workers=[dict(shard=2,returncode=0),dict(shard=4,returncode=0)],restoration=restore),restore
    def test_closed_success(self):w.ready(*self.fixture(),[2,4])
    def test_failure_never_merges(self):
        r,s=self.fixture();r['error']='resource censor'
        with self.assertRaises(AssertionError):w.ready(r,s,[2,4])
    def test_missing_shard_rejected(self):
        r,s=self.fixture();r['workers'].pop()
        with self.assertRaises(AssertionError):w.ready(r,s,[2,4])
    def test_unrestored_rejected(self):
        r,s=self.fixture();s['restored']=False
        with self.assertRaises(AssertionError):w.ready(r,s,[2,4])
    def test_insufficient_memory_evidence_rejected(self):
        r,s=self.fixture();s['gpu']['processes']['123']=512
        with self.assertRaises(AssertionError):w.ready(r,s,[2,4])
    def test_only_approved_four_lanes(self):
        self.assertEqual(set(w.LANES),{3,4,6,7})
        self.assertEqual(w.LANES[6][3],'runtime_v3_gpu6_merge_v1/merge.py')
        self.assertEqual(w.LANES[4][1],'rescue_production')


if __name__=='__main__':unittest.main(verbosity=2)
