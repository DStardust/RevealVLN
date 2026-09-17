from pathlib import Path
import sys
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import transport
class Tests(unittest.TestCase):
    def test_previous_actual_zero_data(self):
        r=transport.previous_evidence();self.assertEqual(set(r['zero_actual_route_evidence']),{'2','4'})
        self.assertAlmostEqual(r['prior_wall_seconds'],12.349006931995973)
    def test_new_source_root(self):
        self.assertEqual(c.LANES,{6:(2,4)})
        self.assertEqual(transport.RESCUE,HERE.parent/'auto_generation_v1/source_v1')
        for s in (2,4):self.assertEqual(c.shard_root(s),HERE/'production'/f'shard_{s:04d}')
    def test_budget_charged(self):
        source=transport.run_source();compile(source,'CPU_run','exec')
        self.assertIn("['prior_wall_seconds']",source);self.assertIn("['prior_worker_seconds']",source)
        self.assertNotIn("'sleep 24000')",source)
    def test_prepare_compiles(self):compile(transport.prepare_source(),'CPU_prepare','exec')
    def test_exact_approval_rejects_extra_metadata(self):
        expected=dict(approved=True,gpu=6,shards=[2,4],input_lock_sha256='x',identity_sha256='y')
        with mock.patch.object(c,'immutable_verify',return_value={}),mock.patch.object(c,'approval_value',return_value=expected),mock.patch.object(c,'read',return_value=dict(expected,reviewer='main')):
            with self.assertRaisesRegex(AssertionError,'MAIN_APPROVAL_REQUIRED'):c.approved(6)
    def test_exact_approval_accepts_only_five_fields(self):
        expected=dict(approved=True,gpu=6,shards=[2,4],input_lock_sha256='x',identity_sha256='y')
        with mock.patch.object(c,'immutable_verify',return_value={'file':'sha'}),mock.patch.object(c,'approval_value',return_value=expected),mock.patch.object(c,'read',return_value=expected):
            self.assertEqual(c.approved(6),{'file':'sha'})
if __name__=='__main__':unittest.main(verbosity=2)
