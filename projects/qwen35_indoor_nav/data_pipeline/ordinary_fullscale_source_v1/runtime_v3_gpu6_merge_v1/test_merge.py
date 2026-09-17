import copy
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import merge as m


class Tests(unittest.TestCase):
    def fixture(self):
        h=m.read(m.V4/'PREWORKER_OWNERSHIP.json')
        r=m.read(m.V3/'lanes/gpu_7/attempt_000/RESULT.json')
        cfg=m.read(m.V4/'PREPARED_CONFIG.json');approval=m.read(m.V4/'MAIN_AGENT_APPROVAL_GPU7.json')
        return h,r,cfg,approval
    def test_real_frozen_handoff_only_no_active_state(self):m.validate_handoff(*self.fixture())
    def test_old_gpu7_actual_worker_rejected(self):
        h,r,cfg,a=self.fixture();r['workers']=[dict(shard=3,returncode=0)]
        with self.assertRaisesRegex(AssertionError,'ACTUAL_WORKERS'):m.validate_handoff(h,r,cfg,a)
    def test_changed_ownership_rejected(self):
        h,r,cfg,a=self.fixture();h['shards']=[2,4]
        with self.assertRaises(AssertionError):m.validate_handoff(h,r,cfg,a)
    def test_old_restoration_cannot_be_rewritten(self):
        h,r,cfg,a=self.fixture();r['restoration']['restored']=True
        with self.assertRaisesRegex(AssertionError,'MUST_REMAIN'):m.validate_handoff(h,r,cfg,a)
    def test_unapproved_v4_rejected(self):
        h,r,cfg,a=self.fixture();a['approved']=False
        with self.assertRaises(AssertionError):m.validate_handoff(h,r,cfg,a)
    def test_wrong_v4_source_lock_rejected(self):
        h,r,cfg,a=self.fixture();a['input_lock_sha256']='wrong'
        with self.assertRaises(AssertionError):m.validate_handoff(h,r,cfg,a)
    def closed(self):return dict(error=None,restoration=dict(restored=True),workers=[dict(shard=2,returncode=0),dict(shard=4,returncode=0)])
    def test_gpu6_success_fixture(self):m.validate_gpu6_closed(self.closed())
    def test_gpu6_failed_or_censored_rejected(self):
        r=self.closed();r['error']="AssertionError('SHARD_WALL_BUDGET')"
        with self.assertRaisesRegex(AssertionError,'NOT_SUCCESSFULLY'):m.validate_gpu6_closed(r)
    def test_unrestored_rejected(self):
        r=self.closed();r['restoration']['restored']=False
        with self.assertRaisesRegex(AssertionError,'NOT_RESTORED'):m.validate_gpu6_closed(r)
    def test_missing_second_shard_rejected(self):
        r=self.closed();r['workers']=r['workers'][:1]
        with self.assertRaises(AssertionError):m.validate_gpu6_closed(r)
    def test_exact_adapter_import_only(self):
        module,source=m.build()
        self.assertEqual(module.c.LANES,{6:(2,4)})
        self.assertEqual(module.c.SELECTED_SHARDS,(2,4))
        self.assertEqual(module.HERE,m.V3)
        self.assertIn("out=ADAPTER/'run_v1'",source)
        self.assertIn('validate_accepted(root,jobmap[ident],part,fresh,physical,aliases)',source)
        self.assertNotIn(".open('a')",source)
        self.assertIn('V4_OWNS_3_5',source)


if __name__=='__main__':unittest.main(verbosity=2)
