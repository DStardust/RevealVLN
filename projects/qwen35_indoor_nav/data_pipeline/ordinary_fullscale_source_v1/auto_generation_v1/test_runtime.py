from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport
import common as c
class Tests(unittest.TestCase):
    def test_real_pending_subset(self):
        rows=transport.rescue_jobs();self.assertEqual({s:len(js) for s,js in rows.items()},{2:426,4:998})
    def test_new_output_roots_gpu6(self):
        self.assertEqual(c.LANES,{6:(2,4)})
        for s in (2,4):
            self.assertEqual(c.shard_root(s),HERE/'production'/f'shard_{s:04d}')
            self.assertIn('cfg.gpu_device_id=6;',c.worker_source(s))
    def test_budget_and_guard_unchanged(self):
        source=transport.run_source();compile(source,'CPU_run','exec')
        self.assertIn("'GPU_MEMORY_ACCOUNTING'",source)
        self.assertNotIn("'sleep 24000')",source)
        self.assertIn('old_source_locks=c.claim_original_source_lanes()',source)
    def test_prepare_source_compiles(self):
        source=transport.prepare_source();compile(source,'CPU_prepare','exec')
        self.assertIn("transport.RESCUE/'INPUT_LOCK.json'",source)
        self.assertIn("{'2':426,'4':998}",source)
    def test_old_gpu6_lane_lock_only(self):
        source=transport.common_source()
        self.assertIn('for gpu in (6,):',source)
        self.assertIn("PARALLEL/'runtime_v3/lanes'",source)
if __name__=='__main__':unittest.main(verbosity=2)
