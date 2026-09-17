import ast
import json
from pathlib import Path
import sys
import time
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import transport as t
import prepare
import run

def fn(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))

class Tests(unittest.TestCase):
    def test_01_live_inventory_and_partition(self):
        jobs,candidates,hashes,proof=prepare.inventory();shards=prepare.partition(jobs)
        self.assertEqual((len(jobs),len(candidates)),(3112,140))
        self.assertEqual(len(shards),64)
        self.assertEqual(len({j['scene_id'] for j in shards[0]}),3)
        self.assertEqual([len(shards[0]),len(shards[-1])],[3,9])
        self.assertLessEqual(max(map(len,shards)),50)
        self.assertEqual({j['job_id'] for s in shards for j in s},{j['job_id'] for j in jobs})
        self.assertNotIn('envdrop_0b264fe96107cd01107c',{j['job_id'] for j in jobs})
        self.assertTrue(proof['other_lanes_disjoint'])
        self.assertGreater(len(hashes),500)

    def test_02_duplicate_rejected(self):
        jobs,*_=prepare.inventory();jobs[1]=jobs[0]
        with self.assertRaises(AssertionError):prepare.partition(jobs)

    def test_03_transports_and_mapping(self):
        for f in (t.common_source,t.run_source,t.merge_source,t.sentinel_source):compile(f(),'transport','exec')
        for s in range(64):
            self.assertEqual(c.lane_for(s),3)
            self.assertEqual(c.shard_root(s).parent,t.DATA)
            self.assertFalse(c.shard_root(s).is_relative_to(HERE))
            source=c.worker_source(s);compile(source,'worker','exec');self.assertIn('cfg.gpu_device_id=3;',source)
        for s in (-1,64,True):
            with self.assertRaises(AssertionError):c.shard_root(s)

    def test_04_quality_and_holder_guards_unchanged(self):
        for name in ('contexts','verify_identity','restore_holder','stop_own','drain_context','active_accounting'):
            self.assertEqual(fn(t.prior.run_source(),name),fn(t.run_source(),name))
        self.assertEqual(fn(t.prior.sentinel_source(),'inspect'),fn(t.sentinel_source(),'inspect'))
        self.assertEqual(fn(t.prior.merge_source(),'validate_accepted'),fn(t.merge_source(),'validate_accepted'))

    def test_05_budget_has_room_and_stops_tail(self):
        jobs,candidates,hashes,proof=prepare.inventory()
        self.assertLess(50*proof['old_route_max_seconds']+180,3480)
        self.assertEqual(64*4+8,264)
        with mock.patch.object(c,'save') as save:
            run.require_phase_budget(43200,39480,3600,HERE,60);save.assert_not_called()
            with self.assertRaisesRegex(AssertionError,'INSUFFICIENT_FULL_PHASE'):
                run.require_phase_budget(43200,39480.1,3600,HERE,60)

    def test_06_no_approval_no_gpu(self):
        with mock.patch.object(c,'approved',side_effect=AssertionError('approval')),mock.patch.object(run,'gpu_snapshot') as gpu:
            with self.assertRaisesRegex(AssertionError,'approval'):run.execute(3)
            gpu.assert_not_called()

    def test_07_no_hidden_retry_or_old_writes(self):
        source=t.run_source()
        self.assertIn('NEW_ENVDROP_LANE_NO_AUTOMATIC_RETRY',source)
        self.assertIn('FULL_BATCH_REQUIRED_NO_PARTIAL_PROMOTION',t.merge_source())
        self.assertNotIn("'sleep 24000'",source)
        self.assertNotIn("'respawn-pane','-k'",source)
        self.assertIn('choices=[3]',source)

    def test_08_sentinel_receipt_binding(self):
        value=dict(strict_pass=True,strict_routes=3,source_jobs_sha256='old',input_lock_sha256='old',input_hashes={})
        with mock.patch.object(c,'read',return_value=value),mock.patch.object(c,'sha',return_value='changed'):
            with self.assertRaises(AssertionError):run.require_sentinel_receipt(HERE,3)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    c.save(HERE/'CPU_TESTS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),unix=time.time(),
        code_sha256={p.name:c.sha(p) for p in HERE.glob('*.py')},gpu_contexts_created=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
