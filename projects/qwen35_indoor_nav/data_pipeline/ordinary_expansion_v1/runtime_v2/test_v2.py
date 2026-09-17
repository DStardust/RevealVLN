import ast
import copy
import json
from pathlib import Path
import sys
import time
import unittest
from unittest import mock
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c
import transport as t
import prepare
import run
import telemetry

def function(source, name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name))

class Tests(unittest.TestCase):
    def test_01_transports_compile(self):
        for f in (t.common_source, t.run_source, t.merge_source, t.sentinel_source):
            compile(f(), 'transport', 'exec')
        for s in c.SELECTED_SHARDS:
            compile(c.worker_source(s), 'worker', 'exec')
            self.assertIn(f'cfg.gpu_device_id={3+s%3};', c.worker_source(s))
            self.assertEqual(c.shard_root(s).parent, t.DATA)
            self.assertFalse(c.shard_root(s).is_relative_to(HERE))

    def test_02_partition_and_live_unattempted(self):
        jobs, shards = prepare.source_checks()
        self.assertEqual(sum(map(len, shards)), 9661)
        self.assertEqual(prepare.partition(jobs), shards)
        self.assertEqual(len(shards), 42)
        self.assertEqual([len(s) for s in shards[:3]], [3,3,3])
        self.assertTrue(all(len({j['scene_id'] for j in s}) == 3 for s in shards[:3]))
        self.assertEqual(sorted(s for lane in c.LANES.values() for s in lane), list(range(42)))
        self.assertEqual(max(map(len, shards)), 250)

    def test_03_duplicate_rejected(self):
        jobs = c.read(t.REC / 'RESCUE_JOBS.json'); jobs[1] = jobs[0]
        with self.assertRaises(AssertionError): prepare.partition(jobs)

    def test_04_bad_shard_rejected(self):
        for s in (-1, 42, True, 3.0):
            with self.assertRaises(AssertionError): c.shard_root(s)

    def test_05_quality_and_merge_validation_unchanged(self):
        self.assertEqual(function(t.sealed('sentinel_gate.py'), 'inspect'), function(t.sentinel_source(), 'inspect'))
        self.assertEqual(function(t.original('merge.py'), 'validate_accepted'), function(t.merge_source(), 'validate_accepted'))

    def test_06_external_and_drain_guards_unchanged(self):
        for name in ('contexts','drain_context','parse_gpu','stop_own','wait_holder_exit','wait_pane_dead'):
            self.assertEqual(function(t.original('run.py'), name), function(t.run_source(), name))

    def test_07_no_approval_no_gpu(self):
        with mock.patch.object(c, 'approved', side_effect=AssertionError('approval')), mock.patch.object(run, 'gpu_snapshot') as gpu:
            with self.assertRaisesRegex(AssertionError, 'approval'): run.execute(3)
            gpu.assert_not_called()

    def test_08_tail_budget(self):
        with mock.patch.object(c, 'save') as save:
            run.require_phase_budget(43200, 40080, 3000, HERE, 39)
            save.assert_not_called()
            with self.assertRaisesRegex(AssertionError, 'INSUFFICIENT_FULL_PHASE'):
                run.require_phase_budget(43200, 40080.1, 3000, HERE, 39)

    def test_09_no_hidden_retry_or_temporary_sleeper(self):
        s = t.run_source()
        self.assertNotIn("'sleep 24000'", s)
        self.assertNotIn("'respawn-pane','-k'", s)
        self.assertIn('NEW_ENVDROP_LANE_NO_AUTOMATIC_RETRY', s)
        self.assertIn('FULL_BATCH_REQUIRED_NO_PARTIAL_PROMOTION', t.merge_source())
        self.assertIn('if shard!=c.LANES[gpu][0]:require_sentinel_receipt(out,gpu)', s)

    def test_10_sentinel_binding(self):
        value = dict(strict_pass=True, strict_routes=3, source_jobs_sha256='old', input_lock_sha256='old', input_hashes={})
        with mock.patch.object(c,'read',return_value=value), mock.patch.object(c,'sha',return_value='changed'):
            for g in c.LANES:
                with self.assertRaises(AssertionError): run.require_sentinel_receipt(HERE,g)

    def test_11_restoration_environment(self):
        value = dict(project_cache_environment={'TMPDIR': str(HERE / 'cache')}, cmdline=['python', 'holder.py'])
        self.assertEqual(run.restore_argv(value), ['env', 'TMPDIR='+str(HERE/'cache'), 'python', 'holder.py'])
        value['project_cache_environment'] = {}
        self.assertEqual(run.restore_argv(value), value['cmdline'])
        self.assertIn("project_environment(pid)==identity['project_cache_environment']", t.run_source())

    def test_12_active_accounting_preserves_raw(self):
        snap = dict(memory_mib=733, processes={1:246, 2:246, 3:245})
        old=copy.deepcopy(snap); records=[]
        result=telemetry.active_assessment(snap,3,run.contexts,records.append)
        self.assertEqual(old,snap)
        self.assertEqual(result['conservative_upper_mib'],245)
        self.assertTrue(result['amended_acceptance'])

    def test_13_active_overage_and_restore_fail_closed(self):
        for total, processes in [(4095,{1:246,3:4000}),(500,{1:769,3:1}),(-1,{3:1}),(float('nan'),{3:1})]:
            with self.assertRaises(AssertionError):
                telemetry.active_assessment(dict(memory_mib=total,processes=processes),3,run.contexts,lambda x:None)
        with self.assertRaises(AssertionError):
            run.contexts(dict(memory_mib=733,processes={1:246,2:246,3:245}),restorable=True)

    def test_14_monitor_targets(self):
        import monitor
        value=monitor.collect()
        self.assertEqual(sum(l['target'] for l in value['lanes']),9661)
        self.assertTrue(value['read_only'])
        self.assertEqual(value['training_status'],'STOPPED')

    def test_15_phase_and_disk_budgets(self):
        self.assertLess(600+13*3000+120,43200)
        self.assertEqual(42*12+8,512)
        self.assertIn('<11*1024**3',t.run_source())
        self.assertIn('shard_limits[shard]-120',t.run_source())

def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt=dict(passed=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        time_unix=time.time(), code_sha256={p.name:c.sha(p) for p in HERE.glob('*.py')}, GPU_contexts_created=0)
    c.save(HERE/'CPU_TESTS_FINAL.json',receipt)
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=='__main__': main()
