from pathlib import Path
import sys
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport
import common as c
import run


class Tests(unittest.TestCase):
    def identity(self):
        return dict(pid=1,pane_id='%1',pane_target='own:0.0',cwd=str(c.ROOT),proc_uid=0,cmdline=['python','holder'])
    def test_dead_pane_success_no_sleep_signal(self):
        i=self.identity();new=dict(pid=3,starttime_ticks=5,proc_uid=0,cwd=i['cwd'],cmdline=i['cmdline'])
        def pane(identity,fmt):return {'#{pane_id}':'%1','#{pane_dead}':'1','#{pane_pid}':'3'}[fmt]
        with mock.patch.object(run,'drain_context',return_value={}),mock.patch.object(run.c,'save'),mock.patch.object(run,'pane',side_effect=pane),mock.patch.object(run,'call') as call,mock.patch.object(run,'process_identity',return_value=new),mock.patch.object(run,'gpu_snapshot',return_value={'processes':{3:29000}}),mock.patch.object(run.time,'sleep'),mock.patch.object(run.os,'kill') as kill:
            result=run.restore_holder(i,None,'off',HERE)
        self.assertTrue(result['restored']);kill.assert_not_called()
        self.assertFalse(any('-k' in args.args for args in call.call_args_list))
        self.assertEqual(call.call_args_list[-1].args[-1],'off')
    def test_unknown_live_pane_is_not_killed_or_respawned(self):
        def pane(identity,fmt):return {'#{pane_id}':'%1','#{pane_dead}':'0'}[fmt]
        with mock.patch.object(run,'drain_context',return_value={}),mock.patch.object(run.c,'save'),mock.patch.object(run,'pane',side_effect=pane),mock.patch.object(run,'call') as call,mock.patch.object(run.time,'sleep'),mock.patch.object(run.os,'kill') as kill:
            with self.assertRaisesRegex(AssertionError,'PANE_DEAD_TRANSITION_TIMEOUT'):
                run.restore_holder(self.identity(),None,'off',HERE)
        kill.assert_not_called();call.assert_not_called()
    def test_changed_pane_id_rejects(self):
        with mock.patch.object(run,'drain_context',return_value={}),mock.patch.object(run.c,'save'),mock.patch.object(run,'pane',return_value='%foreign'),mock.patch.object(run,'call') as call:
            with self.assertRaises(AssertionError):run.restore_holder(self.identity(),None,'off',HERE)
        call.assert_not_called()
    def test_no_temporary_process_source(self):
        source=transport.run_source();compile(source,'CPU_run','exec')
        self.assertNotIn("'sleep 24000')",source)
        self.assertNotIn('sleeper_identity=wait_stable_sleeper(identity)',source)
        self.assertIn('sleeper_identity=None',source)
        self.assertIn('legacy_lane_lock=c.claim_legacy_lane()',source)
        self.assertIn("previous_failed_wall_seconds",source)
    def test_worker_two_locks_metadata_unchanged(self):
        source=transport.worker_entry_source();compile(source,'CPU_worker','exec')
        self.assertIn("c.sha(root/'INPUT_LOCK.json')==c.LEGACY_ROOT_LOCK",source)
        self.assertIn('lock=c.approved(c.lane_for(shard))',source)
        source=transport.prepare_source();compile(source,'CPU_prepare','exec')
        self.assertNotIn("c.save(c.shard_root(shard)/'INPUT_LOCK.json',lock)",source)
        self.assertIn("assert c.read(root/name)==value",source)
    def test_actual_preworker_handoff(self):
        if (HERE/'INPUT_LOCK.json').exists():self.skipTest('already frozen; later production may exist')
        h=transport.inspect_preworker_handoff(c)
        self.assertEqual(h['shards'],[3,5]);self.assertEqual(h['gpu'],7)
        self.assertTrue(h['old_failure_preserved']);self.assertGreater(h['previous_failed_wall_seconds'],0)
    def test_gpu7_only_roots_and_limits(self):
        self.assertEqual(c.LANES,{7:(3,5)})
        self.assertEqual(c.PARALLEL,HERE.parent)
        for s in [3,5]:
            source=c.worker_source(s);compile(source,'CPU_physical_worker','exec')
            self.assertIn('cfg.gpu_device_id=7;',source)
        self.assertIn('<shard_limits[shard]-120',transport.run_source())


if __name__=='__main__':unittest.main()
