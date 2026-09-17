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
    def test_scope_root_and_shard(self):
        self.assertEqual(c.LANES,{3:(6,)})
        self.assertEqual(c.SELECTED_SHARDS,(6,))
        self.assertEqual(c.shard_root(6),HERE.parent/'production/shard_0006')
        self.assertIn('cfg.gpu_device_id=3;',c.worker_source(6))
    def test_no_sleep_creation_or_legacy_transfer(self):
        source=transport.run_source();compile(source,'CPU_run','exec')
        self.assertNotIn("'sleep 24000')",source)
        self.assertNotIn('sleeper_identity=wait_stable_sleeper(identity)',source)
        self.assertNotIn('claim_legacy_lane',source)
        self.assertIn('sleeper_identity=None',source)
    def test_original_worker_lock_no_bypass(self):
        source=transport.original('worker.py')
        self.assertIn("c.read(root/'INPUT_LOCK.json')==lock",source)
    def test_prepare_fresh_only(self):
        source=transport.prepare_source();compile(source,'CPU_prepare','exec')
        self.assertIn('SHARD6_MUST_BE_NEVER_ATTEMPTED',source)
        self.assertIn('root.mkdir(parents=True,exist_ok=False)',source)
        self.assertIn("c.save(c.shard_root(shard)/'INPUT_LOCK.json',lock)",source)
    def test_frozen_budget_transport(self):
        auth=dict(cleanup_margin_seconds=120,shard_wall_seconds={'6':14400},lane_wall_seconds={'3':14520},max_wall_seconds_per_gpu_chain=14520)
        self.assertEqual(transport.budget_from_authorization(auth,{3:(6,)}),({'6':14400},{'3':14520}))
        source=transport.run_source()
        self.assertEqual(source.count('<shard_limits[shard]-120'),2)
        self.assertEqual(source.count('<lane_limit-120'),3)
    def test_dead_pane_restoration_no_signal(self):
        identity=dict(pid=1,pane_id='%1',pane_target='own:0.0',cwd=str(c.ROOT),proc_uid=0,cmdline=['python','holder'])
        new=dict(pid=3,starttime_ticks=5,proc_uid=0,cwd=identity['cwd'],cmdline=identity['cmdline'])
        def pane(identity,fmt):return {'#{pane_id}':'%1','#{pane_dead}':'1','#{pane_pid}':'3'}[fmt]
        with mock.patch.object(run,'drain_context',return_value={}),mock.patch.object(c,'save'),mock.patch.object(run,'pane',side_effect=pane),mock.patch.object(run,'call') as call,mock.patch.object(run,'process_identity',return_value=new),mock.patch.object(run,'gpu_snapshot',return_value={'processes':{3:29000}}),mock.patch.object(run.time,'sleep'),mock.patch.object(run.os,'kill') as kill:
            result=run.restore_holder(identity,None,'off',HERE)
        self.assertTrue(result['restored']);kill.assert_not_called()
        self.assertFalse(any('-k' in args.args for args in call.call_args_list))
        self.assertEqual(call.call_args_list[-1].args[-1],'off')
    def test_external_pane_replacement_refused(self):
        identity=dict(pid=1,pane_id='%1',pane_target='own:0.0',cwd=str(c.ROOT),proc_uid=0,cmdline=['python','holder'])
        def pane(identity,fmt):return {'#{pane_id}':'%1','#{pane_dead}':'0'}[fmt]
        with mock.patch.object(run,'drain_context',return_value={}),mock.patch.object(c,'save'),mock.patch.object(run,'pane',side_effect=pane),mock.patch.object(run,'call') as call,mock.patch.object(run.time,'sleep'),mock.patch.object(run.os,'kill') as kill:
            with self.assertRaisesRegex(AssertionError,'PANE_DEAD_TRANSITION_TIMEOUT'):run.restore_holder(identity,None,'off',HERE)
        kill.assert_not_called();call.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
