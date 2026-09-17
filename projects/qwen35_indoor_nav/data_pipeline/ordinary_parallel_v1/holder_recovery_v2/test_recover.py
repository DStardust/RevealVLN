import copy
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import recover


class Tests(unittest.TestCase):
    def fixture(self):
        frozen=dict(pid=12,starttime_ticks=34,proc_uid=0,cwd=str(recover.ROOT),cmdline=[''])
        correct=dict(frozen,cmdline=['sleep','24000'])
        identity=dict(pane_id='%1')
        values={'#{pane_id}':'%1','#{pane_pid}':'12','#{pane_dead}':'0','#{pane_current_path}':str(recover.ROOT)}
        ops=types.SimpleNamespace(process_identity=mock.Mock(return_value=correct),
            pane=mock.Mock(side_effect=lambda i,f:values[f]))
        return frozen,correct,identity,ops
    def test_empty_argv_same_identity_corrected_not_mutated(self):
        f,c,i,o=self.fixture()
        with mock.patch.object(recover.time,'sleep'):
            self.assertEqual(recover.stable_sleeper(o,i,f),c)
        self.assertEqual(f['cmdline'],[''])
    def test_pid_start_cwd_uid_rejected(self):
        for key,value in [('pid',13),('starttime_ticks',99),('cwd','/tmp'),('proc_uid',10)]:
            f,c,i,o=self.fixture();o.process_identity.return_value=dict(c,**{key:value})
            with self.assertRaises(AssertionError):recover.stable_sleeper(o,i,f)
    def test_non_sleep_or_still_empty_rejected(self):
        for argv in [[''],['sleep','600'],['python','train.py']]:
            f,c,i,o=self.fixture();o.process_identity.return_value=dict(c,cmdline=argv)
            with self.assertRaises(AssertionError):recover.stable_sleeper(o,i,f)
    def test_pane_change_rejected(self):
        f,c,i,o=self.fixture();o.pane.return_value='%unknown';o.pane.side_effect=None
        with self.assertRaises(AssertionError):recover.stable_sleeper(o,i,f)
    def test_unstable_second_read_rejected(self):
        f,c,i,o=self.fixture();o.process_identity.side_effect=[c,dict(c,starttime_ticks=35)]
        with mock.patch.object(recover.time,'sleep'):
            with self.assertRaises(AssertionError):recover.stable_sleeper(o,i,f)
    def test_restore_failure_preserves_pane_no_fallback_signal(self):
        ops=types.SimpleNamespace(restore_holder=mock.Mock(side_effect=AssertionError('EXTERNAL_GPU_RESOURCE')))
        result=recover.perform_restore(ops,{}, {},'off',HERE)
        self.assertFalse(result['restored']);self.assertTrue(result['recovery_pane_preserved'])
        self.assertEqual(result['external_processes_stopped'],0)
    def test_only_original_restore_called(self):
        ops=types.SimpleNamespace(restore_holder=mock.Mock(return_value={'restored':True}))
        result=recover.perform_restore(ops,{'gpu_device':4},{'pid':1},'off',HERE)
        self.assertTrue(result['restored'])
        ops.restore_holder.assert_called_once_with({'gpu_device':4},{'pid':1},'off',HERE,released_pid=None)
    def test_scope_known_four_and_naturalclose(self):
        self.assertEqual(set(recover.EXPECTED_RUNTIME_LOCKS),{3,4,6,7})
        text=(HERE/'recover.py').read_text()
        self.assertIn("result['error'] is None",text)
        self.assertIn("AssertionError('SLEEPER_IDENTITY_CHANGED')",text)
        self.assertNotIn('os.kill(',text)
        self.assertIn("c.state(shard)['complete']",text)
    def test_original_guard_rejects_external_and_env_module_registered(self):
        with recover.registered_runtime(4) as (c,ops):
            self.assertIn('ordinary_fullscale_source_v1/runtime_v2',str(c.HERE))
            with self.assertRaises(AssertionError):ops.contexts(dict(processes={1:1380},memory_mib=1400),restorable=True)
            self.assertTrue(callable(ops.holder_environment_matches))


if __name__=='__main__':unittest.main()
