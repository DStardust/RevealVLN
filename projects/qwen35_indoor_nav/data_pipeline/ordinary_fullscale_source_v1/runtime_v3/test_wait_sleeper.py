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
    def fixture(self):
        identity=dict(pid=1,pane_id='%1',cwd=str(c.ROOT),proc_uid=0)
        sleeper=dict(pid=2,starttime_ticks=10,proc_uid=0,cwd=str(c.ROOT),cmdline=['sleep','24000'])
        values={'#{pane_id}':'%1','#{pane_pid}':'2','#{pane_dead}':'0'}
        return identity,sleeper,lambda i,k:values[k]
    def run_case(self,values):
        identity,sleeper,pane=self.fixture()
        with mock.patch.object(run,'pane',side_effect=pane),mock.patch.object(run,'process_identity',side_effect=values),mock.patch.object(run.time,'sleep'):
            return run.wait_stable_sleeper(identity)
    def test_empty_then_two_stable_reads(self):
        _,s,_=self.fixture()
        self.assertEqual(self.run_case([dict(s,cmdline=['']),s,s]),s)
    def test_process_not_yet_visible(self):
        _,s,_=self.fixture();self.assertEqual(self.run_case([FileNotFoundError(),s,s]),s)
    def test_still_empty_is_bounded_not_frozen(self):
        _,s,_=self.fixture()
        with self.assertRaisesRegex(AssertionError,'STABILITY_TIMEOUT'):
            self.run_case([dict(s,cmdline=[''])]*80)
    def test_changed_pid_cwd_uid_command(self):
        _,s,_=self.fixture()
        for key,value in [('pid',3),('starttime_ticks',12),('cwd','/tmp'),('proc_uid',4),('cmdline',['python','train.py'])]:
            with self.assertRaises(AssertionError):self.run_case([dict(s,cmdline=['']),dict(s,**{key:value})])
    def test_all_sources_compile_and_roots(self):
        for text in (transport.prepare_source(),transport.common_source(),transport.run_source(),transport.merge_source()):compile(text,'CPU_transport','exec')
        self.assertEqual(c.LANES,{6:(2,4),7:(3,5)})
        self.assertEqual(c.PARALLEL,HERE.parent)
        for s,g in [(2,6),(4,6),(3,7),(5,7)]:
            text=c.worker_source(s);compile(text,'CPU_worker','exec');self.assertIn(f'cfg.gpu_device_id={g}',text)
    def test_safety_env_and_budgets_preserved(self):
        text=transport.run_source()
        for token in ['holder_environment_matches(identity,pid)','drain_context(',"os.kill(pid,signal.SIGTERM)",'<shard_limits[shard]-120','<lane_limit-120','<4096']:
            self.assertIn(token,text)
        self.assertNotIn("'respawn-pane','-k'",text)
        self.assertIn("sleeper_identity=wait_stable_sleeper(identity)",text)
        self.assertNotIn("sleeper_identity=process_identity(sleeper)",text)
    def budget_fixture(self):
        return dict(cleanup_margin_seconds=120,shard_wall_seconds={'2':12600,'3':4800,'4':4200,'5':4800},
            lane_wall_seconds={'6':16920,'7':9720},max_wall_seconds_per_gpu_chain=16920)
    def test_authorized_budget_values_and_guards(self):
        shards,lanes=transport.budget_from_authorization(self.budget_fixture(),{6:(2,4),7:(3,5)})
        self.assertEqual(shards['2']-120,12480);self.assertEqual(lanes['6']-120,16800)
        self.assertEqual(transport.run_source().count('<lane_limit-120'),3)
        self.assertEqual(transport.run_source().count('<shard_limits[shard]-120'),2)
    def test_budget_rejects_short_lane_and_wrong_coverage(self):
        b=self.budget_fixture();b['lane_wall_seconds']['6']=16000
        with self.assertRaises(AssertionError):transport.budget_from_authorization(b,{6:(2,4),7:(3,5)})
        b=self.budget_fixture();b['shard_wall_seconds']['6']=1000
        with self.assertRaises(AssertionError):transport.budget_from_authorization(b,{6:(2,4),7:(3,5)})
    def test_budget_sleep_and_cleanup_limits(self):
        b=self.budget_fixture();b['lane_wall_seconds']['6']=24000
        with self.assertRaises(AssertionError):transport.budget_from_authorization(b,{6:(2,4),7:(3,5)})
        b=self.budget_fixture();b['cleanup_margin_seconds']=0
        with self.assertRaises(AssertionError):transport.budget_from_authorization(b,{6:(2,4),7:(3,5)})


if __name__=='__main__':unittest.main()
