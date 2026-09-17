import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import shlex
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport
import common as c
import run
import safe_size
import merge
import prepare


class TransportTests(unittest.TestCase):
    def test_parent_roots(self):
        self.assertEqual(c.ROOT,transport.ROOT)
        self.assertEqual(c.PARALLEL,HERE.parent)
        self.assertEqual(safe_size.BASE,c.BASE)
    def test_defaults_and_no_approval(self):
        self.assertEqual(c.LANES,{3:(0,),4:(1,)})
        with self.assertRaises(FileNotFoundError):c.approved(3)
    def test_all_gpu_mappings_and_collision(self):
        self.assertEqual(transport.selected_queues({'3':[0,1],'4':[2,3],'6':[4,5],'7':[6]})[7],(6,))
        for bad in [{'3':[0],'4':[0]},{'5':[0]},{'3':[0,1,2]},{'3':[7]}]:
            with self.assertRaises(AssertionError):transport.selected_queues(bad)
    def test_exact_sources_unchanged(self):
        for name in transport.EXPECTED:transport.original(name)
        for code in (transport.common_source(),transport.run_source(),transport.merge_source()):compile(code,'CPU_transport','exec')
    def test_exact_count_fail(self):
        for text in ['x','aa']:
            with self.assertRaises(AssertionError):transport.exact(text,'a','b')
    def test_worker_cpu_source(self):
        for shard,gpu in [(0,3),(1,4)]:
            code=c.worker_source(shard);compile(code,'worker_transport','exec')
            self.assertIn(f'cfg.gpu_device_id={gpu};',code)
            self.assertIn(str(c.shard_root(shard)),code)
            self.assertIn('ASSET_HASH_CHANGED',code)
        with self.assertRaises(AssertionError):c.lane_for(2)
    def test_selected_merge(self):
        code=transport.merge_source()
        self.assertIn('for shard in c.SELECTED_SHARDS:',code)
        self.assertIn('SELECTED_WAVE_NOT_FULL_POOL',code)
        self.assertIn("out=HERE/'merge'",code)
        self.assertIn("strict.audit_route(root,jobmap[ident])",code)
    def test_unchanged_limits_and_safety(self):
        code=transport.run_source()
        for token in ['<4096','<3480','<7080','<12*1024**2','<48*1024**3','finally:','restore_holder(', 'drain_context(']:
            self.assertIn(token,code)
        self.assertNotIn("'respawn-pane','-k'",code)
        self.assertNotIn('pidfd_open',code)
    def test_identity_fields(self):
        identity=dict(gpu_device=3,gpu_uuid='GPU-x',pid=1,starttime_ticks=1,proc_uid=0,cwd=str(c.ROOT),
            cmdline=['python','-u','scripts/occupy_idle_gpu.py','--gpu','3','--tag','x'],pane_id='%x',pane_target='x:0.0')
        prepare.validate_identity(identity,3,c.ROOT)
        with self.assertRaises(AssertionError):prepare.validate_identity(identity,4,c.ROOT)
    def test_global_merge_all_shards(self):
        code=(HERE/'merge_all.py').read_text()
        self.assertIn("sorted(assigned)==list(range(7))",code)
        self.assertIn('strict.audit_route(root,jobs[ident])',code)
        self.assertIn('HOLDER_NOT_RESTORED',code)
    def test_restore_preserves_exact_env(self):
        identity=dict(cmdline=['python','a script.py'],project_cache_environment={
            'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(c.ROOT/'test cache')})
        self.assertEqual(shlex.split(run.holder_command(identity)),['env','PYTHONDONTWRITEBYTECODE=1',
            'TMPDIR='+str(c.ROOT/'test cache'),'python','a script.py'])
        raw=('PYTHONDONTWRITEBYTECODE=1\0TMPDIR='+str(c.ROOT/'test cache')+'\0OTHER=x\0').encode()
        with mock.patch.object(Path,'read_bytes',return_value=raw):self.assertTrue(run.holder_environment_matches(identity,123))
        with mock.patch.object(Path,'read_bytes',return_value=b'PYTHONDONTWRITEBYTECODE=1\0'):
            self.assertFalse(run.holder_environment_matches(identity,123))
        with mock.patch.object(Path,'read_bytes',side_effect=FileNotFoundError):
            self.assertFalse(run.holder_environment_matches(identity,123))
    def test_environment_scope_fails_closed(self):
        for value in [{'PATH':'/usr/bin'},{'TMPDIR':'/tmp'}, {'PYTHONDONTWRITEBYTECODE':'0'}]:
            with self.assertRaises(AssertionError):run.checked_holder_environment({'project_cache_environment':value})


if __name__=='__main__':unittest.main()
