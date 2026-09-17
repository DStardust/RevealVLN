"""CPU fault-injection only. Never invoke GPU, tmux, or real signals."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
t=load('gpu5_test_transport',HERE/'transport.py')
p=load('gpu5_test_prepare',HERE/'prepare.py')
IDENTITY={'pid':t.HOLDER_PID,'starttime_ticks':131277976,'command':t.COMMAND,'cwd':str(t.ROOT),
          'pane':t.PANE,'pane_id':t.PANE_ID,'proc_uid':0}

class Signals:
    SIG_IGN='ignore'
    def __init__(self):self.values={2:'old_int',15:'old_term'}
    def getsignal(self,s):return self.values[s]
    def signal(self,s,handler):self.values[s]=handler

class Ops:
    def __init__(self):
        self.state='held';self.events=[];self.bad_identity=False;self.fail_kill=False
        self.stuck=False;self.unknown_pane=False;self.external=0;self.restored_memory=22000
        self.worker_alive=False;self.clean_fail=False;self.start=0
    def identity(self):
        x=dict(IDENTITY)
        if self.bad_identity:x['starttime_ticks']+=1
        return x
    def gpu(self,module):
        procs={}
        if self.state=='held':procs[t.HOLDER_PID]={'mib':22000,'type':'C'}
        if self.state=='restored':procs[999001]={'mib':self.restored_memory,'type':'C'}
        if self.external:procs[777]={'mib':self.external,'type':'C'}
        return {'uuid':t.UUID,'memory_mib':sum(v['mib'] for v in procs.values())+100,'utilization':0,'processes':procs}
    def remain(self):return 'off'
    def set_remain(self,value):self.events.append(('remain',value))
    def terminate(self,pid):
        self.events.append(('signal',pid))
        if self.fail_kill:raise OSError('simulated_signal_failure')
        if not self.stuck:self.state='dead'
    def exists(self,pid):
        if pid==t.HOLDER_PID:return self.state=='held'
        if pid==999001:return self.state=='restored'
        if pid==123456:return self.worker_alive
        return False
    def proc(self,pid):return dict(IDENTITY,pid=pid)
    def pane(self,fmt):
        if fmt=='#{pane_id}':return t.PANE_ID
        if fmt=='#{pane_pid}':return str(999001 if self.state=='restored' else t.HOLDER_PID)
        if fmt=='#{pane_dead}':return '0' if self.unknown_pane or self.state!='dead' else '1'
        raise AssertionError(fmt)
    def restore_command(self):self.events.append(('restore',None));self.state='restored'
    def cleanup_own(self,module):
        self.events.append(('cleanup',None))
        if self.clean_fail:raise RuntimeError('simulated_own_cleanup_failure')
        self.worker_alive=False
    def sleep(self,seconds):self.start+=seconds
    def clock(self):return self.start

def module():
    cfg={'gpu_device':5,'gpu_uuid':t.UUID,'factory_variant':'winding_v1',
         'supervision_wall_seconds':3900,'budget':{'total_seconds':3600}}
    return t.build_supervisor(t.BE/'batch_02',cfg)

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_TEST_');self.out=Path(self.temp.name)
        self.ops=Ops();self.signals=Signals();self.mod=module()
    def tearDown(self):self.temp.cleanup()
    def runlease(self,execute=lambda:None):return t.run_lease(self.out,IDENTITY,self.mod,execute,ops=self.ops,signals=self.signals)
    def test_success_borrow_restore_exact_only(self):
        self.runlease();result=t.read(self.out/'LEASE_RESULT.json')
        self.assertTrue(result['holder_restored']);self.assertEqual([x for x in self.ops.events if x[0]=='signal'],[('signal',t.HOLDER_PID)])
        self.assertEqual(self.signals.values,{2:'old_int',15:'old_term'})
        self.assertLess(self.ops.events.index(('cleanup',None)),self.ops.events.index(('restore',None)))
    def test_failure_still_restores(self):
        def fail():raise ValueError('worker_failed')
        with self.assertRaises(RuntimeError):self.runlease(fail)
        self.assertTrue(t.read(self.out/'RESTORATION.json')['restored'])
    def test_sigterm_during_run_still_restores(self):
        def fail():self.signals.values[15](15,None)
        with self.assertRaises(RuntimeError):self.runlease(fail)
        self.assertIn('GPU5_LEASE_SIGNAL',t.read(self.out/'LEASE_RESULT.json')['error']['message'])
        self.assertTrue(t.read(self.out/'RESTORATION.json')['restored'])
    def test_preworker_failure_still_restores(self):
        def fail():raise TimeoutError('no_real_idle')
        with self.assertRaises(RuntimeError):self.runlease(fail)
        self.assertEqual(self.ops.state,'restored')
    def test_wrong_identity_never_signalled(self):
        self.ops.bad_identity=True
        with self.assertRaises(RuntimeError):self.runlease()
        self.assertFalse(any(x[0]=='signal' for x in self.ops.events))
        self.assertTrue(t.read(self.out/'RESTORATION.json')['not_borrowed'])
    def test_signal_failure_keeps_original(self):
        self.ops.fail_kill=True
        with self.assertRaises(RuntimeError):self.runlease()
        self.assertTrue(t.read(self.out/'RESTORATION.json')['original_holder_retained'])
        self.assertFalse(any(x[0]=='restore' for x in self.ops.events))
    def test_holder_timeout_no_kill_escalation(self):
        self.ops.stuck=True
        with self.assertRaises(RuntimeError):self.runlease()
        self.assertEqual(sum(x[0]=='signal' for x in self.ops.events),1)
        self.assertEqual(self.ops.start,20)
    def test_unknown_pane_not_respawned(self):
        def change():self.ops.unknown_pane=True
        with self.assertRaises(RuntimeError):self.runlease(change)
        self.assertFalse(any(x[0]=='restore' for x in self.ops.events))
        self.assertFalse(t.read(self.out/'RESTORATION.json')['restored'])
    def test_external_load_before_no_signal(self):
        self.ops.external=769
        with self.assertRaises(RuntimeError):self.runlease()
        self.assertFalse(any(x[0]=='signal' for x in self.ops.events))
    def test_restore_high_memory_no_respawn(self):
        def load():self.ops.external=1024
        with self.assertRaises(RuntimeError):self.runlease(load)
        self.assertFalse(any(x[0]=='restore' for x in self.ops.events))
    def test_own_child_cleanup_precedes_restore(self):
        def fail():
            t.save(self.out/'PROCESS.json',{'pid':123456});self.ops.worker_alive=True;raise ValueError('early_cleanup_abort')
        with self.assertRaises(RuntimeError):self.runlease(fail)
        self.assertTrue(t.read(self.out/'RESTORATION.json')['restored'])
        self.assertFalse(self.ops.worker_alive)
    def test_own_cleanup_failure_never_restores_on_live_worker(self):
        self.ops.clean_fail=True
        with self.assertRaises(RuntimeError):self.runlease()
        self.assertFalse(any(x[0]=='restore' for x in self.ops.events))
    def test_restore_reservation_must_really_be_observed(self):
        self.ops.restored_memory=100
        with self.assertRaises(RuntimeError):self.runlease()
        self.assertFalse(t.read(self.out/'RESTORATION.json')['restored'])
    def test_supervisor_exact_substitution_and_unmodified_guard(self):
        raw=(t.RUNTIME/'compact_loop_v2/run.py').read_text();derived=t.supervisor_source(raw)
        self.assertIn("'nvidia-smi','-i','5'",derived)
        self.assertIn("sample['elapsed'] < 3900",derived)
        guard=raw[raw.index('def check_gpu('):raw.index('def disk_size(')]
        self.assertIn(guard,derived)
        with self.assertRaises(AssertionError):t.supervisor_source(raw.replace("'nvidia-smi','-i','1'","'nvidia-smi','-i','2'"))
    def test_worker_adapter_unmodified_and_no_new_method(self):
        source=(t.BE/'prepare.py').read_text();derived=p.adapted_prepare_source(source)
        self.assertIn("assert gpu == 5 and variant == 'winding_v1'",derived)
        self.assertIn('total_actions=60000,total_seconds=3600',derived)
        with self.assertRaises(ValueError):p.adapted_prepare_source(source+'\n'+"for p in code:lock[str(p.resolve())]=sha(p)")
    def test_actual_identity_schema_and_source_hashes(self):
        doc=t.read(t.LINE/'authorizations/WITNESS_BATCH02_GPU5_HOLDER_IDENTITY_V1.json')
        self.assertEqual(t.normalize_identity(doc),IDENTITY);t.verify_sources()
        for field,value in [('pid',1),('proc_uid',42),('gpu_device',2),('starttime_ticks',0)]:
            bad=copy.deepcopy(doc);bad[field]=value
            with self.assertRaises(ValueError):t.normalize_identity(bad)
    def test_actual_authorization_scope_and_no_widening(self):
        auth=t.read(t.LINE/'authorizations/WITNESS_CROSS_HOUSE_BATCH02_GPU5_V1.json')
        paths=(t.BE/'batch_02',t.BE/'batch_02_preparation_v1/composite',[0,1,2],
               t.LINE/'authorizations/WITNESS_BATCH02_GPU5_HOLDER_IDENTITY_V1.json')
        t.validate_authorization(auth,*paths)
        for field,value in [('approved',False),('gpu_device',2),('max_actions',60001),('training_allowed',True)]:
            bad=copy.deepcopy(auth);bad[field]=value
            with self.assertRaises(ValueError):t.validate_authorization(bad,*paths)

if __name__=='__main__':unittest.main()
