"""CPU-only inherited lease fault injections and exact chain boundary tests."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
t=load('holder_test_transport',HERE/'transport.py');c=t.c
p=load('holder_test_prepare',HERE/'prepare.py');a=load('holder_test_audit',HERE/'audit.py')
COLLECTION=c.LINE/'authorizations/DATA_SCALEOUT_HOLDER_IDENTITIES_V1.json'
DOC=next(x for x in c.read(COLLECTION)['holders'] if x['gpu_device']==3)

def cfg():
    result=c.read(c.BE/'batch_220/run_v1/EXECUTION_CONFIG.json')
    result.update(gpu_device=3,gpu_uuid=t.k.GPUS[3],lease_mode='holder_chain',holder_transport_version='special_scale_holder_v1')
    return result

# Run the original lease's fake-only safety cases through the privately rebound
# GPU3 identity. Its fake Ops never uses real signal, GPU, tmux or subprocesses.
OLD_TEST=c.BE/'gpu5_transport_v2/test_transport.py'
assert hashlib.sha256(OLD_TEST.read_bytes()).hexdigest()=='02aed16278a645584db28a87201eaa62f21a1a449c910c6cb7fbdda8d8b285ab'
inherited=load('holder_inherited_fake_tests',OLD_TEST)
inherited.HERE=HERE
inherited.t,inherited.IDENTITY=t.lease_module(DOC)
inherited.module=lambda:t.build_supervisor(c.BE/'batch_999997',cfg())
OMIT={'test_supervisor_exact_substitution_and_unmodified_guard','test_worker_adapter_unmodified_and_no_new_method',
      'test_actual_identity_schema_and_source_hashes','test_actual_authorization_scope_and_no_widening'}

class Tests(unittest.TestCase):
    def test_import_and_build_never_gpu(self):
        with patch('subprocess.check_output',side_effect=AssertionError('GPU_FORBIDDEN')),patch('subprocess.Popen',side_effect=AssertionError('CHILD_FORBIDDEN')):
            for name in ('core','transport','prepare','audit'):load('no_ops_'+name,HERE/(name+'.py'))
            t.build_supervisor(c.BE/'batch_999997',cfg())
    def test_original_lease_functions_unmodified(self):
        lease,_=t.lease_module(DOC)
        for name in ('run_lease','wait_gpu_drain','wait_terminal','restore_dead_holder','wait_dead_pane','identity_equal'):
            self.assertEqual(getattr(lease,name).__code__.co_filename,str(t.LEASE_SOURCE))
        raw=(c.RUNTIME/'compact_loop_v2/run.py').read_text()
        modified=t.core.supervisor_source(raw,3)
        for name in ('gpu','check_gpu','parse_gpu'):
            before=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name)
            after=next(n for n in ast.parse(modified).body if isinstance(n,ast.FunctionDef) and n.name==name)
            if name!='gpu':self.assertEqual(ast.dump(before),ast.dump(after))
        self.assertIn("'nvidia-smi','-i','3'",modified)
    def test_wrong_holder_fields_rejected(self):
        for key,value in [('gpu_device',6),('gpu_uuid',t.k.GPUS[4]),('pid',False),('pane_pid',1),('starttime_ticks',0),('proc_uid',1),('cwd','/'),('pane_dead',True),('unknown_or_nonholder_process_signal_allowed',True)]:
            bad=copy.deepcopy(DOC);bad[key]=value
            with self.assertRaises(ValueError):t.check_identity(bad,3)
    def test_holder_environment_and_command_allowlist(self):
        for env in ({'LD_PRELOAD':'bad'},{'CUDA_CACHE_PATH':'/tmp'},{'PYTHONDONTWRITEBYTECODE':'0'}):
            bad=dict(DOC,project_cache_environment=env)
            with self.assertRaises(ValueError):t.check_identity(bad,3)
        bad=copy.deepcopy(DOC);bad['cmdline'][2]='scripts/train.py'
        with self.assertRaises(ValueError):t.check_identity(bad,3)
    def test_restore_command_restores_exact_cache_subset(self):
        doc=dict(DOC,project_cache_environment={'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(HERE)})
        lease,_=t.lease_module(doc);ops=t.make_ops(doc,lease);calls=[]
        ops.call=lambda *args:calls.append(args)
        ops.restore_command();command=shlex.split(calls[0][-1])
        self.assertEqual(calls[0][:5],('tmux','respawn-pane','-t',doc['pane_target'],'-c'))
        self.assertEqual(command[-11:],doc['cmdline'])
        self.assertIn('TMPDIR='+str(HERE),command)
        self.assertIn('PYTHONDONTWRITEBYTECODE=1',command)
        self.assertIn('-u',command);self.assertIn('CUDA_CACHE_PATH',command)
        self.assertNotIn('-k',calls[0])
    def test_remain_changed_fails_before_borrow(self):
        lease,_=t.lease_module(DOC);ops=t.make_ops(DOC,lease)
        ops.call=lambda *args:'on'
        with self.assertRaisesRegex(ValueError,'FROZEN_REMAIN'):ops.remain()
    def test_single_owned_popen_only(self):
        module=t.build_supervisor(c.BE/'batch_999997',cfg())
        with self.assertRaisesRegex(ValueError,'ONE_OWN'):module.subprocess.Popen(['fake'],start_new_session=False)
    def test_gpu6_and_unregistered_uuid_refused(self):
        for gpu in (0,1,2,6):
            bad=dict(cfg(),gpu_device=gpu)
            with self.assertRaises((KeyError,ValueError)):t.build_supervisor(c.BE/'batch_999997',bad)
    def test_prepare_audit_compile_interfaces(self):
        compile(p.prepare_source((c.BE/'prepare.py').read_text()),'PREPARE_CPU_ONLY','exec')
        a.base.stack()
        self.assertTrue(a.AUDIT_ROOT.is_relative_to(c.WF/'quality_cpu/batch_acceptance_v1'))
    def test_each_gpu_full_raw_decision_resource_audit_interface(self):
        for gpu,uuid in t.k.GPUS.items():
            configuration=dict(cfg(),gpu_device=gpu,gpu_uuid=uuid)
            module=t.build_supervisor(c.BE/'batch_999997',configuration)
            raw='<nvidia_smi_log><gpu><uuid>'+uuid+'</uuid><fb_memory_usage><used>733 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes>'+''.join('<process_info><pid>'+str(pid)+'</pid><used_memory>'+str(mib)+' MiB</used_memory><type>G</type></process_info>' for pid,mib in ((10,246),(2,246),(3,245)))+'</processes></gpu></nvidia_smi_log>'
            events=[]
            with patch.object(module.subprocess,'check_output',return_value=raw),patch.object(module.time,'monotonic',return_value=100.25),patch.object(t.c,'append',side_effect=lambda path,value:events.append(json.loads(json.dumps(value)))):
                snapshot,upper=module._active_sample(3,100)
            context=a.base.verify_events(events,module,3)
            resource=json.loads(json.dumps(dict(snapshot,own_memory_upper_mib=upper,elapsed=.3,active_accounting_sample_index=1)))
            a.base.verify_resource(context,resource)
            self.assertEqual(context['amended'],1);self.assertEqual(context['seen'],{1});self.assertEqual(upper,245)
            self.assertFalse(events[-1]['payload']['original_guard_passed'])
            bad=copy.deepcopy(events);bad[-1]['payload']['original_guard_passed']=True
            with self.assertRaises(ValueError):a.base.verify_events(bad,module,3)

class ChainTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_CHAIN_');self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name);self.prior=self.base/'prior';root=self.prior/'run_v1';root.mkdir(parents=True)
        self.initial=self.base/'initial.json';c.save(self.initial,DOC)
        self.auth=self.base/'AUTH.json';c.save(self.auth,dict(initial_holder_identity_path=str(self.initial),initial_holder_identity_sha256=c.sha(self.initial)))
        previous=dict(gpu_device=3,scale_authorization_path=str(self.auth));c.save(root/'EXECUTION_CONFIG.json',previous)
        c.save(root/'INPUT_LOCK.json',{str(root/'EXECUTION_CONFIG.json'):c.sha(root/'EXECUTION_CONFIG.json')})
        c.save(self.prior/'MAIN_AGENT_SCALE_APPROVAL.json',{'fixture_approved':True})
        c.save(root/'SUPERVISOR_RESULT.json',dict(returncode=0,error=None,cleanup_complete=True))
        c.save(root/'LEASE_RESULT.json',dict(execute_returned=True,error=None,holder_restored=True))
        identity=dict(pid=99999991,starttime_ticks=900001,command=' '.join(DOC['cmdline']),cwd=DOC['cwd'],proc_uid=0)
        restoration=dict(restored=True,remain_on_exit_restored=True,pid=identity['pid'],process_identity=identity,
            gpu=dict(uuid=DOC['gpu_uuid'],processes={str(identity['pid']):{'mib':22001,'type':'C'}}))
        c.save(root/'RESTORATION.json',restoration);c.save(root/'LAUNCH_RESULT.json',dict(status='ORIGINAL_SUPERVISOR_RETURNED'))
        self.cfg=dict(gpu_device=3,gpu_uuid=DOC['gpu_uuid'],scale_authorization_path=str(self.auth),
            previous_holder_batch=str(self.prior),previous_holder_input_lock_sha256=c.sha(root/'INPUT_LOCK.json'))
        self.patch=patch.object(t,'approval_value',return_value={'fixture_approved':True});self.patch.start();self.addCleanup(self.patch.stop)
    def test_first_batch_uses_exact_main_identity(self):
        document,proof=t.resolve_identity(self.base,dict(self.cfg,previous_holder_batch=None))
        self.assertEqual(document,DOC);self.assertFalse(proof['source_files'])
    def test_immediate_restoration_pid_bound(self):
        document,proof=t.resolve_identity(self.base,self.cfg)
        self.assertEqual(document['pid'],99999991);self.assertEqual(document['pane_pid'],99999991)
        self.assertEqual(document['starttime_ticks'],900001);self.assertEqual(document['cmdline'],DOC['cmdline'])
        self.assertEqual(len(proof['source_files']),7)
    def test_predecessor_error_missing_or_changed_refused(self):
        original=c.read
        def failed(path):
            value=original(path)
            if Path(path).name=='LEASE_RESULT.json':value=dict(value,error='failed')
            return value
        with patch.object(t.c,'read',side_effect=failed):
            with self.assertRaisesRegex(ValueError,'LEASE_FAILED'):t.resolve_identity(self.base,self.cfg)
        with self.assertRaisesRegex(ValueError,'INPUT_CHANGED'):
            t.resolve_identity(self.base,dict(self.cfg,previous_holder_input_lock_sha256='0'*64))
    def test_restored_other_process_refused(self):
        original=c.read
        def wrong(path):
            value=original(path)
            if Path(path).name=='RESTORATION.json':value=dict(value,process_identity=dict(value['process_identity'],command='training'))
            return value
        with patch.object(t.c,'read',side_effect=wrong):
            with self.assertRaisesRegex(ValueError,'OTHER_COMMAND'):t.resolve_identity(self.base,self.cfg)
    def test_low_restored_occupancy_refused(self):
        original=c.read
        def wrong(path):
            value=original(path)
            if Path(path).name=='RESTORATION.json':value=dict(value,gpu=dict(value['gpu'],processes={'99999991':{'mib':100,'type':'C'}}))
            return value
        with patch.object(t.c,'read',side_effect=wrong):
            with self.assertRaisesRegex(ValueError,'NOT_PROVEN'):t.resolve_identity(self.base,self.cfg)
    def test_holder_terminal_complete_chain_audit_interface(self):
        root=self.base/'current/run_v1';root.mkdir(parents=True)
        document,proof=t.resolve_identity(root.parent,self.cfg)
        c.save(root/'CHAIN_IDENTITY_BINDING.json',proof)
        c.save(root/'LEASE_RESULT.json',dict(execute_returned=True,error=None,holder_restored=True,external_processes_stopped=0))
        identity=dict(pid=99999992,starttime_ticks=900002,command=' '.join(DOC['cmdline']),cwd=DOC['cwd'],proc_uid=0)
        restoration=dict(restored=True,remain_on_exit_restored=True,pid=identity['pid'],process_identity=identity,
            gpu=dict(uuid=DOC['gpu_uuid'],processes={str(identity['pid']):{'mib':22001,'type':'C'}}))
        c.save(root/'RESTORATION.json',restoration)
        with patch.object(a.t,'resolve_identity',side_effect=t.resolve_identity):
            result=a.verify_holder_terminal(root,self.cfg)
            self.assertTrue(result['holder_chain_verified']);self.assertTrue(result['restore_verified'])
            self.assertEqual(result['resolved_holder_pid'],99999991)
            self.assertEqual(result['restored_holder_pid'],99999992)
            original=a.c.read
            def mismatched(path):
                value=original(path)
                if Path(path)==root/'CHAIN_IDENTITY_BINDING.json':value=dict(value,name_based_discovery=True)
                return value
            with patch.object(a.c,'read',side_effect=mismatched):
                with self.assertRaisesRegex(ValueError,'CHAIN_PROOF'):a.verify_holder_terminal(root,self.cfg)

def load_tests(loader,tests,pattern):
    suite=unittest.TestSuite([loader.loadTestsFromTestCase(Tests),loader.loadTestsFromTestCase(ChainTests)])
    suite.addTests(inherited.Tests(name) for name in loader.getTestCaseNames(inherited.Tests) if name not in OMIT)
    return suite
if __name__=='__main__':unittest.main(verbosity=2)
