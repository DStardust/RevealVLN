import importlib.util
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
t=load('tested_v5_transport',HERE/'transport.py')
p=load('tested_v5_prepare',HERE/'prepare.py')
def cfg():
    return dict(sampling_amendment=t.AMENDMENT.copy(),thresholds_unchanged=True,
        supervision_sampling_policy_changed=True,runtime_transport_version='gpu5_transport_v5',
        cohort_member=False,cohort_replacement=False,engineering_retry_count=1,
        engineering_retry_of=str(t.base.BE/'batch_06r1/run_v1'),source_selection_indices=[9,10,11],
        gpu_device=5,gpu_uuid=t.base.UUID,factory_variant='winding_v1',supervision_wall_seconds=3900,
        budget={'total_seconds':3600})
def xml(total=560,own=300,external=246,uuid=None):
    return '<nvidia_smi_log><gpu><uuid>'+str(uuid or t.base.UUID)+'</uuid><fb_memory_usage><used>'+str(total)+' MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes>'+''.join(
        '<process_info><pid>'+str(pid)+'</pid><type>'+kind+'</type><used_memory>'+str(mib)+' MiB</used_memory></process_info>'
        for pid,kind,mib in [(101,'C',external),(202,'G',own)])+'</processes></gpu></nvidia_smi_log>'
class Tests(unittest.TestCase):
    def module(self,folder):
        (folder/'run_v1').mkdir()
        module=t.build_supervisor(folder,cfg());module.time=types.SimpleNamespace(monotonic=lambda:0.)
        return module
    @contextmanager
    def scoped_module(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            module=self.module(Path(d))
            try:yield module
            finally:module._telemetry_close()
    def test_exact_single_loop_change(self):
        source=(t.base.RUNTIME/'compact_loop_v2/run.py').read_text()
        old=t.original_supervisor_source(source);new=t.supervisor_source(source)
        self.assertEqual(new.replace('snapshot,upper=_telemetry_sample(proc.pid,started+3900)',
                                    'snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)'),old)
        self.assertIn('sample[\'elapsed\'] < 3900',new)
    def test_no_source_match_fail_closed(self):
        with self.assertRaises((AssertionError,ValueError)):t.supervisor_source('')
    def test_sealed_lease_functions_unchanged(self):
        for name in ('run_lease','wait_gpu_drain','restore_dead_holder','wait_terminal','normalize_identity','check_gpu'):
            if name=='check_gpu':continue
            self.assertEqual(getattr(t.base,name).__code__.co_filename,str(t.base.HERE/'transport.py'))
        self.assertEqual(t.base.RealOps.gpu.__code__.co_filename,str(t.base.HERE/'transport.py'))
    def test_private_import_does_not_change_fresh_v4(self):
        fresh=load('another_independent_v4',t.checked_v4('transport.py'))
        self.assertEqual(fresh.v3.base.build_supervisor.__code__.co_filename,str(fresh.v3.base.HERE/'transport.py'))
        self.assertIsNot(fresh.v3.base,t.base)
    def test_cpu_build_no_query_or_receipt(self):
        with self.scoped_module() as module:
            self.assertFalse((module.OUT/'GPU_RAW_SAMPLES.jsonl').exists())
            self.assertIs(module._gpu5_raw_gpu,module.gpu)
    def test_real_parser_and_durable_raw_before_pass(self):
        with self.scoped_module() as module:
            raw=xml()
            with patch.object(module.subprocess,'check_output',return_value=raw) as query:
                snapshot,upper=module._telemetry_sample(202,3900.)
            self.assertEqual(snapshot['memory_mib'],560);self.assertEqual(upper,314)
            self.assertEqual(query.call_count,1);self.assertEqual(query.call_args.kwargs['timeout'],15.)
            rows=[json.loads(s) for s in (module.OUT/'GPU_RAW_SAMPLES.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['raw_xml'],raw);self.assertEqual(rows[-1]['event'],'guard_pass')
    def test_one_bounded_retry_original_guard(self):
        with self.scoped_module() as module:
            with patch.object(module.subprocess,'check_output',side_effect=[xml(total=532),xml()]) as query:
                snapshot,upper=module._telemetry_sample(202,3900.)
            self.assertEqual(query.call_count,2);self.assertLessEqual(query.call_args.kwargs['timeout'],1.)
            rows=[json.loads(s) for s in (module.OUT/'GPU_RAW_SAMPLES.jsonl').read_text().splitlines()]
            self.assertEqual([r['raw_xml'] for r in rows if r['event']=='raw_xml'],[xml(total=532),xml()])
            self.assertEqual(rows[-1]['retries'],1);self.assertEqual(snapshot['memory_mib'],560)
    def test_never_retry_actual_external_limit(self):
        with self.scoped_module() as module:
            with patch.object(module.subprocess,'check_output',return_value=xml(total=1200,external=769)) as query:
                with self.assertRaisesRegex(AssertionError,'EXTERNAL_RESOURCE_LOAD'):module._telemetry_sample(202,3900.)
            self.assertEqual(query.call_count,1)
    def test_never_retry_wrong_uuid(self):
        with self.scoped_module() as module:
            with patch.object(module.subprocess,'check_output',return_value=xml(uuid='bad')) as query:
                with self.assertRaisesRegex(AssertionError,'GPU_IDENTITY'):module._telemetry_sample(202,3900.)
            self.assertEqual(query.call_count,1)
    def test_three_inconsistencies_stop_no_fourth_query(self):
        with self.scoped_module() as module:
            with patch.object(module.subprocess,'check_output',return_value=xml(total=532)) as query:
                with self.assertRaisesRegex(AssertionError,'MEMORY_ACCOUNTING'):module._telemetry_sample(202,3900.)
            self.assertEqual(query.call_count,3)
    def test_original_deadline_no_query(self):
        with self.scoped_module() as module:
            with patch.object(module.subprocess,'check_output') as query:
                with self.assertRaises(TimeoutError):module._telemetry_sample(202,0.)
            self.assertEqual(query.call_count,0)
    def test_lease_raw_query_not_logged_or_retried(self):
        with self.scoped_module() as module:
            with patch.object(module.subprocess,'check_output',return_value=xml(total=532)) as query:
                snapshot=module._gpu5_raw_gpu()
            self.assertEqual(snapshot['memory_mib'],532);self.assertEqual(query.call_count,1)
            self.assertFalse((module.OUT/'GPU_RAW_SAMPLES.jsonl').exists())
    def test_log_failure_no_retry(self):
        with self.scoped_module() as module:
            with patch.object(module.subprocess,'check_output',return_value=xml(total=532)) as query:
                with patch.object(t.telemetry.DurableLog,'__call__',side_effect=OSError('disk')):
                    with self.assertRaises(OSError):module._telemetry_sample(202,3900.)
            self.assertEqual(query.call_count,1)
    def test_cfg_requires_sampling_change_not_threshold_relaxation(self):
        for key,value in [('cohort_replacement',True),('engineering_retry_count',2),('thresholds_unchanged',False),('supervision_sampling_policy_changed',False)]:
            c=cfg();c[key]=value
            with self.assertRaises(AssertionError):t.check_amendment(c)
    def test_prepare_source_only_explicit_new_metadata(self):
        source=p.adapted_prepare_source((t.base.BE/'prepare.py').read_text());compile(source,'v5','exec')
        self.assertIn('supervision_sampling_policy_changed=True',source)
        self.assertIn('cohort_replacement=False',source)
        self.assertIn('gpu5_transport_v5/transport.py',p.p.LAUNCHER)
        self.assertEqual(p.p.WORKER,t.load('original_v2_prepare_for_test',t.v4.v3.checked('prepare.py')).WORKER)
    def test_authorization_matches_new_main_record(self):
        auth=t.base.LINE/'authorizations/WITNESS_MULTI_PROGRAM_BATCH06R2_GPU5_V1.json'
        identity=t.base.LINE/'authorizations/WITNESS_BATCH06R2_GPU5_HOLDER_IDENTITY_V1.json'
        doc=t.v4.read(auth)
        t.validate_authorization(doc,t.base.BE/'batch_06r2',t.base.WF/'multi_program_bank_v1/language_ready_v2',[9,10,11],identity)
        doc['old_cohort_replacement_allowed']=True
        with self.assertRaises(AssertionError):t.validate_authorization(doc,t.base.BE/'batch_06r2',t.base.WF/'multi_program_bank_v1/language_ready_v2',[9,10,11],identity)
    def test_dependency_chain_fixed(self):
        t.verify_sources();self.assertEqual(len(t.dependency_paths()),24)
    def test_other_batch_not_prepared(self):
        with self.assertRaises(AssertionError):p.prepare('notused','batch_06r3',[9,10,11],'notused','notused')
    def test_main_baseexception_closes_durable_receipt(self):
        original=t.original_build_supervisor;logs=[];real_log=t.telemetry.DurableLog
        class Capture(real_log):
            def __init__(self,*args):super().__init__(*args);logs.append(self)
        def fake_main_builder(batch,configuration):
            module=original(batch,configuration)
            module.time=types.SimpleNamespace(monotonic=lambda:0.)
            module.subprocess.check_output=lambda *args,**kwargs:xml()
            def failing():
                module._telemetry_sample(202,3900.)
                raise KeyboardInterrupt('cpu-only simulated main interruption')
            module.main=failing
            return module
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            folder=Path(d);(folder/'run_v1').mkdir()
            with patch.object(t,'original_build_supervisor',side_effect=fake_main_builder),patch.object(t.telemetry,'DurableLog',Capture):
                module=t.build_supervisor(folder,cfg())
                with self.assertRaises(KeyboardInterrupt):module.main()
            self.assertEqual(len(logs),1);self.assertTrue(logs[0].stream.closed)
    def test_wrong_prior_wall_rejected(self):
        auth=t.v4.read(t.base.LINE/'authorizations/WITNESS_MULTI_PROGRAM_BATCH06R2_GPU5_V1.json')
        auth['previous_supervisor_wall_seconds']+=1
        with self.assertRaises(AssertionError):t.validate_prior_attempt(auth,t.base.WF/'multi_program_bank_v1/language_ready_v2',[9,10,11],
            t.base.LINE/'authorizations/WITNESS_BATCH06R2_GPU5_HOLDER_IDENTITY_V1.json')

if __name__=='__main__':unittest.main()
