"""CPU fake-query tests; no GPU, holder, simulation or external process calls."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location('scale_test_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
t=load('transport');a=load('audit');p=load('prepare');c=t.c
def config():
    cfg=c.read(c.BE/'batch_07/run_v1/EXECUTION_CONFIG.json')
    cfg.update(runtime_transport_version='special_scale_transport_v1',accounting_amendment=c.AMENDMENT,
        lease_mode='nonexclusive',cohort_membership=None,auto_retry=False)
    return cfg
def xml(memory=733,procs=((1,246),(2,246),(3,245))):
    return '<nvidia_smi_log><gpu><uuid>'+t.GPUS[2]+'</uuid><fb_memory_usage><used>'+str(memory)+' MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes>'+''.join('<process_info><pid>'+str(pid)+'</pid><used_memory>'+str(mib)+' MiB</used_memory><type>G</type></process_info>' for pid,mib in procs)+'</processes></gpu></nvidia_smi_log>'
def module():return t.build_supervisor(c.BE/'batch_999999',config())
def events_for(raw=None):
    m=module();records=[];raw=raw or xml()
    with patch.object(m.subprocess,'check_output',return_value=raw),patch.object(m.time,'monotonic',return_value=100.25),patch.object(t.c,'append',side_effect=lambda path,value:records.append(json.loads(json.dumps(value)))):
        snapshot,upper=m._active_sample(3,100)
    return m,records,snapshot,upper

class Tests(unittest.TestCase):
    def test_import_no_gpu(self):
        with patch('subprocess.check_output',side_effect=AssertionError('NO_GPU_QUERY')),patch('subprocess.Popen',side_effect=AssertionError('NO_CHILD')):
            for name in ('common','transport','telemetry','prepare','audit'):load(name)
    def test_original_context_functions_unchanged(self):
        src=(c.RUNTIME/'compact_loop_v2/run.py').read_text()
        base=c.load('baseline_shared',c.BE/'shared.py').supervisor_source(src,2)
        new=t.supervisor_source(src,2)
        def funcs(code):return {n.name:n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
        old=funcs(base);fresh=funcs(new)
        for name in ('parse_gpu','gpu','check_gpu','disk_size','save'):
            self.assertEqual(ast.dump(old[name]),ast.dump(fresh[name]))
        old_try=next(x for x in old['main'].body if isinstance(x,ast.Try))
        new_try=next(x for x in fresh['main'].body if isinstance(x,ast.Try))
        self.assertEqual([ast.dump(x) for x in old_try.finalbody],[ast.dump(x) for x in new_try.finalbody])
    def test_actual_guard_difference_explicit(self):
        m,ev,snap,upper=events_for();self.assertEqual(upper,245)
        with self.assertRaisesRegex(AssertionError,'MEMORY_ACCOUNTING'):m.check_gpu(snap,3)
        self.assertFalse(ev[-1]['payload']['original_guard_passed']);self.assertTrue(ev[-1]['payload']['amended_acceptance'])
        context=a.verify_events(ev,m,3);self.assertEqual(context['amended'],1)
    def test_consistent_original_guard(self):
        m,ev,snap,upper=events_for(xml(900));self.assertEqual(upper,408)
        self.assertTrue(ev[-1]['payload']['original_guard_passed']);self.assertEqual(a.verify_events(ev,m,3)['amended'],0)
    def test_original_idle_accounting_not_amended(self):
        m=module()
        with self.assertRaisesRegex(AssertionError,'OWN_GPU_MEMORY_UPPER_BOUND'):m.check_gpu(m.parse_gpu(xml()))
    def test_real_overage_and_external_reject(self):
        for total,procs in ((4095,((1,700),(3,3500))),(700,((1,769),(3,1))),(6000,((1,246),(3,5000)))):
            with self.assertRaises(AssertionError):events_for(xml(total,procs))
    def test_absent_worker_no_amendment(self):
        with self.assertRaisesRegex(AssertionError,'OWN_GPU_MEMORY_UPPER_BOUND'):events_for(xml(733,((1,246),(2,246),(4,245))))
    def test_invalid_values_fail(self):
        m=module()
        for bad in (-1,float('nan'),float('inf'),True):
            with self.assertRaises(AssertionError):t.telemetry.assess(dict(memory_mib=bad,processes={3:{'mib':1,'type':'G'}}),3,m.check_gpu,lambda x:None)
    def test_raw_hash_parse_and_order_counterexamples(self):
        m,ev,_,_=events_for()
        variations=[]
        wrong=copy.deepcopy(ev);wrong[0]['payload']['sha256']='0'*64;variations.append(wrong)
        wrong=copy.deepcopy(ev);wrong[1]['payload']['memory_mib']=900;variations.append(wrong)
        wrong=copy.deepcopy(ev);wrong[0],wrong[1]=wrong[1],wrong[0];variations.append(wrong)
        wrong=copy.deepcopy(ev);wrong[-1]['payload']['original_guard_passed']=True;variations.append(wrong)
        variations.append(ev[:-1])
        for wrong in variations:
            with self.assertRaises(ValueError):a.verify_events(wrong,m,3)
    def test_resource_values_and_index(self):
        m,ev,snapshot,upper=events_for();ctx=a.verify_events(ev,m,3)
        sample=json.loads(json.dumps(dict(snapshot,elapsed=.3,own_memory_upper_mib=upper,active_accounting_sample_index=1)))
        a.verify_resource(ctx,sample);self.assertEqual(ctx['seen'],{1})
        with self.assertRaises(ValueError):a.verify_resource(ctx,sample)
        for key,value in [('memory_mib',737),('own_memory_upper_mib',241),('elapsed',.1),('active_accounting_sample_index',2)]:
            altered=dict(sample,**{key:value})
            with self.assertRaises(ValueError):a.verify_resource(a.verify_events(ev,m,3),altered)
    def test_clock_window_and_reversal(self):
        m,ev,_,_=events_for()
        for key,value in [('monotonic',4001),('supervisor_started_monotonic',99),('monotonic',99)]:
            wrong=copy.deepcopy(ev);wrong[-1][key]=value
            with self.assertRaises(ValueError):a.verify_events(wrong,m,3)
    def test_durable_error_stops_before_return(self):
        m=module()
        with patch.object(m.subprocess,'check_output',return_value=xml()),patch.object(t.c,'append',side_effect=OSError('io')):
            with self.assertRaises(OSError):m._active_sample(3,100)
    def test_unknown_exception_not_amended(self):
        m=module();snap=m.parse_gpu(xml())
        for exc in (AssertionError('DIFFERENT'),OSError('io'),KeyboardInterrupt()):
            with self.assertRaises(type(exc)):
                t.telemetry.assess(snap,3,lambda *args:(_ for _ in ()).throw(exc),lambda x:None)
    def test_private_auditor_compiles_and_grade_explicit(self):
        gate=a.stack()
        self.assertIn('QUALITY_VERIFIED_FIT_AMENDED_ACTIVE_ACCOUNTING_NOT_MODEL_GAIN',a.adapted_acceptance(c.load('gate_test',a.GATE).ADAPTED_SOURCE))
        self.assertIs(gate.old.verify_resource,a.verify_resource)
    def test_prepare_source_cpu_compile(self):
        compile(p.prepare_source((c.BE/'prepare.py').read_text()),'CPU_PREPARE_ONLY','exec')
    def test_actual_project_temp_append_and_fsync(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_RECEIPT_') as folder:
            path=Path(folder)/'sample.jsonl';c.append(path,{'test_only':True})
            self.assertEqual(json.loads(path.read_text()),{'test_only':True})
    def test_runtime_migration_guard_rejects_old_new_attempt(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_PRIOR_') as folder:
            base=Path(folder);root=base/'batch_203/run_v1';root.mkdir(parents=True)
            cfg={'candidates':[{'candidate_id':'CPU_FIXTURE'}]}
            c.save(root/'EXECUTION_CONFIG.json',cfg);c.save(root/'INPUT_LOCK.json',{})
            item={'previous_run_root':str(root),'runtime_recheck_unstarted_required':True}
            with patch.object(t.c,'BE',base):
                t.check_prior_unattempted(item,cfg)
                with self.assertRaisesRegex(ValueError,'CANDIDATES_CHANGED'):
                    t.check_prior_unattempted(item,{'candidates':[{'candidate_id':'DIFFERENT'}]})
                c.save(root/'LAUNCH_RESERVATION.json',{'test_fixture':True})
                with self.assertRaisesRegex(ValueError,'NOW_ATTEMPTED'):t.check_prior_unattempted(item,cfg)
        with self.assertRaisesRegex(ValueError,'TARGET_REQUIRED'):
            t.check_prior_unattempted({'runtime_recheck_unstarted_required':True},cfg)
        import inspect
        self.assertIn('check_prior_unattempted(item,cfg)',inspect.getsource(t.check_inputs))

if __name__=='__main__':unittest.main(verbosity=2)
