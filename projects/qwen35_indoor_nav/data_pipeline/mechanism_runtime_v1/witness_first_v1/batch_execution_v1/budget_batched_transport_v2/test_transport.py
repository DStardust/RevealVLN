import importlib.util
import inspect
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
t=load('test_batch08_transport',HERE/'transport.py');p=load('test_batch08_prepare',HERE/'prepare.py')
def cfg():
    return {'budget_transport':'clock_batching_fresh_only_v1','fresh_only':True,'resume_allowed':False,
        'runtime_allowed':True,'executable':True,'training_allowed':False,'gpu_device':2,
        'gpu_uuid':'GPU-be1b30d0-517b-b079-871b-de195d35a1a2','supervision_wall_seconds':3900,'factory_variant':'winding_v1',
        'budget':dict(total_actions=60000,total_seconds=3600,discovery_actions=15000,discovery_seconds=1000,certification_actions=20000,certification_seconds=1500)}
class Tests(unittest.TestCase):
    def test_exact_three_new_physical_hubs(self):
        indices,positions,ledger=p.selection(p.SNAPSHOT);self.assertEqual(indices,[0,24,48]);self.assertEqual(positions,[0,1,2])
        self.assertEqual(sum(r['status']=='PREDECLARED_SELECTED' for r in ledger),3)
        self.assertEqual(len(ledger),12)
    def test_exact_transport_reverse(self):
        old=t.ORIGINAL.read_text();new=t.adapted_source(old)
        self.assertEqual(new.replace("BE/'batch_08'","BE/'batch_07'").replace('EXACT_FRESH_BATCH08_ONLY','EXACT_FRESH_BATCH07_ONLY'),old)
    def test_mutated_transport_rejected(self):
        with self.assertRaises(AssertionError):t.adapted_source(t.ORIGINAL.read_text()+'\n')
    def test_wrong_batch_denied_before_gpu(self):
        with self.assertRaises(AssertionError):t.check_inputs(t.BE/'batch_07')
    def test_budget_and_fresh_only_unchanged(self):
        t.check_config(cfg())
        for k,v in [('fresh_only',False),('resume_allowed',True),('training_allowed',True),('gpu_device',1)]:
            c=cfg();c[k]=v
            with self.assertRaises(AssertionError):t.check_config(c)
        c=cfg();c['budget']['total_actions']=60001
        with self.assertRaises(AssertionError):t.check_config(c)
    def test_worker_same_factory_and_trace_runner(self):
        worker=t.build_worker(t.BE/'batch_08',cfg())
        self.assertEqual(worker.WitnessFactory.__name__,'WindingBalancedFactory')
        self.assertEqual(worker.PartialTraceRunner.__name__,'PartialTraceRunner')
        self.assertEqual(set(inspect.signature(worker.durable_budget).parameters),{'journal','limits','clock'})
    def test_no_resume_argument(self):
        with self.assertRaises(TypeError):t.bind_budget(None,{},clock=lambda:0,journal_class=object,state={})
    def test_exact_original_preparer_scope_kept(self):
        old=(t.BE/'prepare.py').read_text();new=p.adapted_prepare_source(old,[0,1,2])
        self.assertIn("snapshot.is_relative_to(WF)",new);self.assertIn('code += extra_code',new)
        self.assertIn('next12_positions=[0, 1, 2]',new);compile(new,'CPU_PREPARE','exec')
    def test_original_27_and_resource_guards(self):
        shared=t.base('shared.py');worker=shared.worker_source((t.WF/'assembly_v1/worker.py').read_text())
        self.assertIn('assert len(retained)==27',worker)
        supervisor=shared.supervisor_source((t.RUNTIME/'compact_loop_v2/run.py').read_text(),2)
        for token in ("sample['elapsed'] < 3900","sample['disk_bytes'] < 7*1024**3",'upper < 4096','sum(external) <= 2048'):
            self.assertIn(token,supervisor)
    def test_wrappers_compile(self):
        for entry in ('worker_main','run_main'):compile(p.wrapper(entry),'BATCH08_WRAPPER','exec')
if __name__=='__main__':unittest.main()
