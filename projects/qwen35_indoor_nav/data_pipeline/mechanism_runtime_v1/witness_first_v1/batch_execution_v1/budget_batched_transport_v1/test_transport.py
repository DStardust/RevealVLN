import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
t=load('tested_clock_transport',HERE/'transport.py')
p=load('tested_clock_prepare',HERE/'prepare.py')
LIMITS=dict(total_actions=60000,total_seconds=3600,discovery_actions=15000,
            discovery_seconds=1000,certification_actions=20000,certification_seconds=1500)
def cfg():
    return dict(budget_transport='clock_batching_fresh_only_v1',fresh_only=True,resume_allowed=False,
        runtime_allowed=True,executable=True,training_allowed=False,gpu_device=2,
        gpu_uuid='GPU-be1b30d0-517b-b079-871b-de195d35a1a2',supervision_wall_seconds=3900,
        budget=LIMITS.copy(),factory_variant='winding_v1')

class Tests(unittest.TestCase):
    def test_fixed_selection_three_distinct_houses(self):
        self.assertEqual(p.selection(p.SNAPSHOT),([0,34,60],[0,2,4]))
        d=json.loads((p.SNAPSHOT/'CONFIG_DRAFT.json').read_text())
        self.assertEqual(len({d['candidates'][i]['house_id'] for i in [0,34,60]}),3)
    def test_cfg_no_budget_or_resume_relaxation(self):
        t.check_config(cfg())
        for key,value in [('fresh_only',False),('resume_allowed',True),('gpu_device',1),('training_allowed',True)]:
            c=cfg();c[key]=value
            with self.assertRaises(AssertionError):t.check_config(c)
        c=cfg();c['budget']['total_seconds']=3601
        with self.assertRaises(AssertionError):t.check_config(c)
    def test_original_sources_hashes_and_exact_worker_adapter(self):
        shared=t.base('shared.py');t.base('prepare.py');t.base('readiness_v1.py')
        source=shared.worker_source((t.WF/'assembly_v1/worker.py').read_text())
        self.assertIn('assert len(retained)==27',source)
        self.assertIn("split='candidate_fit_pool'",source)
        self.assertIn('loader.validate_supervision_contract()',source)
        self.assertIn('factory.replay_seeds(candidate)',source)
        self.assertNotIn('resume=True',source)
        self.assertIn("Journal(out/'journal',cfg)",source)
    def test_build_worker_cpu_identity(self):
        worker=t.build_worker(t.BE/'batch_07',cfg())
        self.assertEqual(worker.WitnessFactory.__name__,'WindingBalancedFactory')
        self.assertEqual(worker.PartialTraceRunner.__name__,'PartialTraceRunner')
        self.assertEqual(worker.export_family.__module__,'exporter')
        self.assertEqual(set(inspect.signature(worker.durable_budget).parameters),{'journal','limits','clock'})
    def test_original_supervisor_resource_bounds(self):
        ready=t.base('readiness_v1.py');worker=ready.build_supervisor(t.BE/'batch_07',cfg())
        self.assertEqual(worker.UUID,cfg()['gpu_uuid'])
        self.assertIn('OWN_GPU_MEMORY_UPPER_BOUND',(t.RUNTIME/'compact_loop_v2/run.py').read_text())
        source=t.base('shared.py').supervisor_source((t.RUNTIME/'compact_loop_v2/run.py').read_text(),2)
        for text in ('sample[\'elapsed\'] < 3900',"sample['disk_bytes'] < 7*1024**3",'8*1024**2','m <= 768','sum(external) <= 2048','upper < 4096'):
            self.assertIn(text,source)
    def test_prepare_changes_only_explicit_config_and_closure(self):
        source=(t.BE/'prepare.py').read_text();new=p.adapted_prepare_source(source)
        compile(new,'newprepare','exec')
        self.assertIn('code += extra_code',new)
        self.assertIn('resume_allowed=False',new)
        with self.assertRaises(AssertionError):p.adapted_prepare_source('')
    def test_no_state_argument(self):
        with self.assertRaises(TypeError):t.bind_budget(None,LIMITS,clock=lambda:0,journal_class=object,state={})
    def test_real_sync_binding_pending_checks_and_no_restart(self):
        worker=t.build_worker(t.BE/'batch_07',cfg())
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory)/'journal'
            with worker.Journal(root,{'cpu_only':True}) as journal:
                clock=[0.];ledger=worker.durable_budget(journal,LIMITS,clock=lambda:clock[0])
                ledger.start_bundle('cpu','certification');before=len(journal._records)
                clock[0]=1.;ledger.check_time();self.assertEqual(len(journal._records),before)
                ledger.reserve_action();self.assertEqual(len(journal._records),before+1)
                self.assertEqual(journal._records[-1]['payload'],ledger.snapshot())
                with self.assertRaises(AssertionError):worker.durable_budget(journal,LIMITS,clock=lambda:1.)
                with self.assertRaises(FileExistsError):worker.Journal(root,{'cpu_only':True})
                ledger.finish_phase();self.assertEqual(ledger.snapshot(),ledger.durable_snapshot())
    def test_non_journal_denied(self):
        worker=t.build_worker(t.BE/'batch_07',cfg())
        with self.assertRaises(AssertionError):worker.durable_budget(object(),LIMITS,clock=lambda:0.)
    def test_fault_poison_no_action(self):
        worker=t.build_worker(t.BE/'batch_07',cfg())
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            with worker.Journal(Path(directory)/'journal',{'cpu_only':True}) as journal:
                ledger=worker.durable_budget(journal,LIMITS,clock=lambda:0.);ledger.start_bundle('cpu')
                calls=[]
                with patch.object(journal,'append',side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):ledger.step_after_durable_reservation(lambda:calls.append(1))
                self.assertEqual(calls,[]);self.assertTrue(ledger.poisoned)
                self.assertEqual(ledger.snapshot()['total_reserved_actions'],1)
    def test_wrong_target_before_io(self):
        with self.assertRaises(AssertionError):t.check_inputs(HERE)
    def test_generated_wrappers_compile(self):
        for entry in ('worker_main','run_main'):compile(p.wrapper(entry),'wrapper','exec')

if __name__=='__main__':unittest.main()
