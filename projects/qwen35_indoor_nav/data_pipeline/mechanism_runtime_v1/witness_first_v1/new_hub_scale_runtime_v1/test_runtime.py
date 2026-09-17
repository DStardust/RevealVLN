"""CPU-only negative/differential tests; no GPU, sim, invented replay, or launch."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location('scale_test_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('common');a=load('adapters');bank=load('bank');r=load('runtime');prep=load('prepare')
JOB=HERE/'jobs/scout_000'

def fixture():
    base=c.WF/'new_hub_scout_v1/run_v1';cfg=c.read(base/'EXECUTION_CONFIG.json');house=cfg['new_hub_plan']['house_id']
    folder=base/'houses'/house
    return cfg,c.read(folder/'result.json'),c.read(folder/'FROZEN_HUB_CONFIGS.json')

def function(source,name):
    tree=ast.parse(source);return ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),include_attributes=False)

class Tests(unittest.TestCase):
    def test_import_has_no_gpu_or_subprocess(self):
        with patch('subprocess.Popen',side_effect=AssertionError('NO_CHILD')),patch('subprocess.check_output',side_effect=AssertionError('NO_GPU')):
            for name in ['common','adapters','bank','runtime','prepare','audit']:load(name)
    def test_all_sources_compile(self):
        for p in HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    def test_guard_exact_function(self):
        old=a.a.supervisor_source();new=a.supervisor_source(JOB)
        for name in ['check_gpu','parse_gpu','disk_size','gpu']:
            if name=='gpu':continue
            self.assertEqual(function(old,name),function(new,name))
    def test_worker_only_job_and_gpu(self):
        source=a.worker_source(JOB)
        source=source.replace(repr(str(JOB)),repr(str(c.OLD)))
        source=source.replace("HabitatBackend(candidate['scene_glb'],7,candidate['roles']","HabitatBackend(candidate['scene_glb'],2,candidate['roles']")
        self.assertEqual(source,a.a.worker_source())
    def test_supervisor_scope_and_budgets(self):
        module=a.build_supervisor(JOB);self.assertEqual(module.HERE,JOB);self.assertEqual(module.UUID,c.UUID)
        source=a.supervisor_source(JOB)
        for text in ["< 3000, 'WALL_BUDGET'","< 7*1024**3, 'DISK_BUDGET'","< 8*1024**2, 'RSS_BUDGET'",'proc.wait(timeout=20)','proc.wait(timeout=10)']:
            self.assertIn(text,source)
    def test_original_guard_reject_external(self):
        sup=a.build_supervisor(JOB)
        with self.assertRaisesRegex(AssertionError,'EXTERNAL_RESOURCE_LOAD'):
            sup.check_gpu(dict(processes={1:dict(mib=769)},memory_mib=1000))
    def test_worker_and_store_are_job_scoped(self):
        with patch('subprocess.Popen',side_effect=AssertionError('NO_CHILD')),patch('subprocess.check_output',side_effect=AssertionError('NO_GPU')):
            worker=a.build_worker(JOB,c.prepared_config(c.item_for(JOB)))
            self.assertEqual(worker.HERE,JOB)
            cls=worker.store_class();self.assertEqual(cls.__init__.__globals__['FEEDBACK_ROOT'],JOB)
    def test_sealed_holder_restore_api(self):
        holder=c.load('scale_test_final_holder',c.WF/'special_scale_holder_v1/transport.py')
        self.assertTrue(callable(holder.make_ops));self.assertTrue(callable(holder.lease_module))
        self.assertEqual(c.sha(c.WF/'special_scale_holder_v1/SHA256SUMS'),c.PINNED[c.WF/'special_scale_holder_v1/SHA256SUMS'])
    def test_scoped_paths_reject(self):
        for p in [HERE/'jobs/scout_000/x',HERE/'jobs/scout_abc',HERE/'other/scout_000']:
            with self.assertRaises(ValueError):c.job_root(p)
    def test_exact_change_reject_ambiguous(self):
        for source in ['x x','y']:
            with self.assertRaises(ValueError):c.exact(source,'x','z')
    def test_all_43_source_configs_preserved(self):
        rows=c.read(c.QUEUE)['jobs'];self.assertEqual(len(rows),43)
        mutable={'gpu_device','gpu_uuid','supervision_wall_seconds','total_disk_budget_bytes','max_hubs_per_house','intended_output_root','runtime_transport_version'}
        for row in rows:
            old=c.read(row['prepared_config']);new=c.prepared_config(row)
            self.assertEqual({k:v for k,v in old.items() if k not in mutable},{k:v for k,v in new.items() if k not in mutable})
            self.assertEqual(new['budget'],c.BUDGET);self.assertFalse(new['runtime_allowed']);self.assertEqual(new['supervision_wall_seconds'],3000)
    def test_actual_original_four_hubs(self):
        self.assertEqual(bank.coverage(*fixture())['actual_hubs'],4)
    def test_subset_allowed_only_explicit_no_reachable_group(self):
        cfg,result,frozen=fixture();frozen=copy.deepcopy(frozen);result=copy.deepcopy(result)
        removed=frozen['hubs'][1:];frozen['hubs']=frozen['hubs'][:1]
        result['hubs']=result['hubs'][:1];result['selected_hubs']=1
        for g in frozen['geometry']['geometry_checks']:
            if any(g['position']==h['position'] for h in removed):g['reachable_groups']=0
        report=bank.coverage(cfg,result,frozen)
        self.assertEqual(report['actual_hubs'],1);self.assertEqual(len(report['positions']),4)
        self.assertEqual(sum(p['status']=='NO_REACHABLE_ROLE_GROUP' for p in report['positions']),3)
    def test_resource_censor_rejected(self):
        cfg,result,frozen=fixture();result['status']='RESOURCE_CENSORED'
        with self.assertRaisesRegex(ValueError,'NATURAL_HOUSE_CLOSURE'):bank.coverage(cfg,result,frozen)
    def test_missing_geometry_rejected(self):
        cfg,result,frozen=fixture();frozen['geometry']['geometry_checks'].pop()
        with self.assertRaisesRegex(ValueError,'ALL_POSITIONS_GEOMETRY'):bank.coverage(cfg,result,frozen)
    def test_missing_hub_terminal_rejected(self):
        cfg,result,frozen=fixture();result['hubs'].pop()
        with self.assertRaisesRegex(ValueError,'ALL_SELECTED_HUBS_COMPLETE'):bank.coverage(cfg,result,frozen)
    def test_unexplained_dropped_hub_rejected(self):
        cfg,result,frozen=fixture();frozen['hubs'].pop();result['hubs'].pop();result['selected_hubs']=3
        with self.assertRaisesRegex(ValueError,'UNEXPLAINED_UNSELECTED_POSITION'):bank.coverage(cfg,result,frozen)
    def test_hub_substitution_rejected(self):
        cfg,result,frozen=fixture();frozen['hubs'][0]['position']=[100,100,100]
        with self.assertRaisesRegex(ValueError,'NO_HUB_SUBSTITUTION'):bank.coverage(cfg,result,frozen)
    def test_bank_original_checks_still_present(self):
        source=bank.adapted(JOB)
        for marker in ['BANK_TRACE_COMMIT_BINDING','SEMANTIC_INSTANCE_IDENTITY','CONTINUATION_ACTION_CAP','STORED_OBSERVATION_QUERY_LENGTH_GT160','c.select_programs(viable,24)','c.acceptance.parse_journal(raw,head,cfg)']:
            self.assertIn(marker,source)
    def test_active_source_cannot_build(self):
        with patch.object(bank,'closure',side_effect=ValueError('NOT_CLOSED')):
            with self.assertRaisesRegex(ValueError,'NOT_CLOSED'):bank.main(JOB)
        self.assertFalse((JOB/'bank').exists())
    def test_bank_zero_hubs_reject(self):
        with patch.object(bank,'closure',return_value=({}, {},dict(actual_hubs=0))):
            with self.assertRaisesRegex(ValueError,'NO_ACTUAL_HUBS'):bank.main(JOB)
    def test_authorization_no_old_cert_budget(self):
        auth=dict(approved=True,node='Q35N_NEW_HUB_SCALE_RUNTIME_V1',gpu=7,gpu_uuid=c.UUID,
            queue_path=str(c.QUEUE),queue_sha256='test',budget=dict(c.BUDGET,total_seconds=3600))
        with patch.object(c,'read',return_value=auth),patch.object(c,'sha',return_value='test'):
            with self.assertRaisesRegex(ValueError,'ORIGINAL_SCOUT_BUDGET'):c.authorization(HERE/'TEST_AUTH.json')

    def fake_run(self,fail_lease=False,fail_bank=False):
        # Clearly isolated TEST_FIXTURE, fake calls only; never simulator data.
        order=[]
        with tempfile.TemporaryDirectory(prefix='TEST_FIXTURE_',dir=HERE) as directory:
            job=Path(directory);mutex=job/'gpu_7.lock';launch=dict(previous_job=None)
            cfg={'TEST_FIXTURE':True};auth={'shared_gpu_lock':str(mutex)}
            (job/'SOURCE_LOCK.json').write_text('{}')
            class Lease:
                def run_lease(self,out,identity,module,execute,ops):
                    order.append('borrow')
                    try:
                        execute()
                        if fail_lease:raise RuntimeError('TEST_LEASE_FAILURE')
                    finally:order.append('restore')
            holder=types.SimpleNamespace(lease_module=lambda d:(Lease(),{}),make_ops=lambda d,l:object())
            def bank_main(path):
                order.append('bank')
                if fail_bank:raise RuntimeError('TEST_BANK_FAILURE')
            def fake_load(name,path):
                if Path(path).name=='transport.py':return holder
                if Path(path).name=='adapters.py':return types.SimpleNamespace(build_supervisor=lambda job:types.SimpleNamespace(main=lambda:order.append('supervisor')))
                if Path(path).name=='bank.py':return types.SimpleNamespace(main=bank_main)
                raise AssertionError('UNEXPECTED_LOAD')
            with patch.object(r.c,'job_root',side_effect=lambda p:Path(p)),patch.object(r,'check',return_value=(cfg,launch,auth)),patch.object(r,'resolve_identity',return_value=({},{})),patch.object(r.c,'load',side_effect=fake_load):
                if fail_lease or fail_bank:
                    with self.assertRaises(RuntimeError):
                        r.run_main(job);r.audit_main(job)
                else:r.run_main(job);r.audit_main(job)
            launch_result=json.loads((job/'run_v1/LAUNCH_RESULT.json').read_text())
            self.assertTrue(launch_result['supervisor_returned'])
            if fail_lease:self.assertIn('TEST_LEASE_FAILURE',launch_result['error'])
            if fail_bank:self.assertTrue((job/'run_v1/BANK_REJECTED.json').is_file())
        return order
    def test_restore_before_bank(self):
        self.assertEqual(self.fake_run(),['borrow','supervisor','restore','bank'])
    def test_lease_failure_blocks_bank(self):
        self.assertEqual(self.fake_run(fail_lease=True),['borrow','supervisor','restore'])
    def test_bank_failure_is_after_restore_and_recorded(self):
        self.assertEqual(self.fake_run(fail_bank=True),['borrow','supervisor','restore','bank'])

if __name__=='__main__':unittest.main()
