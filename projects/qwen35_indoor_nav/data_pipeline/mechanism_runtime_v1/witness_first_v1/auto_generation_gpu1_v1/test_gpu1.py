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
    s=importlib.util.spec_from_file_location('gpu1_cpu_test_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=load('adapter');q=load('queue');t=load('transport');p=load('prepare');audit=load('audit')
def row(cid='x',house='h',point=(0,0,0),a_role='plant',b_role='chair',terminal='table'):
    return {'candidate_id':cid,'house_id':house,'configuration':{'u_position':list(point)},
        'roles':{k:{'mpcat40':v,'room':'office'} for k,v in [('anchor_A',a_role),('anchor_B',b_role),('terminal',terminal)]}}
def gpu1_cfg():
    cfg=json.loads((t.BE/'batch_08/run_v1/EXECUTION_CONFIG.json').read_text())
    cfg['gpu_device']=1;cfg['gpu_uuid']='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8';return cfg
class GPU1Tests(unittest.TestCase):
    def test_01_all_exact_reverse(self):
        for name in ('queue.py','transport.py','prepare.py','audit.py'):
            source=(a.OLD/name).read_text();value=a.exact(source,a.changes(name),a.HASHES[name])
            for old,new in reversed(a.changes(name)):value=value.replace(new,old)
            self.assertEqual(value,source)
    def test_02_changed_source_rejected(self):
        for name in ('queue.py','transport.py','prepare.py','audit.py'):
            with self.assertRaises(AssertionError):a.exact((a.OLD/name).read_text()+'\n',a.changes(name),a.HASHES[name])
    def test_03_duplicate_replacement_rejected(self):
        src='x x';h=hashlib.sha256(src.encode()).hexdigest()
        with self.assertRaises(AssertionError):a.exact(src,[('x','y')],h)
    def test_04_import_no_gpu(self):
        with patch('subprocess.run',side_effect=AssertionError('NO_PROCESS')),patch('subprocess.check_output',side_effect=AssertionError('NO_QUERY')):
            for name in ('queue','transport','prepare','audit'):load(name)
    def test_05_gpu1_config_accepted(self):t.private.check_config(gpu1_cfg())
    def test_06_gpu2_config_refused(self):
        cfg=gpu1_cfg();cfg['gpu_device']=2;cfg['gpu_uuid']='GPU-be1b30d0-517b-b079-871b-de195d35a1a2'
        with self.assertRaises(AssertionError):t.private.check_config(cfg)
    def test_07_wrong_uuid_refused(self):
        cfg=gpu1_cfg();cfg['gpu_uuid']='GPU-wrong'
        with self.assertRaises(AssertionError):t.private.check_config(cfg)
    def test_08_budget_not_relaxed(self):
        for field,value in [('total_actions',60001),('total_seconds',3601),('certification_actions',20001)]:
            cfg=gpu1_cfg();cfg['budget'][field]=value
            with self.assertRaises(AssertionError):t.private.check_config(cfg)
    def test_09_new_batch_numbers_deterministic(self):
        rows=[(i,row(str(i),house=str(i))) for i in range(6)]
        batches,ledger=q.choose([(Path('/pool'),rows)],[],2)
        self.assertEqual([x['id'] for x in batches],['batch_200','batch_201'])
        self.assertEqual([x['gpu'] for x in batches],[1,1])
        self.assertEqual((batches,ledger),q.choose([(Path('/pool'),rows)],[],2))
    def test_10_semantic_swap_cannot_retry(self):
        old=row('old');new=row('new',a_role='chair',b_role='plant')
        batches,ledger=q.choose([(Path('/pool'),[(0,new)])],[{'row':old,'attempted':True}],1)
        self.assertFalse(batches);self.assertEqual(ledger[0]['status'],'EXCLUDED_PRIOR_ATTEMPT_NO_RETRY')
    def test_11_near_position_not_new_hub(self):
        self.assertFalse(q.m.distinct_hub(row(),row(point=(.999,0,0))))
        self.assertTrue(q.m.distinct_hub(row(),row(point=(1,0,0))))
    def test_12_actual_gpu2_full_reservation(self):
        old=json.loads((a.OLD/'queue_v1/QUEUE.json').read_text());prior=[]
        for b in old['batches']:
            root=t.BE/b['id']/'run_v1'
            for r in json.loads((root/'EXECUTION_CONFIG.json').read_text())['candidates']:
                prior.append({'candidate_id':r['candidate_id'],'run_root':str(root)})
        lock={};self.assertEqual(len(a.verify_gpu2_reservations(prior,lock)),36)
        with self.assertRaises(AssertionError):a.verify_gpu2_reservations(prior[:-1],{})
    def test_13_original_guard_cpu_examples(self):
        # Compile the unchanged original supervisor; never call main or gpu().
        mod=t.base('readiness_v1.py').build_supervisor(t.BE/'batch_200',gpu1_cfg())
        snapshot={'memory_mib':1200,'processes':{'a':{'mib':650},'b':{'mib':246},'c':{'mib':246}}}
        self.assertEqual(mod.check_gpu(snapshot),58)
        excessive=copy.deepcopy(snapshot);excessive['processes']['a']['mib']=769
        with self.assertRaises(AssertionError):mod.check_gpu(excessive)
        excessive={'memory_mib':2300,'processes':{str(i):{'mib':700} for i in range(3)}}
        with self.assertRaises(AssertionError):mod.check_gpu(excessive)
    def test_14_actual_old_save_output_scope(self):
        g=audit.m.t.load('gpu1_test_original_gate3',audit.m.GATE)
        root=audit.m.GATE.parent/'auto_generation_gpu1_v1';root.mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix='TEST_FIXTURE_CPU_SCOPE_',dir=root))
        value={'kind':'TEST_FIXTURE','physical_family':False,'scientific_pass':False}
        dest=out/'CPU_SAVE_ONLY.json';g.gate.old.save(dest,value)
        self.assertEqual(json.loads(dest.read_text()),value)
        with self.assertRaises(Exception):g.gate.old.save(HERE/'TEST_FIXTURE_FORBIDDEN.json',value)
        self.assertFalse((HERE/'TEST_FIXTURE_FORBIDDEN.json').exists())
    def test_15_guard_source_change_only_device(self):
        src=t.m.ORIGINAL.read_text();value=t.m.adapted_source(src)
        for old,new in reversed(t.m.source_changes()):value=value.replace(new,old)
        self.assertEqual(value,src)
        self.assertIn(a.GPU1_ASSERT,t.m.adapted_source(src))
        self.assertNotIn(a.GPU2_ASSERT,t.m.adapted_source(src))
    def test_16_preparation_launcher_argv(self):
        src=a.exact((a.OLD/'prepare.py').read_text(),a.changes('prepare.py'),a.HASHES['prepare.py'])
        self.assertIn("command=[str(PYTHON),'-I','-B',str(batch/'run.py')]",src)
        self.assertIn("module.prepare(snapshot,item['id'],[0,1,2],1,'winding_v1')",src)
    def test_17_prior_dependencies_verified(self):
        deps=a.original_dependencies();self.assertEqual(len(deps),7)
        self.assertTrue(all(p.is_file() for p in deps))
    def test_18_gpu2_active_journal_never_read(self):
        src=a.exact((a.OLD/'queue.py').read_text(),a.changes('queue.py'),a.HASHES['queue.py'])
        self.assertIn("if run.parent.name not in GPU2_RESERVED_BATCHES and (run/'journal/HEAD.json').exists():",src)
        root=t.BE/'batch_100/run_v1'
        a.assert_no_gpu2_mutable({str(root/'EXECUTION_CONFIG.json'):'test'})
        for relative in ('journal/HEAD.json','journal/events.jsonl','PROGRESS.json','INPUT_LOCK.json'):
            with self.assertRaises(AssertionError):a.assert_no_gpu2_mutable({str(root/relative):'test'})
if __name__=='__main__':unittest.main()
