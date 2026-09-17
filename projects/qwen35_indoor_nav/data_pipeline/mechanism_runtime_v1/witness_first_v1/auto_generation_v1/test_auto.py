import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location('test_auto_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
q=load('queue');t=load('transport');p=load('prepare')
def row(cid='x',house='h',point=(0,0,0),a='plant',b='chair',terminal='table'):
    return {'candidate_id':cid,'house_id':house,'configuration':{'u_position':list(point)},
        'roles':{k:{'mpcat40':v,'room':'office'} for k,v in [('anchor_A',a),('anchor_B',b),('terminal',terminal)]}}
class TestAuto(unittest.TestCase):
    def test_import_no_gpu(self):
        with patch('subprocess.run',side_effect=AssertionError('GPU_OR_PROCESS')),patch('subprocess.check_output',side_effect=AssertionError('GPU_QUERY')):
            load('queue');load('transport');load('prepare');load('audit')
    def test_exact_reverse(self):
        original=t.ORIGINAL.read_text();adapted=t.adapted_source(original)
        for old,new in reversed(t.source_changes()):adapted=adapted.replace(new,old)
        self.assertEqual(adapted,original)
    def test_changed_frozen_source_reject(self):
        with self.assertRaises(AssertionError):t.adapted_source(t.ORIGINAL.read_text()+'\n')
    def test_guard_unchanged(self):
        source=t.adapted_source(t.ORIGINAL.read_text())
        for text in ("cfg['gpu_device']==2",'supervision_wall_seconds\']==3900',"total_actions=60000,total_seconds=3600",'discovery_actions=15000',"certification_actions=20000",'WORKER_FRESH_ONLY'):
            self.assertIn(text,source)
        self.assertIn('len(lock)<=2048',source)
    def test_gpu1_refused(self):
        cfg=json.loads((t.BE/'batch_08/run_v1/EXECUTION_CONFIG.json').read_text());cfg['gpu_device']=1
        with self.assertRaises(AssertionError):t.private.check_config(cfg)
    def test_prepare_source_preserved(self):
        source=(t.BE/'prepare.py').read_text();adapted=p.adapted_prepare_source(source)
        self.assertIn('code += extra_code',adapted);self.assertIn("auto_retry=False",adapted)
    def test_physical_not_inferred_from_freeze(self):
        records=[{'kind':'freeze','payload':{'x':{'attempt':1,'status':'discovering'}}}]
        attempted,physical=q.attempted_ids(records)
        self.assertEqual(attempted,{'x'});self.assertEqual(physical,set())
    def test_physical_action(self):
        a,b=q.attempted_ids([{'kind':'action_completed','payload':{'bundle':'x'}}]);self.assertEqual(a,b);self.assertEqual(a,{'x'})
    def test_reservation_not_attempt(self):
        _,ledger=q.choose([(Path('/pool'),[(0,row())])],[{'row':row(),'attempted':False,'evidence':'reserved'}],1)
        self.assertEqual(ledger[0]['status'],'RESERVED_PRIOR_FROZEN_NOT_PHYSICAL')
    def test_failed_attempt_no_retry_even_renamed_or_swapped(self):
        r=row('new',a='chair',b='plant')
        _,ledger=q.choose([(Path('/pool'),[(0,r)])],[{'row':row('old'),'attempted':True}],1)
        self.assertEqual(ledger[0]['status'],'EXCLUDED_PRIOR_ATTEMPT_NO_RETRY')
    def test_near_same_hub_not_independent(self):
        self.assertFalse(q.distinct_hub(row(),row(point=(.99,0,0))))
        self.assertTrue(q.distinct_hub(row(),row(point=(1,0,0))))
    def test_need_three_different_hubs(self):
        rs=[(i,row(str(i),point=(0,0,0),terminal=str(i))) for i in range(5)]
        batches,ledger=q.choose([(Path('/pool'),rs)],[],1);self.assertEqual(batches,[])
    def test_deterministic_pool_round_robin(self):
        pools=[(Path('/pool'+str(k)),[(i,row(str(k)+'_'+str(i),house=str(i),terminal=str(k))) for i in range(6)]) for k in range(2)]
        a,l=q.choose(pools,[],4);b,_=q.choose(pools,[],4)
        self.assertEqual(a,b);self.assertEqual([r['source_snapshot'] for r in a],['/pool0','/pool1','/pool0','/pool1'])
        self.assertEqual(len({x for b in a for x in b['candidate_ids']}),12)
    def test_queue_bound(self):
        with self.assertRaises(AssertionError):q.choose([],[],13)
    def test_language_actual_structures_unchanged(self):
        cfg=q.read(q.POOLS[0]/'CONFIG_DRAFT.json');v=q.load('test_verbalizer',q.VERBALIZER,q.VERBALIZER_SHA)
        original=copy.deepcopy(cfg['candidates'][0]);new,revision=q.normalize(original,v)
        for key in ('roles','expected_eligible','components','balance','configuration'):
            self.assertEqual(new[key],original[key])
        self.assertEqual(original,cfg['candidates'][0]);self.assertTrue(revision['task_program_unchanged'])
    def test_command_python_project_scope(self):
        self.assertTrue(p.PYTHON.is_file());self.assertTrue(p.PYTHON.is_relative_to(q.ROOT))
    def test_wrapper_absolute_fixed_source(self):
        source=p.wrapper('run_main');self.assertIn(str(HERE/'transport.py'),source)
        with self.assertRaises(AssertionError):p.wrapper('other')
if __name__=='__main__':unittest.main()
