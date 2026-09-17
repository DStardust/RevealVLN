import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location('scaleout_test_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=load('planner');c=load('continuation')
def row(key,hub,program):
    return {'candidate_id':key,'house_id':'house','configuration':{'u_position':[hub*2,0,0]},
        'roles':{k:{'mpcat40':str(v),'room':'office'} for k,v in [('anchor_A',program),('anchor_B',program+1),('terminal',program+2)]}}
def entry(key,hub,program,index=0,pool='pool'):
    return {'row':row(key,hub,program),'source_snapshot':pool,'source_index':index,'pool_order':0}
class ScaleoutTests(unittest.TestCase):
    def test_01_cpu_only_import(self):
        with patch('subprocess.run',side_effect=AssertionError('NO_PROCESS')),patch('subprocess.check_output',side_effect=AssertionError('NO_GPU')):load('planner');load('continuation')
    def test_02_source_hashed(self):self.assertEqual(p.sha(p.ORIGINAL),p.ORIGINAL_SHA)
    def test_03_prior_renamed_not_retry(self):
        a=entry('old',0,0);b=entry('new',0,0)
        jobs,l=p.plan([b],[{'row':a['row'],'source':'failure'}]);self.assertFalse(jobs);self.assertEqual(l[0]['status'],'EXCLUDED_ALL_PRIOR_FROZEN_CONFIGS_NO_RETRY')
    def test_04_swap_alias_padding_not_innovation(self):
        a=entry('old',0,0);b=copy.deepcopy(a);b['row']['candidate_id']='new';b['row']['roles']['anchor_A'],b['row']['roles']['anchor_B']=b['row']['roles']['anchor_B'],b['row']['roles']['anchor_A'];b['row']['padding']='different'
        jobs,l=p.plan([b],[{'row':a['row']}]);self.assertFalse(jobs)
    def test_05_cross_bank_fills_late_two_hubs(self):
        entries=[entry(str(h)+'_'+str(i),h,i*10,i,'a' if h<2 else 'b') for h in range(3) for i in range(4)]
        jobs,l=p.plan(entries,[]);self.assertEqual(len(jobs),4);self.assertEqual([x['gpu'] for x in jobs],[3,4,5,7]);self.assertEqual(len({k for x in jobs for k in x['candidate_ids']}),12)
    def test_06_non_multiple_three_preserved(self):
        entries=[entry(str(h)+'_'+str(i),h,i*10,i) for h in range(4) for i in range(4)]+[entry('extra',4,90)]
        jobs,l=p.plan(entries,[]);self.assertEqual(len(jobs),5);self.assertEqual(sum(x['status']=='PENDING_NO_COMPLETE_THREE_HUB_BATCH' for x in l),2)
    def test_07_only_two_hubs_not_fake_three(self):
        jobs,l=p.plan([entry(str(h)+'_'+str(i),h,i*10,i) for h in range(2) for i in range(10)],[]);self.assertFalse(jobs)
    def test_08_near_hubs_not_distinct(self):
        entries=[entry(str(i),i,0) for i in range(3)];entries[1]['row']['configuration']['u_position']=[.9,0,0]
        jobs,l=p.plan(entries,[]);self.assertFalse(jobs)
    def test_09_deterministic(self):
        entries=[entry(str(h)+'_'+str(i),h,i*10,i) for h in range(10) for i in range(7)]
        self.assertEqual(p.plan(entries,[]),p.plan(entries,[]))
    def test_10_mutable_sources_forbidden(self):
        for batch in ('batch_100','batch_202','batch_300'):
            root=p.BE/batch/'run_v1';p.assert_no_rolling_inputs({str(root/'EXECUTION_CONFIG.json'):'hash'})
            for file in ('journal/HEAD.json','journal/events.jsonl','PROGRESS.json','result.json','INPUT_LOCK.json'):
                with self.assertRaises(AssertionError):p.assert_no_rolling_inputs({str(root/file):'hash'})
    def test_11_migration_exact_mapping(self):
        self.assertEqual(len(c.MAPPINGS),17);self.assertEqual(c.MAPPINGS[0],(1,203,220));self.assertEqual(c.MAPPINGS[-1],(2,111,247))
        self.assertFalse(any(old in (103,202) for _,old,_ in c.MAPPINGS))
    def test_12_real_unstarted_current_migrations_readonly(self):
        for gpu,old,new in c.MAPPINGS:
            cfg,lock=c.check_unstarted(p.BE/('batch_'+str(old))/'run_v1');self.assertEqual(cfg['gpu_device'],gpu)
    def test_13_fake_extra_launch_file_rejected(self):
        root=HERE/'TEST_FIXTURES';root.mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix='UNSTARTED_',dir=root))
        cfg=out/'EXECUTION_CONFIG.json';cfg.write_text('{}')
        (out/'INPUT_LOCK.json').write_text(json.dumps({str(cfg):p.sha(cfg)}))
        c.check_unstarted(out)
        (out/'LAUNCH_RESERVATION.json').write_text('{}')
        with self.assertRaises(AssertionError):c.check_unstarted(out)
    def test_14_tampered_unstarted_config_rejected(self):
        root=HERE/'TEST_FIXTURES';root.mkdir(exist_ok=True);out=Path(tempfile.mkdtemp(prefix='BAD_HASH_',dir=root))
        cfg=out/'EXECUTION_CONFIG.json';cfg.write_text('{}');(out/'INPUT_LOCK.json').write_text(json.dumps({str(cfg):'0'*64}))
        with self.assertRaises(AssertionError):c.check_unstarted(out)
    def test_15_real_current_capacity(self):
        prior=[{'row':r} for path in p.BE.glob('batch_*/run_v1/EXECUTION_CONFIG.json') for r in p.read(path)['candidates']]
        prior += [{'row':r} for r in p.read(p.WF/'short_revisit_v3/run_v1/EXECUTION_CONFIG.json')['candidates']]
        entries=[dict(row=r,source_snapshot=str(pool),source_index=i,pool_order=n) for n,pool in enumerate(p.q.POOLS) for i,r in enumerate(p.read(pool/'CONFIG_DRAFT.json')['candidates'])]
        jobs,ledger=p.plan(entries,prior);self.assertEqual(len(entries),442);self.assertLessEqual(len(jobs),114)
        self.assertEqual(len({i for j in jobs for i in j['candidate_ids']}),3*len(jobs))
        self.assertTrue(all(not p.prior_matches(e['row'],prior) for e in entries if any(e['row']['candidate_id'] in j['candidate_ids'] for j in jobs)))
if __name__=='__main__':unittest.main()
