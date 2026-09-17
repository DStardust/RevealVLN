"""CPU-only semantic dedup and direction-screening tests; no new physical data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('multi_test_core',HERE/'core.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
RUN=c.WF/'batch_execution_v1/batch_02r2/run_v1'
CFG=json.loads((RUN/'EXECUTION_CONFIG.json').read_text())['candidates'][0]
TRACE=json.loads((RUN/'bundles'/CFG['candidate_id']/'traces/000001.json').read_text())
def comp():return c.Compiler({k:(v['mpcat40'],v['room']) for k,v in CFG['roles'].items()},CFG['tasks'],CFG['expected_eligible'])
def fake_spin(role=None,count=256,same_id=True):
    # Explicit synthetic CPU fixture based on real schema/poses, never exported.
    trace=copy.deepcopy(TRACE)
    for obs in trace['observations']:obs['pixels']={}
    if role:
        ident=CFG['expected_eligible'][role][0]
        trace['observations'][0]['pixels']={str(ident):count}
        trace['observations'][1]['pixels']={str(ident if same_id else ident+100000):count}
    return {'direction':'L','trace':trace,'trace_ref':'TEST_FIXTURE_NOT_PHYSICAL','sha256':'0'*64}
def proposal():return {'house_id':'h','hub_key':'hub','hub_pose':{'position':[0,0,0]},'roles':copy.deepcopy(CFG['roles'])}
def candidate(group,length,index):
    return {'canonical_program_id':group,'history_actions_before_public_tail':length,'max_continuation_actions':20,
        'source_proposal_id':str(index),'candidate_id':'c'+str(index),'balance':{'k':2,'j':1}}

class Tests(unittest.TestCase):
    def test_alias_and_anchor_swap_same_program(self):
        first=proposal();second=copy.deepcopy(first)
        second['roles']['anchor_A'],second['roles']['anchor_B']=second['roles']['anchor_B'],second['roles']['anchor_A']
        second['roles']['anchor_A']['raw_match']['value']='synthetic_alias'
        self.assertEqual(c.semantic_id(first),c.semantic_id(second))
    def test_irrelevant_role_and_padding_not_new_semantic_program(self):
        first=proposal();second=copy.deepcopy(first);second['roles']['irrelevant']['room']='synthetic_room'
        second.update(seed=999,padding=['L']*24)
        self.assertEqual(c.semantic_id(first),c.semantic_id(second))
    def test_raw_observation_hub_key_is_not_new_physical_hub(self):
        first=proposal();second=copy.deepcopy(first);second['hub_key']='different_rgb_or_semantic_hash'
        self.assertEqual(c.semantic_id(first),c.semantic_id(second))
    def test_terminal_room_or_anchor_category_distinct(self):
        first=proposal();second=copy.deepcopy(first);second['roles']['terminal']['room']='synthetic_new_room'
        self.assertNotEqual(c.semantic_id(first),c.semantic_id(second))
    def test_per_group_minimal_then_24_cap(self):
        rows=[candidate(str(i),40,i) for i in range(30)]+[candidate('0',30,999)]
        selected,rejected=c.select_programs(rows)
        self.assertEqual(len(selected),24);self.assertEqual(len({x['canonical_program_id'] for x in selected}),24)
        self.assertEqual(next(x for x in selected if x['canonical_program_id']=='0')['candidate_id'],'c999')
        self.assertEqual(len(rejected),7)
    def test_256_two_consecutive_same_instance_rejects(self):
        ok,evidence=c.spin_verdict(comp(),['L'],[fake_spin('anchor_A')])
        self.assertFalse(ok);self.assertEqual(evidence['L']['status'],'REJECTED_OBSERVED_ANCHOR_EVENT')
    def test_255_pixels_not_event(self):
        self.assertTrue(c.spin_verdict(comp(),['L'],[fake_spin('anchor_A',255)])[0])
    def test_two_different_instance_ids_not_event(self):
        self.assertTrue(c.spin_verdict(comp(),['L'],[fake_spin('anchor_A',256,False)])[0])
    def test_terminal_seen_without_stop_allowed(self):
        self.assertTrue(c.spin_verdict(comp(),['L'],[fake_spin('terminal')])[0])
    def test_direction_not_inferred(self):
        ok,evidence=c.spin_verdict(comp(),['R'],[fake_spin('anchor_A')])
        self.assertTrue(ok);self.assertEqual(evidence['R']['status'],'UNTESTED_NO_DIRECTION_WITNESS')
    def test_incomplete_spin_not_accepted(self):
        witness=fake_spin();witness['trace']['complete']=False
        with self.assertRaises(AssertionError):c.spin_verdict(comp(),['L'],[witness])
    def test_committed_prefix_ignores_uncommitted_append(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_FIXTURE_') as tmp:
            base=Path(tmp);run=base/'run';(run/'journal').mkdir(parents=True)
            cfg={};(run/'EXECUTION_CONFIG.json').write_text('{}')
            h=hashlib.sha256(b'{}').hexdigest();(run/'INPUT_LOCK.json').write_text(json.dumps({str(run/'EXECUTION_CONFIG.json'):h}))
            body={'seq':0,'prev':'0'*64,'kind':'__config__','payload':cfg};record=dict(body,hash=hashlib.sha256(c.acceptance.canonical(body)).hexdigest())
            raw=c.acceptance.canonical(record)+b'\n'
            head={'count':1,'last_hash':record['hash'],'byte_length':len(raw),'config_hash':hashlib.sha256(c.acceptance.canonical(cfg)).hexdigest()}
            (run/'journal/HEAD.json').write_text(json.dumps(head));(run/'journal/events.jsonl').write_bytes(raw+b'{uncommitted')
            frozen,records=c.read_committed_prefix(run,base/'snapshot',{})
            self.assertEqual(len(records),1);self.assertEqual((base/'snapshot/events.jsonl').read_bytes(),raw)
    def test_head_atomic_replacement_keeps_opened_committed_prefix(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='CPU_HEAD_FIXTURE_') as tmp:
            base=Path(tmp);run=base/'run';(run/'journal').mkdir(parents=True)
            cfg={};(run/'EXECUTION_CONFIG.json').write_text('{}')
            (run/'INPUT_LOCK.json').write_text(json.dumps({str(run/'EXECUTION_CONFIG.json'):hashlib.sha256(b'{}').hexdigest()}))
            body={'seq':0,'prev':'0'*64,'kind':'__config__','payload':cfg}
            record=dict(body,hash=hashlib.sha256(c.acceptance.canonical(body)).hexdigest());raw=c.acceptance.canonical(record)+b'\n'
            head={'count':1,'last_hash':record['hash'],'byte_length':len(raw),'config_hash':hashlib.sha256(c.acceptance.canonical(cfg)).hexdigest()}
            headpath=run/'journal/HEAD.json';headpath.write_text(json.dumps(head));(run/'journal/events.jsonl').write_bytes(raw)
            original=Path.open
            def swapping(path,*args,**kwargs):
                handle=original(path,*args,**kwargs)
                if path==headpath and args and args[0]=='rb':
                    nxt=path.with_name('HEAD.next')
                    with original(nxt,'w') as f:json.dump({'synthetic_later_head':True},f)
                    nxt.replace(path)
                return handle
            with mock.patch.object(Path,'open',swapping):
                _,records=c.read_committed_prefix(run,base/'snapshot',{})
            self.assertEqual(len(records),1)
            self.assertEqual(json.loads((base/'snapshot/HEAD.json').read_text()),head)

if __name__=='__main__':unittest.main()
