"""Synthetic CPU fixtures only: these are not simulator trajectories."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('witness_bank_tests',HERE/'bank.py')
bank=importlib.util.module_from_spec(spec);spec.loader.exec_module(bank)

ROLE_ROWS=[dict(signature=['chair','living room','chair'],eligible_ids=[1]),
           dict(signature=['table','kitchen','table'],eligible_ids=[2]),
           dict(signature=['tv_monitor','living room','tv'],eligible_ids=[3]),
           dict(signature=['plant','hallway','plant'],eligible_ids=[4])]

def pose():
    simple=dict(position=[0.,0.,0.],rotation=[1.,0.,0.,0.])
    return dict(simple,sensors={key:copy.deepcopy(simple) for key in ('rgb','semantic')})

def record(name,actions,instance=None,closed=True,house='mock_house'):
    observations=[]
    for step in range(len(actions)+1):
        pixels={str(instance):300} if instance and step in (1,2) else {}
        observations.append(dict(step=step,pixels=pixels,evidence_complete=True,pose=pose(),
            rgb_hash='a'*64,semantic_hash='b'*64))
    return dict(id=house+':'+name,trace_ref='synthetic_cpu_fixture:'+name,house_id=house,
        scene_fingerprint='synthetic_scene_not_runtime',split='FIT',roles=copy.deepcopy(ROLE_ROWS),closed_loop=closed,
        trace=dict(actions=list(actions),observations=observations,complete=True,collisions=0))

def fixture(house='mock_house'):
    return [record('A','FLFR',1,house=house),record('B','FRFL',2,house=house),
            record('T','FF',3,closed=False,house=house),record('I','LFFR',4,house=house),
            record('tail','LRLRLRLR',house=house)]

class BankTests(unittest.TestCase):
    def chosen(self,result):
        return next(p for p in result['proposals'] if p['roles']['anchor_A']['mpcat40']=='chair'
                    and p['roles']['anchor_B']['mpcat40']=='table' and p['roles']['terminal']['mpcat40']=='tv_monitor')
    def test_component_program_not_certificate(self):
        result=bank.propose(fixture());p=self.chosen(result)
        self.assertTrue(p['all_components_stored'])
        self.assertTrue(p['irrelevant_has_at_least_2F'])
        self.assertEqual(p['continuations_unreplayed']['C0'],['F','F','S'])
        self.assertEqual(p['histories_unbalanced']['H_A_I'],list('FLFRLFFRLRLRLRLR'))
        self.assertFalse(p['physical_replay_certified'])
        self.assertTrue(p['same_action_count_control_required'])
    def test_deterministic_input_order(self):
        self.assertEqual(bank.propose(fixture()),bank.propose(list(reversed(fixture()))))
    def test_no_cross_house_components(self):
        first=fixture('one')[:2];second=fixture('two')[2:]
        self.assertFalse(bank.propose(first+second)['proposals'])
    def test_complete_houses_stay_separate(self):
        result=bank.propose(fixture('one')+fixture('two'))
        self.assertEqual(result['hub_groups'],2)
        for p in result['proposals']:
            self.assertTrue(all(c is None or c['trace_id'].startswith(p['house_id']+':') for c in p['components'].values()))
    def test_public_tail_anchor_event_blocks(self):
        records=fixture();records[-1]=record('tail','LRLRLRLR',1)
        result=bank.propose(records)
        self.assertFalse(any(p['roles']['anchor_A']['mpcat40']=='chair' or p['roles']['anchor_B']['mpcat40']=='chair' for p in result['proposals']))
    def test_same_raw_query_collision_blocked(self):
        self.assertFalse(bank.compat.semantic_distinct(['["chair","living room","chair"]','["chair","living room","armchair"]']))
    def test_prefer_moving_I(self):
        records=fixture();records.append(record('I_short','LR',4))
        p=self.chosen(bank.propose(records))
        self.assertEqual(p['components']['irrelevant_loop']['actions'],list('LFFR'))
    def test_minimum_movement_excludes_turn_only(self):
        records=fixture();records[3]=record('I_short','LR',4)
        result=bank.propose(records,min_irrelevant_forward=2)
        self.assertFalse(any(p['roles']['irrelevant']['mpcat40']=='plant' for p in result['proposals']))
    def test_false_closed_flag_cannot_skip_pose_check(self):
        records=fixture();records[0]['trace']['observations'][-1]['pose']['position'][0]=1
        with self.assertRaisesRegex(ValueError,'CLAIMED_LOOP'):bank.propose(records)
    def test_unknown_trace_not_negative(self):
        records=fixture();records[0]['trace']['complete']=False
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):bank.propose(records)
    def test_heldout_forbidden(self):
        records=fixture();records[0]['split']='INTERNAL_DEV'
        with self.assertRaisesRegex(ValueError,'ONLY_FIT'):bank.propose(records)
    def test_whole_house_catalog_must_match(self):
        records=fixture();records[1]['roles'][0]['eligible_ids']=[11]
        with self.assertRaisesRegex(ValueError,'INCONSISTENT'):bank.propose(records)
    def test_input_not_mutated(self):
        records=fixture();before=copy.deepcopy(records);bank.propose(records)
        self.assertEqual(records,before)
    def test_output_bound(self):
        result=bank.propose(fixture(),max_programs=1)
        self.assertEqual(len(result['proposals']),1)
    def test_different_initial_image_does_not_mix(self):
        records=fixture();records[1]['trace']['observations'][0]['rgb_hash']='c'*64
        result=bank.propose(records)
        self.assertFalse(any(p['roles']['anchor_A']['mpcat40']=='chair' and p['roles']['anchor_B']['mpcat40']=='table' for p in result['proposals']))
    def test_below_256_not_a_role_witness(self):
        records=fixture()
        for step in (1,2):records[0]['trace']['observations'][step]['pixels']={'1':255}
        self.assertFalse(bank.propose(records)['proposals'])
    def test_distinct_instances_do_not_confirm_each_other(self):
        records=fixture()
        for row in records:row['roles'][0]['eligible_ids']=[1,11]
        records[0]['trace']['observations'][2]['pixels']={'11':300}
        self.assertFalse(bank.propose(records)['proposals'])

if __name__=='__main__':unittest.main()
