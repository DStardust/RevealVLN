import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import torch
from model import ExecutionAdaptation
from runtime import FeatureMemoryState, ChunkActionProcessor
from scope_processor import ScopedProcessor
from review import summarize


class ScopeTests(unittest.TestCase):
    def make(self, scope, enabled=True):
        torch.manual_seed(42)
        head=ExecutionAdaptation(16,'DELTA').eval()
        with torch.no_grad(): head.actor[-1].bias.copy_(torch.tensor([-5.,5.,1.,-1.]))
        state=FeatureMemoryState(head);state.observe(torch.randn(16),None);state.begin_query()
        feature=torch.randn(1,16)
        processor=ScopedProcessor(state,[8,9],[0,1,2,3],lambda:feature,scope=scope,apply_residual=enabled)
        processor(torch.tensor([[10]]),torch.zeros(1,12))
        return processor

    def test_all_is_original_exact_and_first_differs_only_later(self):
        all_tokens=self.make('ALL');first=self.make('FIRST')
        original=ChunkActionProcessor(all_tokens.state,[8,9],[0,1,2,3],all_tokens.actor_feature_provider)
        original(torch.tensor([[10]]),torch.zeros(1,12))
        ids=torch.tensor([[10,8,9]]);scores=torch.full((1,12),-100.);scores[0,:4]=torch.tensor([4.,1.,0.,-1.])
        memory=first.state.memory.clone()
        for offset in range(4):
            all_scores=all_tokens(ids,scores);old_scores=original(ids,scores);first_scores=first(ids,scores)
            self.assertTrue(torch.equal(all_scores,old_scores))
            self.assertTrue(torch.equal(first_scores,all_scores if offset==0 else scores))
            self.assertEqual(first.records[-1]['residual_applied'],offset==0)
            self.assertTrue(torch.equal(first_scores[:,4:],scores[:,4:]))
            ids=torch.cat([ids,torch.tensor([[1]])],-1)
        self.assertTrue(torch.equal(memory,first.state.memory));self.assertEqual(first.state.writes,1)
        scores[0,11]=200.
        self.assertTrue(torch.equal(first(ids,scores),scores))
        self.assertEqual(first.records[-1]['method_token'],11)

    def test_native_and_stop_definition_unchanged(self):
        p=self.make('FIRST',enabled=False);scores=torch.randn(1,12)
        self.assertTrue(torch.equal(p(torch.tensor([[10,8,9]]),scores),scores))
        p=self.make('FIRST');scores=torch.full((1,12),-100.);scores[0,:4]=torch.tensor([4.,1.,0.,-1.])
        p(torch.tensor([[10,8,9]]),scores)
        self.assertEqual(p.records[-1]['native_action'],0)
        self.assertEqual(p.records[-1]['method_action'],1)
        p.state.end_query();p.state.mark_stopped()
        with self.assertRaisesRegex(ValueError,'STOP_MUST_NOT_CREATE_OBSERVATION'):
            p.state.observe(torch.randn(16),0)

    def test_new_real_query_reenables_first_offset(self):
        p=self.make('FIRST');p.state.end_query();p.state.observe(torch.randn(16),2);p.state.begin_query()
        q=ScopedProcessor(p.state,[8,9],[0,1,2,3],p.actor_feature_provider,scope='FIRST')
        q(torch.tensor([[10]]),torch.zeros(1,12))
        scores=torch.randn(1,12);q(torch.tensor([[10,8,9]]),scores)
        self.assertTrue(q.records[0]['residual_applied']);self.assertEqual(p.state.writes,2)

    def test_no_full_sr_for_missing_groups(self):
        p={'heads':{'ALL':{},'FIRST':{}},'comparisons':{'scope':{'CURRENT':'ALL','DELTA':'FIRST'}}}
        entries=[{'id':1,'house':'a'},{'id':2,'house':'b'}]
        row=dict(house='a',outcomes={a:dict(success=int(a=='FIRST'),spl=0.,steps=10) for a in ['NATIVE','ALL','FIRST']})
        r=summarize(p,entries,{1:row})
        self.assertIsNone(r['arms']['FIRST']['sr']);self.assertEqual(r['arms']['FIRST']['sr_identification_bounds'],[.5,1.])
        self.assertIsNone(r['paired']['scope']['delta_sr'])
        self.assertEqual(r['paired']['scope']['wins'],[1])
        row2=dict(house='b',outcomes={a:dict(success=int(a=='ALL'),spl=0.,steps=10) for a in ['NATIVE','ALL','FIRST']})
        r=summarize(p,entries,{1:row,2:row2});self.assertEqual(r['paired']['scope']['delta_sr'],0)

    def test_unknown_scope_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNREGISTERED_INTERVENTION_SCOPE'):self.make('CHOSEN_BY_SCORE')


if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScopeTests))
    receipt=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        passed=result.wasSuccessful(),gpu_hours=0,optimizer_updates=0)
    with (HERE/'CPU_TEST_RESULT.json').open('x') as f:json.dump(receipt,f,indent=2)
    raise SystemExit(not result.wasSuccessful())
