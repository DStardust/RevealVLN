"""Behavioral CPU tests with the actual frozen policy and inherited losses."""
import inspect
from pathlib import Path
import sys
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import shared
from shared import *
import torch
model=load('stop_test_model',HERE/'model.py')
gate=load('stop_test_gate',HERE/'gate.py')
trainer=load('stop_test_train',HERE/'train.py')

class RepairTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(12)
    def test_zero_init_and_motion_invariance(self):
        net=model.StopReadout(1209);x=torch.randn(9,4);z=torch.rand(9,4)
        self.assertTrue(torch.equal(x,net(x,z)))
        before=c.model_identity(net)['sha256'];opt=torch.optim.AdamW(net.parameters(),lr=.001)
        loss=torch.nn.functional.cross_entropy(net(x,z),torch.full((9,),3));loss.backward();opt.step()
        self.assertNotEqual(before,c.model_identity(net)['sha256']);self.assertTrue(torch.equal(x[:,:3],net(x,z)[:,:3]))
        self.assertEqual(sum(p.numel() for p in net.parameters()),113)
    def test_frozen_state_and_gradient(self):
        net=make_head('STOPFIX_1209');before=c.model_identity(net.frozen)['sha256']
        x=torch.randn(2,2048);native=torch.randn(2,4);state=net.reset(2,'cpu')
        original,oldstate,detail=net.frozen.step(x,native,state);fixed,newstate,newdetail=net.step(x,native,state)
        self.assertTrue(torch.equal(original,fixed))
        for a,b in zip(oldstate,newstate):self.assertTrue(torch.equal(a,b))
        torch.nn.functional.cross_entropy(fixed,torch.tensor([3,3])).backward()
        self.assertTrue(all(p.grad is None for p in net.frozen.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in net.readout.parameters()),0)
        self.assertEqual(before,c.model_identity(net.frozen)['sha256'])
    def test_no_query_or_truth_input(self):
        self.assertEqual(list(inspect.signature(model.StopReadout.forward).parameters),['self','logits','state'])
        self.assertEqual(list(inspect.signature(model.StopPolicy.step).parameters),['self','feature','native_logits','state'])
    def test_objective_preserves_registered_losses(self):
        # One padded-looking trace: only actual teacher positions are action labels.
        row=dict(logits=torch.randn(4,4),state=torch.rand(4,4),native=torch.randn(4,4),targets=torch.tensor([0,1,2,3]),
            action_mask=torch.tensor([False,False,True,True]),cutoff=2,preservation=torch.tensor([True,True,True,False]))
        natural=dict(logits=torch.randn(3,4),state=torch.rand(3,4),targets=torch.tensor([0,1,3]))
        head=model.StopReadout(1209);total,stats=trainer.objective(head,dict(rows=[row]),[natural]);F=torch.nn.functional
        expected=F.cross_entropy(row['logits'][2:],row['targets'][2:])+F.cross_entropy(row['logits'][2:3],row['targets'][2:3])+F.kl_div(row['logits'][:3].log_softmax(-1),row['native'][:3].softmax(-1),reduction='batchmean')+F.cross_entropy(natural['logits'],natural['targets'])
        self.assertTrue(torch.allclose(total,expected));total.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
    def rows(self):
        out=[]
        for split in ('FIT','DEV','ORDINARY_CHECK'):
            old=dict(n=100,correct=90,missed_stop=5,correct_state_missed_stop=5,false_stop=2)
            out.append(dict(split=split,old=old,repair=dict(old,correct=91,missed_stop=4,correct_state_missed_stop=4)))
        return out
    def test_gate_rejects_false_stop_and_ordinary_regression(self):
        values=[self.rows() for _ in range(3)];self.assertTrue(gate.assess(values)['admit_closed_loop'])
        values[0][1]['repair']['false_stop']=3;self.assertFalse(gate.assess(values)['admit_closed_loop'])
        values=[self.rows() for _ in range(3)];values[0][2]['repair']['correct']=80
        self.assertFalse(gate.assess(values)['admit_closed_loop'])
    def test_registry_preserves_original_conditions(self):
        module=load('stop_test_runtime',HERE/'evaluate_continuations.py')
        reg=module.registry_value(read(CPU/'DATA.json')['raw_families'],dict(seeds=[1209,1210,1211],arms=['MONOTONIC','STOPFIX']))
        self.assertEqual(len(reg['slots']),768);self.assertEqual(reg['main_denominator_per_arm'],192)
        old=read(PARENT/'gpu_runtime_r1/runs/gpu_001/EVALUATION_REGISTRY.json')
        self.assertEqual([{k:v for k,v in x.items() if k!='available'} for x in reg['conditions']],old['conditions'])
    def test_action_selection_has_no_native_stop_override(self):
        from select_action import select
        value=select([0,0,0,2],[0,3,0,1]);self.assertEqual(value['executed_action'],c.ACTIONS[1])

if __name__=='__main__':
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(RepairTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-start,
        scope='CPU behavior/gradient contracts; no GPU training or navigation effectiveness'))
    raise SystemExit(not result.wasSuccessful())
