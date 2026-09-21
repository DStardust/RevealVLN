"""Real cached-feature contracts, with synthetic cases explicitly labelled."""
import copy
import argparse
import io
import json
import random
import tempfile
import time
import unittest
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from model import EvidencePolicy, MODES, State
from runtime import RuntimePolicy
from prepare import event_labels
from train import save, restore
import objective as o

RUN = HERE/'runs/cpu_001'
EVIDENCE = {}


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        cls.data = read(RUN/'DATA.json')
        cls.cache = torch.load(SOURCE/'features/FEATURES.pt',map_location='cpu',weights_only=True)
        cls.fit = [f for f in cls.data['families'] if f['split']=='FIT']
        cls.family = copy.deepcopy(cls.fit[0])
        rows = [r for r in cls.family['sequences'] if any(r['action_masks']) and r['task']=='task_A']
        cls.family['sequences'] = [max(rows,key=lambda r:len(r['features']))]
        cls.b = o.batch(cls.family,'cpu'); cls.w=o.weights(cls.fit)

    def inputs(self):
        return self.cache['features'][self.b['indices']], self.cache['logits'][self.b['indices']], self.b['alive']

    def test_initialization_and_native_start(self):
        nets = [EvidencePolicy(mode=m) for m in MODES]
        self.assertEqual(len({c.model_identity(n)['sha256'] for n in nets}),1)
        x,y,alive=self.inputs()
        with torch.no_grad():
            for net in nets:
                self.assertTrue(torch.equal(net(x,y,alive)['logits'],y))
                self.assertFalse(any('result_head' in name for name,_ in net.named_parameters()))
        EVIDENCE['initial_parameter_count']=sum(p.numel() for p in nets[0].parameters())

    def test_future_prefix_and_label_firewall(self):
        net=EvidencePolicy();x,y,alive=self.inputs();change=x.clone();cut=12;change[:,cut+1:]*=-7
        with torch.no_grad():
            a=net(x,y,alive);b=net(change,y,alive)
            self.assertTrue(torch.equal(a['state'][:,:cut+1],b['state'][:,:cut+1]))
            altered=copy.deepcopy(self.b);altered['query']+=123;altered['y']=1-altered['y']
            before=o.losses(net,self.cache,self.b,self.w)[2]['logits']
            after=o.losses(net,self.cache,altered,self.w)[2]['logits']
            self.assertTrue(torch.equal(before,after))
        with self.assertRaises(TypeError):
            net.step(x[:,0],y[:,0],net.reset(1,'cpu'),query=torch.ones(4))

    def test_online_batch_padding_and_reset(self):
        x,y,alive=self.inputs()
        for mode in MODES:
            net=EvidencePolicy(mode=mode);state=net.reset(1,'cpu');values=[]
            with torch.no_grad():
                for t in range(x.shape[1]):
                    _,state,d=net.step(x[:,t],y[:,t],state);values.append(d['state'])
                full=net(x,y,alive)
                self.assertTrue(torch.equal(full['state'],torch.stack(values,1)))
                padded=net(torch.cat([x,x[:,:3]],1),torch.cat([y,y[:,:3]],1),torch.cat([alive,torch.zeros(1,3,dtype=torch.bool)],1))
                self.assertTrue(torch.equal(state.memory,padded['final_state'].memory))
                self.assertTrue(torch.equal(state.belief,padded['final_state'].belief))
                first=net.reset(1,'cpu');self.assertTrue(first.first.all());self.assertEqual(float(first.belief.sum()),0)

    def test_unknown_is_masked(self):
        class Compiler:
            def atoms(self,unused):return [dict(anchor=None,terminal=False),dict(anchor=True,terminal=None)]
        labels,masks=event_labels(Compiler(),[],'task_A')
        self.assertEqual(labels,[[0,0],[1,0]]);self.assertEqual(masks,[[0,1],[1,0]])
        _,masks=event_labels(Compiler(),[],'task_T')
        self.assertEqual(masks,[[0,1],[0,0]])
        with self.assertRaises(ValueError):o.weights([self.data['families'][-1]]) if self.data['families'][-1]['split']=='DEV' else o.weights([dict(self.fit[0],split='DEV')])

    def test_revise_can_retract_but_no_same_step_order_shortcut(self):
        x,y,_=self.inputs();net=EvidencePolicy(mode='REVISE')
        with torch.no_grad():
            for p in net.events.parameters():p.zero_()
            net.events[-1].bias.copy_(torch.tensor([-30.,30.]))
            for p in net.revision.parameters():p.zero_()
            net.revision[-1].bias.fill_(30.)
            state=State(net.core.reset(1,'cpu'),torch.ones(1),torch.zeros(1,dtype=torch.bool))
            _,updated,d=net.step(x[:,0],y[:,0],state)
            self.assertLess(float(updated.belief[0]),1e-6)
            net.mode='MONOTONIC';_,updated,_=net.step(x[:,0],y[:,0],state)
            self.assertEqual(float(updated.belief[0]),1.)
            net.events[-1].bias.fill_(30.)
            state=State(state.memory,torch.zeros(1),state.first)
            _,_,d=net.step(x[:,0],y[:,0],state)
            self.assertGreater(float(d['state'][0,1]),.99)
            self.assertEqual(float(d['state'][0,3]),0.)
        EVIDENCE['retraction_test']='Synthetic gate wiring only, not learned correction accuracy.'

    def test_real_backward_update_and_long_gradient(self):
        records=[]
        for mode in MODES:
            net=EvidencePolicy(mode=mode);opt=torch.optim.AdamW(net.parameters(),lr=.001)
            initial=c.model_identity(net)['sha256']
            for step in range(3):
                opt.zero_grad();loss,_,detail=o.losses(net,self.cache,self.b,self.w);loss.backward()
                self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None));opt.step()
            x,y,alive=self.inputs();x=x.clone().requires_grad_();result=net(x,y,alive)
            # Final action only: excludes local event/state auxiliary losses.
            action=torch.nn.functional.cross_entropy(result['logits'][:,-1],self.b['targets'][:,-1])
            grad=torch.autograd.grad(action,x)[0];early=float(grad[:,0].norm())
            self.assertGreater(early,0);self.assertNotEqual(initial,c.model_identity(net)['sha256'])
            influence=float(torch.linalg.vector_norm(net.state_action(result['state'][:,-1])).detach())
            self.assertGreater(influence,0)
            records.append(dict(mode=mode,cpu_updates=3,sequence_length=x.shape[1],early_step=0,loss_step=x.shape[1]-1,
                                early_action_gradient_norm=early,state_action_contribution_norm=influence,
                                active_gradient_parameters=sum(p.numel() for p in net.parameters() if p.grad is not None)))
        EVIDENCE['real_cached_feature_gradients']=records

    def test_optimizer_rng_resume_exact(self):
        net=EvidencePolicy();opt=torch.optim.AdamW(net.parameters(),lr=.001)
        binding=dict(test='real FIT optimizer RNG resume')
        def update():
            opt.zero_grad();loss,_,_=o.losses(net,self.cache,self.b,self.w);loss.backward();opt.step()
        update()
        with tempfile.TemporaryDirectory(dir=RUN) as directory:
            path=Path(directory)/'STEP.pt';save(path,net,opt,1,binding)
            expected=(random.random(),float(np.random.rand()),float(torch.rand(())))
            update();final=c.model_identity(net)['sha256']
            self.assertEqual(restore(path,net,opt,binding),1)
            self.assertEqual(expected,(random.random(),float(np.random.rand()),float(torch.rand(()))))
            update();self.assertEqual(final,c.model_identity(net)['sha256'])
            with self.assertRaises(ValueError):restore(path,net,opt,dict(other='changed'))

    def test_runtime_budget_history_stop(self):
        # Tiny deterministic actor tests runtime control, not model capability.
        class Actor(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(()))
            def reset(self,batch,device):return 0
            def step(self,x,y,state):return y,state+1,dict(state=torch.zeros(1,4))
        runtime=RuntimePolicy(Actor());x=torch.zeros(1,2048);y=torch.tensor([[3.,0.,0.,-1.]])
        for t in range(499):
            runtime.propose(x,y);runtime.commit(0)
        self.assertEqual(runtime.history,['F']*8)
        runtime.propose(x,torch.tensor([[0.,0.,0.,3.]]));runtime.commit(3)
        self.assertEqual(runtime.decisions,500);self.assertTrue(runtime.stopped)
        with self.assertRaises(ValueError):runtime.propose(x,y)
        runtime.reset();runtime.propose(x,y)
        with self.assertRaises(ValueError):runtime.propose(x,y)
        with self.assertRaises(ValueError):runtime.commit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN)
    RUN=parser.parse_args().run.resolve()
    started=time.monotonic();stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue())
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),seconds=time.monotonic()-started,device='cpu',cuda_initialized=torch.cuda.is_initialized(),
                new_base_forwards=0,navigation_episodes=0,evidence=EVIDENCE,output=stream.getvalue())
    write(RUN/'CPU_TEST_RESULT.json',record,True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
