"""CPU correctness tests. Synthetic cases are not training or method evidence."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from select_action import select, index
from evaluator_v16 import legacy, evaluate, state_sequence, query_context, compose

class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.compiler=legacy.Compiler(roles={'anchor_A':['chair','dining room'],'terminal':['bed','bedroom']},
            eligible={'anchor_A':[1],'terminal':[2]},tasks={'task_A':dict(anchor='anchor_A',terminal='terminal',instruction='test')})
    def trace(self):
        pixels=[{'1':300},{'1':300},{'2':300},{'2':300}]
        return dict(actions=['F','L','R','S'],collisions=0,complete=True,
                    observations=[dict(step=t,pixels=p,evidence_complete=True) for t,p in enumerate(pixels)])
    def test_prospective_collision_and_legacy_unknown(self):
        t=self.trace();self.assertEqual(evaluate(self.compiler,t,'task_A',1)['safe_v16_label'],'PASS')
        t['collisions']=1;r=evaluate(self.compiler,t,'task_A',1)
        self.assertEqual((r['safe_v16_label'],r['legacy_v15_label']),('FAIL','UNKNOWN'))
    def test_stop_cannot_create_observation(self):
        t=self.trace();t['observations'].append(dict(t['observations'][-1],step=4))
        self.assertEqual(evaluate(self.compiler,t,'task_A',1)['safe_v16_label'],'UNKNOWN')
    def test_missing_evidence_not_fail(self):
        t=self.trace();t['observations'][2]['evidence_complete']=False
        self.assertEqual(evaluate(self.compiler,t,'task_A',1)['safe_v16_label'],'UNKNOWN')
    def test_see2_instance_and_threshold(self):
        t=self.trace();t['observations'][1]['pixels']={'3':300}
        self.assertEqual(evaluate(self.compiler,t,'task_A',1)['safe_v16_label'],'FAIL')
        t=self.trace();t['observations'][1]['pixels']={'1':255}
        self.assertEqual(evaluate(self.compiler,t,'task_A',1)['safe_v16_label'],'FAIL')
    def test_query_state_reconstructs_every_cut(self):
        t=self.trace();states=state_sequence(self.compiler,t['observations'],'task_A')
        for i,z in enumerate(states):self.assertTrue(compose(z,query_context(self.compiler,t,'task_A',i)))
        from evaluator_v16 import query_contexts
        for task in ('task_A','task_T'):
            self.assertEqual(query_contexts(self.compiler,t,task),[query_context(self.compiler,t,task,i) for i in range(len(t['actions']))])
    def test_no_stop_fails(self):
        t=self.trace();t['actions'].pop()
        self.assertEqual(evaluate(self.compiler,t,'task_A',1)['safe_v16_label'],'FAIL')
    def test_method_argmax_can_continue_after_native_stop(self):
        result=select([0,0,0,2],[2,0,0,0])
        self.assertEqual(result['executed_action'],'move_forward');self.assertTrue(result['native_stop_method_continue'])
    def test_tie_and_nonfinite(self):
        self.assertEqual(index([1,1,1,1]),0)
        with self.assertRaises(ValueError):index([0,0,float('nan'),0])
    def test_process_cleanup_rejects_wrong_owner(self):
        import pipeline
        class Fake:
            pid=os.getpid()
            def poll(self):return None
        with self.assertRaisesRegex(RuntimeError,'REFUSE_NONOWNED'):
            pipeline.cleanup(Fake(),dict(pid=os.getpid(),uid=-1,pgid=-1,starttime='0'))
    def test_full500_and_stop_counts(self):
        trace=self.trace();trace['actions']=['L']*499+['S']
        trace['observations']=[dict(step=t,pixels={'1':300} if t<2 else {'2':300},evidence_complete=True) for t in range(500)]
        self.assertEqual(evaluate(self.compiler,trace,'task_A',100)['safe_v16_label'],'PASS')
        trace['actions'].insert(0,'L');trace['observations'].append(dict(step=500,pixels={'2':300},evidence_complete=True))
        self.assertEqual(evaluate(self.compiler,trace,'task_A',100)['safe_v16_label'],'FAIL')
    def test_prefix_audit_accepts_float_drift_but_rejects_flip(self):
        from evaluate_continuations import prefix_audit
        def row(logits,executed='move_forward'):
            return dict(raw=dict(key='same'),processed=dict(sha='same'),logits=logits,native_action='move_forward',executed_action=executed,override=False,decision=0)
        left=row([2.,1.,0.,-1.]);right=row([2.0001,1.,0.,-1.])
        result=prefix_audit([left],[right]);self.assertTrue(result['input_prefix_matched']);self.assertFalse(result['logits_bitwise_equal'])
        right['native_action']='turn_left'
        self.assertIn('first_divergence',prefix_audit([left],[right]))
        right=row([2.,1.,0.,-1.],'turn_left')
        self.assertEqual(prefix_audit([left],[right])['first_method_action_difference'],0)
    def test_planned_unknown_denominator(self):
        from review import rates,difference
        a=rates([dict(safe_v16_label='PASS'),dict(safe_v16_label='NOT_RUN')]);b=rates([dict(safe_v16_label='FAIL'),dict(safe_v16_label='UNKNOWN')])
        self.assertEqual(a['N'],2);self.assertEqual(a['identification_bounds'],[.5,1.])
        self.assertEqual(difference(a,b)['identification_bounds'],[0.,1.])
    def test_half_group_never_admitted(self):
        from evaluate_continuations import admitted
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            run=Path(tmp);session=run/'evaluate/session_001';session.mkdir(parents=True)
            write(session/'GROUP_000.json',dict(condition=0,ranks=[0],files={}),True)
            reg=dict(slots=[dict(rank=i,condition=0) for i in range(9)])
            self.assertEqual(admitted(run,reg),{})
            write(session/'STATE_SEAL_000.json',dict(base_unchanged=True,heads_unchanged=True,completed_ranks=[0]),True)
            with self.assertRaises(ValueError):admitted(run,reg)
    def test_manifest_house_isolation(self):
        manifest=read(HERE/'DATA_MANIFEST.json')
        houses=[r['house'] for r in manifest['houses']]
        self.assertEqual(len(houses),len(set(houses)));self.assertEqual(len(houses),9)
        old=read(V15/'NEW_HOUSE_PROTOCOL.json')
        exposed=set(old['excluded_exposed_houses'])|{h['house_id'] for h in old['houses']}
        self.assertFalse(set(houses)&exposed)

class GradientResumeTests(unittest.TestCase):
    def test_arm_auxiliary_gradients_and_shared_parameters(self):
        import torch
        import objective as o
        torch.set_num_threads(2);n=12
        family=dict(family_id='SYNTHETIC_CONTRACT',split='FIT',prefixes=[],cells=[],teacher_admission=dict(selected_cells={}),
            sequences=[dict(features=list(range(n)),targets=[0,1,2,3]*3,action_masks=[1]*n,
                state_targets=[[t>1,t>=1,t%2,t>1 and t%2] for t in range(n)],state_masks=[1]*n,
                query_contexts=[dict(zip(o.FIELDS,[t==n-1,1,t<2,t%2])) for t in range(n)],
                y=[0,1]*6,query_masks=[1]*n,cutoff=4)])
        for suffix in ('','_R'):
            for task in ('task_A','task_B','task_T'):
                for h in ('H_A','H_B'):family['prefixes'].append(dict(history_id=h+suffix,task_id=task))
        weights=dict(state=[[1.,1.]]*4,query=[1.,1.]);b=o.batch(family,'cpu')
        cache=dict(features=torch.randn(n,2048),logits=torch.randn(n,4));identities=[]
        for arm in ('B1','B2','Ours'):
            net=o.initialize(1209);identities.append(c.model_identity(net)['sha256'])
            loss,stats,parts=o.losses(net,cache,b,arm,weights,retain_steps=(0,));loss.backward()
            # The saved initialization has a zero residual actor, so B1's first
            # update learns the actor before BC can backpropagate into the writer.
            if arm!='B1':self.assertGreater(float(parts['writes'][0].grad.abs().sum()),0)
            self.assertEqual(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.state_head.parameters()),arm=='B2')
            self.assertEqual(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.result_head.parameters()),arm=='Ours')
            opt=torch.optim.AdamW(net.parameters(),lr=.001);old=c.model_identity(net)['sha256'];opt.step()
            self.assertNotEqual(old,c.model_identity(net)['sha256'])
            if arm=='B1':
                opt.zero_grad(set_to_none=True);loss,_,parts=o.losses(net,cache,b,arm,weights,retain_steps=(0,));loss.backward()
                self.assertGreater(float(parts['writes'][0].grad.abs().sum()),0)
        self.assertEqual(len(set(identities)),1)
    def test_old_write_receives_gradient_and_query_isolated(self):
        import torch
        import objective as o
        torch.set_num_threads(2);net=o.initialize(1209)
        x=torch.randn(2,24,2048);alive=torch.ones(2,24,dtype=torch.bool)
        states,writes=o.encode_masked(net,x,alive,retain_steps=(0,))
        before=states.detach().clone();q=torch.zeros(2,4)
        loss=net.reader(states[:,-1],q).square().mean()
        loss.backward()
        self.assertGreater(float(writes[0].grad.abs().sum()),0)
        self.assertTrue(torch.equal(before,states.detach()))
        self.assertTrue(torch.equal(net.reset(2,'cpu'),torch.zeros(2,8,64)))
    def test_padded_tail_does_not_update_memory(self):
        import torch
        import objective as o
        net=o.initialize(1209);x=torch.randn(2,12,2048);alive=torch.ones(2,12,dtype=torch.bool);alive[0,7:]=False
        a,_=o.encode_masked(net,x,alive);x[0,7:]+=1000;b,_=o.encode_masked(net,x,alive)
        self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[0,6],a[0,-1]))
    def test_optimizer_rng_resume_exact(self):
        import torch
        import train
        import random
        import numpy as np
        torch.manual_seed(9);net=torch.nn.Linear(3,2);opt=torch.optim.AdamW(net.parameters(),lr=.001)
        schedule=[1,2];binding=dict(test='CPU_ONLY')
        def update():
            opt.zero_grad();loss=net(torch.randn(4,3)).square().sum();loss.backward();opt.step()
        update()
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp)/'CHECKPOINT_0001.pt';train.save_checkpoint(path,net,opt,1,schedule,binding)
            update();expected=copy.deepcopy(net.state_dict());expected_random=(random.random(),float(np.random.rand()))
            step=train.restore_checkpoint(path,net,opt,schedule,binding);self.assertEqual(step,1)
            update();actual=(random.random(),float(np.random.rand()))
            self.assertEqual(expected_random,actual)
            self.assertTrue(all(torch.equal(expected[k],v) for k,v in net.state_dict().items()))
            with self.assertRaises(ValueError):train.restore_checkpoint(path,net,opt,schedule,dict(test='changed'))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]);result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        scope='Synthetic CPU correctness, real frozen architecture gradient and optimizer/RNG readback; no new GPU training or method-effect evidence'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
