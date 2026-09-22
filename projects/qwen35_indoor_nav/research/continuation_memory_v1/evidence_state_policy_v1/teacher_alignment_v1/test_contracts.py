"""CPU tests for actual model gradients and the new supervision/provenance boundaries."""
import copy
import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from prepare import first_ready
from build_data import apply_overlay
from collect import validate_prefix

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=read(CPU/'DATA.json')
    def case(self):
        f=next(f for f in self.data['families'] if f['split']=='FIT')
        i,r=next((i,r) for i,r in enumerate(f['sequences']) if r['task']=='task_A' and r['history']=='missing' and any(r['action_masks']))
        return f,i,r
    def test_cutoff_is_lower_bound(self):
        row=dict(cutoff=3,targets=[1]*6,state_masks=[1]*6,state_targets=[[0,0,0,x] for x in [0,1,0,0,1,1]])
        self.assertEqual(first_ready(row),4)
    def test_only_action_supervision_changes(self):
        f,i,r=self.case();t=first_ready(r);entry=dict(family_id=f['family_id'],sequence=i,stop_step=t,plan='cpu_fixture')
        new=apply_overlay(self.data,[entry]);nf=next(x for x in new['families'] if x['family_id']==f['family_id']);nr=nf['sequences'][i]
        self.assertEqual(nr['targets'][t],3);self.assertFalse(any(nr['action_masks'][t+1:]))
        for key in r:
            if key not in ('targets','action_masks'):self.assertEqual(nr[key],r[key])
        self.assertEqual([x for x in self.data['families'] if x['split']=='DEV'],[x for x in new['families'] if x['split']=='DEV'])
    def test_nonfit_overlay_rejected(self):
        f=next(f for f in self.data['families'] if f['split']=='DEV')
        with self.assertRaisesRegex(ValueError,'NONFIT'):apply_overlay(self.data,[dict(family_id=f['family_id'],sequence=0,stop_step=0,plan='fixture')])
    def test_fake_stop_observation_rejected(self):
        obs=dict(rgb_hash='a',semantic_hash='b',pose=dict(position=[0,0,0],rotation=[1,0,0,0]))
        ref=dict(actions=['L','R','S'],observations=[obs]*3)
        actual=dict(actions=['L','S'],observations=[obs]*2)
        validate_prefix(actual,ref,1)
        with self.assertRaisesRegex(ValueError,'STOP_MUST_NOT'):validate_prefix(dict(actual,observations=[obs]*3),ref,1)
        altered=copy.deepcopy(actual);altered['observations'][0]['rgb_hash']='changed'
        with self.assertRaisesRegex(ValueError,'PHYSICAL_PREFIX'):validate_prefix(altered,ref,1)
    def test_real_model_backward_and_update(self):
        import torch
        torch.set_num_threads(2)
        module=load('teacher_train_contract',HERE/'train_one.py')
        f,i,r=self.case();overlay=dict(family_id=f['family_id'],sequence=i,stop_step=first_ready(r),plan='fixture')
        altered=apply_overlay(dict(families=[f]),[overlay])['families'][0]
        a=module.EvidencePolicy(1209,'MONOTONIC');b=module.EvidencePolicy(1209,'MONOTONIC')
        self.assertEqual(c.model_identity(a)['sha256'],c.model_identity(b)['sha256'])
        w=module.objective.weights([x for x in self.data['families'] if x['split']=='FIT'])
        self.assertEqual(w,module.objective.weights([altered]+[x for x in self.data['families'] if x['split']=='FIT' and x['family_id']!=f['family_id']]))
        cache=torch.load(CONTROL/'features/FEATURES.pt',map_location='cpu',weights_only=True)
        cache={k:v.float() for k,v in cache.items()}
        ba=module.objective.batch(f,'cpu');bb=module.objective.batch(altered,'cpu')
        for key in ['alive','indices','state','state_mask','events','event_mask','preservation_mask']:self.assertTrue(torch.equal(ba[key],bb[key]))
        before=c.model_identity(b)['sha256'];opt=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.01)
        loss,stats,detail=module.objective.losses(b,cache,bb,w);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in b.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters()))
        opt.step();self.assertNotEqual(before,c.model_identity(b)['sha256'])
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    began=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        successful=result.wasSuccessful(),seconds=time.monotonic()-began,
        actual_cpu_backward=True,production_updates=0,physical_replay_certified=False,
        note='One discarded CPU optimizer update exercises the actual model/loss. Physical teacher alternatives require separate real simulator certificates.'))
    raise SystemExit(not result.wasSuccessful())
