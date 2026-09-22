"""Real ORIGINAL event-only update, frozen-path identity, and restore correctness."""
from pathlib import Path
import copy
import tempfile
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from train_one import configure_trainable, frozen_state, save, restore, objective
from event_loss import Pool
from select_action import select


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg=read(HERE/'PROTOCOL.json');cls.data=read(CPU/'DATA.json')
        cls.families=[f for f in cls.data['families'] if f['split']=='FIT']
        cls.weights=objective.weights(cls.families)
        cls.cache={k:v.float() for k,v in torch.load(CONTROL/'features/FEATURES.pt',map_location='cpu',weights_only=True).items()}
        cls.records=[r for r in read(PARENT/'recovery_localization_v1/runs/diagnosis_001/EVENT_INDEX.json')['records'] if r['split']=='FIT']

    def net(self):
        net=make_head('REPAIR_1209')
        path=Path(self.cfg['source_original_run'])/'train/ORIGINAL_1209/FINAL.pt'
        net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        configure_trainable(net)
        return net

    def step(self,net,opt,step=0):
        opt.zero_grad(set_to_none=True)
        b=objective.batch(self.families[0],'cpu')
        loss,stats,detail=objective.losses(net,self.cache,b,self.weights)
        loss=loss-detail['event_loss']+Pool(self.records).sample_loss(net,self.cache,1209,step,256)
        loss.backward();opt.step()

    def test_actual_update_changes_only_events_and_preserves_memory_actor(self):
        net=self.net();before=copy.deepcopy(net.state_dict());frozen=frozen_state(net)
        feature=self.cache['features'][:3];native=self.cache['logits'][:3]
        def core_outputs():
            with torch.no_grad():
                m=net.core.reset(1,'cpu');saved=[]
                for x,n in zip(feature,native):
                    m,_=net.core.update(x[None],m)
                    saved.append((m.clone(),net.core.action_logits(m,n[None],x[None]).clone()))
                return saved
        original=core_outputs()
        opt=torch.optim.AdamW(net.events.parameters(),lr=.001,weight_decay=.01)
        self.step(net,opt)
        changed=[k for k,v in before.items() if not torch.equal(v,net.state_dict()[k])]
        self.assertTrue(changed);self.assertTrue(all(k.startswith('events.') for k in changed))
        self.assertEqual(frozen,frozen_state(net))
        self.assertTrue(all(p.grad is None for n,p in net.named_parameters() if not n.startswith('events.')))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.events.parameters()))
        self.assertTrue(all(torch.equal(a,c_) and torch.equal(b,d) for (a,b),(c_,d) in zip(original,core_outputs())))

    def test_resumed_event_update_equals_uninterrupted(self):
        net=self.net();opt=torch.optim.AdamW(net.events.parameters(),lr=.001,weight_decay=.01)
        self.step(net,opt,0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'STEP_0001.pt';binding=dict(scope='EVENT_ONLY_CPU_RESTORE')
            save(path,net,opt,1,binding)
            self.step(net,opt,1);expected=copy.deepcopy(net.state_dict())
            self.assertEqual(restore(path,net,opt,binding),1)
            self.step(net,opt,1)
            self.assertTrue(all(torch.equal(v,net.state_dict()[k]) for k,v in expected.items()))
            with self.assertRaises(ValueError):restore(path,net,opt,dict(scope='WRONG_SCOPE'))

    def test_fixed_optimizer_scope(self):
        net=self.net()
        names=[n for n,p in net.named_parameters() if p.requires_grad]
        self.assertEqual(names,['events.0.weight','events.0.bias','events.2.weight','events.2.bias'])
        self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),262530)

    def test_pool_rejects_dev_and_preserves_label_prevalence(self):
        with self.assertRaises(ValueError):Pool([dict(self.records[0],split='DEV')])
        pool=Pool(self.records);mass=pool.mass()
        for ids in pool.cells.values():
            self.assertEqual(len({mass[i] for i in ids}),1)
        self.assertAlmostEqual(sum(mass),1.)

    def test_method_argmax_no_new_stop_guard(self):
        self.assertEqual(select([0.,0.,0.,10.],[10.,0.,0.,0.])['executed_action'],'move_forward')
        self.assertEqual(select([10.,0.,0.,0.],[0.,0.,0.,10.])['executed_action'],'STOP')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        success=result.wasSuccessful(),cuda_initialized=torch.cuda.is_initialized(),
        scope='Real CPU event updates and frozen path identity; not formal training or navigation benefit.'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
