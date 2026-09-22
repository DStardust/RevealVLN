"""CPU operator and observed-input checks; no new trajectory or optimizer update."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
import torch
positive=load('positive_stop_test_model',HERE/'model.py')
SOURCE=PARENT/'state_stop_readout_v1/runs/repair_001'

class StopTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(1209)
    def test_negative_residual_cannot_veto_stop(self):
        model=positive.StopReadout(1209)
        with torch.no_grad():model.net[-1].bias.fill_(-50)
        x=torch.randn(32,4);z=torch.rand(32,4)
        self.assertTrue(torch.equal(model(x,z),x))
    def test_positive_residual_preserves_motion_and_existing_stop(self):
        model=positive.StopReadout(1209)
        with torch.no_grad():model.net[-1].bias.fill_(.7)
        x=torch.randn(200,4);z=torch.rand(200,4);y=model(x,z)
        self.assertTrue(torch.equal(x[:,:3],y[:,:3]));self.assertTrue(torch.all(y[:,3]>=x[:,3]))
        self.assertTrue(torch.all(y[x.argmax(-1)==3].argmax(-1)==3))
    def test_all_trained_weights_unchanged(self):
        for seed in (1209,1210,1211):
            folder=SOURCE/'train'/f'STOPFIX_{seed}';record=read(folder/'RESULT.json');net=make_head(f'STOPPLUS_{seed}')
            self.assertEqual(sha(folder/'FINAL.pt'),record['checkpoint_sha256'])
            net.load_state_dict(torch.load(folder/'FINAL.pt',map_location='cpu',weights_only=True),strict=True)
            self.assertEqual(c.model_identity(net)['sha256'],record['final'])
    def test_memory_update_unchanged(self):
        net=make_head('STOPPLUS_1209').eval();f=torch.randn(2,2048);native=torch.randn(2,4);state=net.reset(2,'cpu')
        with torch.inference_mode():
            _,old,_=net.frozen.step(f,native,state);_,new,_=net.step(f,native,state)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(old,new)))
    def test_previous_five_veto_inputs_protected(self):
        source=LINE/'reviews/Q35N_STOP_READOUT_REVIEW_20260922/FIRST_DIVERGENCES.json';cases=read(source)['cases'];protected=0;wins=0
        for row in cases:
            m=positive.StopReadout(row['seed']);state=torch.load(SOURCE/'train'/f"STOPFIX_{row['seed']}"/'FINAL.pt',map_location='cpu',weights_only=True)
            m.load_state_dict({k.removeprefix('readout.'):v for k,v in state.items() if k.startswith('readout.')})
            with torch.inference_mode():x=torch.tensor([row['original_method_logits']]);y=m(x,torch.tensor([row['predicted_state']]))
            self.assertEqual(y.argmax(-1).item(),3)
            if row['outcome']=='losses':protected+=1;self.assertEqual(x.argmax(-1).item(),3)
            else:wins+=1
        self.assertEqual((protected,wins),(5,1))
    def test_original_dev_and_new_model_tags(self):
        module=load('positive_stop_test_runtime',HERE/'evaluate_continuations.py')
        registry=module.registry_value(read(SOURCE/'DATA.json')['raw_families'],dict(seeds=[1209,1210,1211],arms=['MONOTONIC','STOPPLUS']))
        self.assertEqual(registry['conditions'],read(SOURCE/'EVALUATION_REGISTRY.json')['conditions'])
        self.assertEqual(len(registry['slots']),768);self.assertEqual(registry['main_denominator_per_arm'],192)
        self.assertTrue(all(x.startswith(('MONOTONIC_','STOPPLUS_')) for x in registry['models']))
    def test_raw_qwen_stop_does_not_override_method(self):
        from select_action import select
        self.assertEqual(select([0,0,0,5],[0,3,0,1])['executed_action'],c.ACTIONS[1])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StopTests))
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),new_updates=0,
        scope='CPU invariant checks plus six previously observed first-divergence inputs. No new navigation successes inferred.'))
    raise SystemExit(not result.wasSuccessful())
