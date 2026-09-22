"""CPU tests of continuation state, registered scheduling and single-head denominators."""
import sys,unittest,tempfile,random
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
from shared import *
prepare=local_module('prepare');ev=local_module('evaluate_continuations');trainer=local_module('train')
import torch
import numpy as np

class Tests(unittest.TestCase):
    def test_full_schedule_prefix_and_coverage(self):
        cfg=read(D/'PROTOCOL.json');r=Path(cfg['source_run']);data=read(r/'TRAIN_DATA.json');old=read(r/'SCHEDULES.json')['EXPANDED']['1209']
        result=prepare.extend(data,old,100000,1209)
        self.assertEqual(result[:1200],old);self.assertEqual(len(result),100000)
        self.assertEqual({x['family'] for x in result},set(data['arms']['EXPANDED']))
        self.assertEqual(result[1200]['ordinary'],old[0]['ordinary'])
        self.assertEqual(result[99999]['ordinary'],old[99999%1200]['ordinary'])
    def test_single_head_full_registry(self):
        cfg=read(D/'PROTOCOL.json');source=Path(cfg['source_run']);reg=ev.registry_value(read(source/'DATA.json')['raw_families'],cfg)
        self.assertEqual(len(reg['slots']),256);self.assertEqual(reg['models'],['EXPANDED_1209']);self.assertEqual(reg['expected_prefix_comparisons'],0)
        self.assertEqual(sum(c['available'] for c in reg['conditions']),248)
        self.assertEqual(reg['main_denominator_per_arm'],128)
        self.assertEqual([c['house'] for c in reg['conditions']], [c['house'] for c in read(source/'EVALUATION_REGISTRY.json')['conditions']])
    def test_checkpoint_plan(self):
        cfg=read(D/'PROTOCOL.json');self.assertEqual(cfg['checkpoint_steps'],list(range(5000,100001,5000)))
        self.assertEqual(cfg['new_training_updates'],98800);self.assertEqual(cfg['arms'],['EXPANDED']);self.assertEqual(cfg['seeds'],[1209])
    def test_optimizer_and_rng_restore(self):
        torch.set_num_threads(2);torch.manual_seed(1209);random.seed(1209);np.random.seed(1209)
        net=torch.nn.Linear(3,2);opt=torch.optim.AdamW(net.parameters(),lr=.001)
        def update():
            opt.zero_grad();x=torch.randn(2,3)*(random.random()+float(np.random.rand()));loss=net(x).square().mean();loss.backward();opt.step()
        update();binding={'fixture':'CPU serialization only'}
        with tempfile.TemporaryDirectory(dir=D) as temp:
            p=Path(temp)/'STEP.pt';trainer.save(p,net,opt,1200,binding);update();expected={n:v.clone() for n,v in net.state_dict().items()}
            self.assertEqual(trainer.load_resume(p,net,opt,binding),1200);update()
            self.assertTrue(all(torch.equal(v,expected[n]) for n,v in net.state_dict().items()))
            with self.assertRaises(ValueError):trainer.load_resume(p,net,opt,{'wrong':True})
    def test_actual_source_checkpoint(self):
        cfg=read(D/'PROTOCOL.json');p=Path(cfg['source_resume']);self.assertEqual(sha(p),cfg['source_resume_sha256'])
        record=torch.load(p,map_location='cpu',weights_only=False);meta=read(p.with_suffix('.json'))
        self.assertEqual(record['step'],1200);self.assertEqual(record['cursor'],1200);self.assertEqual(record['binding'],meta['binding'])
        self.assertEqual(record['binding']['arm'],'EXPANDED');self.assertEqual(record['binding']['seed'],1209)
        self.assertTrue(record['optimizer']['state']);self.assertTrue(record['rng']['cuda'])
    def test_no_eval_in_training(self):
        s=(D/'train.py').read_text();self.assertNotIn('evaluate_continuations',s);self.assertNotIn('EVALUATION_REGISTRY',s);self.assertNotIn('DATA.json\')',s.replace('TRAIN_DATA.json','').replace('natural_transfer_v9/DATA.json',''))
    def test_local_import_identity(self):
        self.assertEqual(Path(ev.__file__).resolve(),D/'evaluate_continuations.py');self.assertEqual(Path(trainer.__file__).resolve(),D/'train.py')
        self.assertEqual(Path(trainer.objective.__file__).resolve(),D.parent/'objective.py')
        self.assertEqual(trainer.EvidencePolicy.__module__,'model')

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(D/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),cuda_initialized=torch.cuda.is_initialized(),scope='CPU serialization/schedule/registration. No new training or navigation result.'),True)
    raise SystemExit(not result.wasSuccessful())
