"""Real-cache CPU tests for changed actor admission, ties and gradient flow."""
import copy
import os
from pathlib import Path
import sys
import time
import unittest

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import teacher
import train
c=train.c


class TeacherTest(unittest.TestCase):
    def test_all_real_admissions_preserve_source_and_cost(self):
        data=c.read(HERE.parent/'multifamily_v7/DATA.json')
        ledger=c.read(HERE/'ACTOR_ADMISSION.json')
        for family,recorded in zip(data['families'],ledger['family_rows']):
            original=copy.deepcopy(family)
            admitted=teacher.admit(family)
            self.assertEqual(admitted['masks'],recorded['masks'])
            for prefix,indices in admitted['selected_cells'].items():
                cost=min(len(x['tail']) for x in family['cells'] if x['prefix']==int(prefix)
                         and x['mask'] and x['y'] and x['tail'])
                for i in indices:
                    self.assertEqual(len(family['cells'][i]['tail']),cost)
                    self.assertEqual(family['cells'][i]['y'],1)
            for cell,mask in zip(family['cells'],admitted['masks']):
                self.assertTrue(all(m<=r['mask'] for m,r in zip(mask,cell['tail'])))
            self.assertEqual(teacher.branch_cases(family,admitted)['cases'],recorded['cases'])
            self.assertEqual(family,original)

    def test_equal_cost_conflicting_first_actions_abstain(self):
        family=dict(prefixes=[dict(features=[11])],cells=[
            dict(prefix=0,mask=1,y=1,tail=[dict(feature=11,target=0,mask=1),dict(feature=12,target=3,mask=1)]),
            dict(prefix=0,mask=1,y=1,tail=[dict(feature=11,target=1,mask=1),dict(feature=13,target=3,mask=1)])])
        value=teacher.admit(family)
        self.assertEqual(value['masks'],[[0,1],[0,1]])
        self.assertEqual(value['tied_conflicting_contexts_abstained'],1)

    def test_real_updates_and_old_writer_gradient(self):
        torch.set_num_threads(4)
        data=c.read(HERE.parent/'multifamily_v7/DATA.json')
        cache={k:v.float() for k,v in torch.load(HERE.parent/'multifamily_v7/run_001/FEATURES.pt',map_location='cpu',weights_only=True).items()}
        batch=train.special.tensors(data['families'][0],'cpu')
        for arm in ('B2','Ours'):
            torch.manual_seed(1209)
            net=train.models.MemoryPolicy(2048,len(data['query_vocabulary']),8,64,.99)
            before=c.model_identity(net)['sha256']
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss,_,_=train.special.losses(net,cache,batch,arm)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward();optimizer.step()
            self.assertNotEqual(before,c.model_identity(net)['sha256'])
            evidence=train.gradient_audit(net,cache,batch,arm)
            self.assertGreater(evidence['critical_writer_auxiliary_gradient_norm'],0)
            self.assertGreater(evidence['action_memory_gradient_norm'],0)


if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    began=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TeacherTest))
    c.write(HERE/'CPU_TEST_RESULT.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-began,
        real_cache_used=True,disposable_probe_optimizer_updates=4 if result.wasSuccessful() else None,
        base_optimizer_updates=0,new_qwen_forwards=0,gpu_used=False),True)
    raise SystemExit(not result.wasSuccessful())
