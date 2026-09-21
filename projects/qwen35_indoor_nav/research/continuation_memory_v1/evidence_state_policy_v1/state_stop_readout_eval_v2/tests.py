"""Check the corrected distinction without changing the frozen task evaluator."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

class EvalTests(unittest.TestCase):
    def test_exact_model_and_environment_logic_reused(self):
        for name in ('model.py','encoder.py','continuation_service.py','evaluate_continuations.py'):
            self.assertEqual(sha(HERE/name),sha(PARENT/'state_stop_readout_v1'/name))
    def test_registry_and_all_finals(self):
        runtime=load('stop_v2_test_eval',HERE/'evaluate_continuations.py');source=PARENT/'state_stop_readout_v1/runs/repair_001'
        cfg=read(PARENT/'state_stop_readout_v1/PROTOCOL.json');data=read(source/'DATA.json');reg=runtime.registry_value(data['raw_families'],cfg)
        self.assertEqual(reg,read(source/'EVALUATION_REGISTRY.json'));self.assertEqual(len(reg['slots']),768)
        for tag in reg['models']:
            folder=source/'train'/tag;self.assertEqual(sha(folder/'FINAL.pt'),read(folder/'RESULT.json')['checkpoint_sha256'])
    def test_semantic_stop_is_not_teacher_match(self):
        import torch
        audit=load('stop_v2_test_semantic',PARENT/'state_stop_readout_v1/semantic_stop_audit_v1.py')
        value=audit.metrics(torch.tensor([3,3,1]),torch.tensor([1,1,3]),torch.tensor([True,False,True]))
        self.assertEqual(value['teacher_stop_disagreement'],2);self.assertEqual(value['premature_stop'],1)
        self.assertEqual(value['ready_stop'],1);self.assertEqual(value['ready_defer'],1)
    def test_prefix_rejects_input_and_native_flips(self):
        runtime=load('stop_v2_test_eval_prefix',HERE/'evaluate_continuations.py')
        row=dict(raw={'rgb':['same']},processed={'ids':'same'},logits=[1.,0.,0.,-1.],native_action='FORWARD',executed_action='FORWARD',override=False,decision=0)
        self.assertTrue(runtime.prefix_audit([row],[row])['input_prefix_matched'])
        mismatch=dict(row,raw={'rgb':['different']})
        self.assertIn('first_divergence',runtime.prefix_audit([row],[mismatch]))
        flipped=dict(row,logits=[0.,1.,0.,-1.],native_action='LEFT',executed_action='LEFT')
        self.assertEqual(runtime.prefix_audit([row],[flipped])['argmax_flip_count'],1)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvalTests))
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),new_updates=0))
    raise SystemExit(not result.wasSuccessful())
