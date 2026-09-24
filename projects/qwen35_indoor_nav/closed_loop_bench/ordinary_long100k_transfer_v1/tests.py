"""Load the real100k checkpoint and test exact causal adapter behavior on CPU."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch,common as u
from adapter import Candidate
from evaluate import audit_prefix
torch.set_num_threads(2)
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=u.read(u.HERE/'PROTOCOL.json');cls.net=Candidate(cls.cfg,'cpu');cls.initial=cls.net.identity()
        torch.manual_seed(1209);cls.h=torch.randn(5,2048);cls.z=torch.randn(5,4)
    def test_loaded_exact_final_checkpoint(self):self.assertEqual(self.initial['sha256'],self.cfg['memory_state_sha256'])
    def test_adapter_is_original_chronological_step(self):
        self.net.reset();state=self.net.net.reset(1,'cpu')
        with torch.inference_mode():
            for h,z in zip(self.h,self.z):
                expected,state,_=self.net.net.step(h[None],z[None],state);actual,_=self.net.step(h[None],z[None]);self.assertTrue(torch.equal(expected[0],actual))
        self.assertEqual(self.net.updates,5);self.assertEqual(self.net.identity(),self.initial)
    def test_reset_does_not_leak_previous_episode(self):
        with torch.inference_mode():
            self.net.reset();a,_=self.net.step(self.h[:1],self.z[:1]);self.net.step(self.h[1:2],self.z[1:2]);self.net.reset();b,_=self.net.step(self.h[:1],self.z[:1]);self.assertTrue(torch.equal(a,b));self.assertEqual(self.net.updates,1)
    def test_warmup_prediction_does_not_write_memory(self):
        self.net.reset();before=self.net.state_identity()
        with torch.inference_mode():self.net.first(self.h[:1],self.z[:1])
        self.assertEqual(before,self.net.state_identity());self.assertEqual(self.net.updates,0)
    def test_invalid_feature_rejected(self):
        with torch.inference_mode(),self.assertRaises(ValueError):self.net.first(torch.full((1,2048),float('nan')),self.z[:1])
    def test_stop_budget(self):self.assertEqual(u.c.advance('STOP',499,False,500),(500,True))
    def test_prefix_comparison_rejects_mismatch(self):
        a=dict(raw={},processed={},base_logits=[1.,0.,0.,0.],base_action='move_forward')
        with self.assertRaises(ValueError):audit_prefix(a,dict(a,raw={'instruction':'changed'}))
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));u.write(u.HERE/'CPU_TEST_RESULT.json',dict(tests=r.testsRun,passed=r.wasSuccessful(),actual_checkpoint_loaded=True,scope='Actual100k head onCPU, no QwenGPU or navigation claim'));raise SystemExit(not r.wasSuccessful())
