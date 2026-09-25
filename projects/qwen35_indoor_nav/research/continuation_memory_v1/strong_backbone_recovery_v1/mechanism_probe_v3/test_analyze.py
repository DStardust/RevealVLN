from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze
import torch


class ProbeTests(unittest.TestCase):
    def test_real_readout_dependency_and_gradient_scope(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        for architecture in ('CONCAT','LOCAL'):
            model=analyze.ConfirmationMemory(16,architecture)
            with torch.no_grad():model.actor[-1].weight.normal_(0,.1)
            model.requires_grad_(False)
            row=dict(memory_features=torch.randn(13,16),actor_features=torch.randn(2,16),base_logits=torch.zeros(2,4),
                query_steps=torch.tensor([3,12]),targets=torch.tensor([1,0]))
            full,zero,gradient=analyze.compare(model,row)
            self.assertEqual(full.shape,(2,4));self.assertFalse(torch.equal(full,zero))
            self.assertEqual(gradient['old_input_gradient_mass']>0,architecture=='CONCAT')
            self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
