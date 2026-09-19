"""Real-cache gradient and input isolation for the new action-loss stratum."""
import copy
import math
import os
from pathlib import Path
import sys
import time
import unittest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train
c = train.c


class ForkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.data = c.read(HERE.parent/'multifamily_v7/DATA.json')
        cls.cache = {k: v.float() for k, v in torch.load(HERE.parent/'multifamily_v7/run_001/FEATURES.pt',
                     map_location='cpu', weights_only=True).items()}

    def test_real_fork_gradient_reaches_old_write(self):
        batch = train.special.tensors(self.data['families'][0], 'cpu')
        case = batch['fork_cases'][0]
        critical = case['critical_step']
        torch.manual_seed(1209)
        net = train.models.MemoryPolicy(2048, len(self.data['query_vocabulary']), 8, 64, .99)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
        before = c.model_identity(net)['sha256']
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = train.special.losses(net, self.cache, batch, 'B1')
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            optimizer.step()
        self.assertNotEqual(before, c.model_identity(net)['sha256'])
        states, writes = net.encode(self.cache['features'][batch['prefix_indices']], retain_steps=[critical])
        loss, _ = train.special.fork_loss(net, self.cache, batch, states[:, -1])
        gradient = torch.autograd.grad(loss, writes[critical])[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.norm()), 0)
        self.assertGreater(case['old_event_gap'], 8)

    def test_no_memory_bound_and_query_firewall(self):
        batch = train.special.tensors(self.data['families'][0], 'cpu')
        net = train.models.MemoryPolicy(2048, len(self.data['query_vocabulary']), 8, 64, .99, no_memory=True)
        states, _ = net.encode(self.cache['features'][batch['prefix_indices']])
        loss, both = train.special.fork_loss(net, self.cache, batch, states[:, -1])
        self.assertEqual(both, 0)
        self.assertGreaterEqual(float(loss.detach()), math.log(2)-1e-6)
        changed = copy.copy(batch)
        changed['y'] = 1-batch['y']
        changed['contexts'] = list(reversed(batch['contexts']))
        other, _ = train.special.fork_loss(net, self.cache, changed, states[:, -1])
        self.assertTrue(torch.equal(loss, other))


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    began = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ForkTest))
    c.write(HERE/'CPU_TEST_RESULT.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), seconds=time.monotonic()-began,
        disposable_probe_optimizer_updates=2, base_updates=0, new_qwen_forwards=0, gpu_hours=0), True)
    raise SystemExit(not result.wasSuccessful())
