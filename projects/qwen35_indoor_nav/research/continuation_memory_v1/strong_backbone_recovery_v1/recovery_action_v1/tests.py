"""CPU checks for causal replay, shared initialization and actual long-unroll gradients."""
import json
from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parent.parent)]
import torch
from prepare import cutoff
from recovery_model import RecoveryMemory,sequence_logits
from replay import replay_token


class Contracts(unittest.TestCase):
    def test_terminal_query_is_not_extra_observation(self):
        r=dict(actions=[1,1,2,0],query_steps=[0,3])
        self.assertEqual([replay_token(r,0,i,[10,11,12,13],99) for i in range(4)],[11,11,12,99])
        self.assertEqual(replay_token(r,3,0,[10,11,12,13],99),10)
        self.assertEqual(replay_token(r,3,1,[10,11,12,13],99),99)

    def test_cutoff_is_existing_causal_query(self):
        trace=[dict(event='generation',environment_step=x) for x in [0,4,8,60,64,68]]
        self.assertEqual(cutoff(trace),64)

    def test_same_initialization_and_zero_action(self):
        torch.manual_seed(42);a=RecoveryMemory(12,'CONCAT');b=RecoveryMemory(12,'EVIDENCE');b.load_state_dict(a.state_dict())
        self.assertTrue(all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items()))
        x=torch.randn(1,12)
        for m in (a,b):self.assertEqual(float(m.action_delta(x,m.reset()).abs().max()),0)

    def test_empty_memory_cancels_current_bias(self):
        m=RecoveryMemory(12,'EVIDENCE')
        with torch.no_grad():m.actor[-1].weight.normal_();m.actor[-1].bias.fill_(5)
        self.assertTrue(torch.equal(m.action_delta(torch.randn(2,12),m.reset(2)),torch.zeros(2,4)))

    def test_late_loss_updates_early_writer_and_parameters(self):
        torch.manual_seed(42);m=RecoveryMemory(12,'EVIDENCE');optimizer=torch.optim.AdamW(m.parameters(),lr=.01)
        x=torch.randn(24,12,requires_grad=True)
        row=dict(memory_features=x,query_steps=torch.tensor([23]),actor_features=torch.randn(1,12),base_logits=torch.zeros(1,4))
        before=m.writer.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad();x.grad=None
            loss=torch.nn.functional.cross_entropy(sequence_logits(m,row),torch.tensor([2]));loss.backward();optimizer.step()
        self.assertGreater(float(x.grad[0].abs().sum()),0)
        self.assertFalse(torch.equal(before,m.writer.weight))
        self.assertTrue(torch.isfinite(loss))

    def test_reset_does_not_retain_episode(self):
        m=RecoveryMemory(12,'EVIDENCE');m.update(torch.ones(1,12),m.reset())
        self.assertEqual(float(m.reset().abs().sum()),0)


if __name__=='__main__':
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    path=Path(__file__).with_name('CPU_TEST_RESULT.json')
    path.write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
