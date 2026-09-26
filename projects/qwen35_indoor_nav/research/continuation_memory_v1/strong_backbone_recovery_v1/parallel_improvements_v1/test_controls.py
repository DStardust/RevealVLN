from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from control_models import UnitReadoutMemory


class ControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_positive_scaling_cannot_change_readout(self):
        head=UnitReadoutMemory(16,'RECURRENT')
        with torch.no_grad():
            head.actor[-1].weight.normal_(0,.1)
        x=torch.randn(2,16);m=torch.randn(2,8,64)
        self.assertTrue(torch.allclose(head.action_delta(x,m),head.action_delta(x,19*m),atol=1e-6))
        self.assertTrue(torch.allclose(head.read_state(m).norm(dim=-1),torch.ones(2)))
        self.assertTrue(torch.isfinite(head.action_delta(x,torch.zeros_like(m))).all())

    def test_history_gradient_and_reset_follow_each_control(self):
        for mode in ('LOCAL','EMA','RECURRENT'):
            head=UnitReadoutMemory(16,mode)
            with torch.no_grad():head.actor[-1].weight.normal_(0,.1)
            x=torch.randn(12,16,requires_grad=True);m=head.reset()
            for frame in x:m=head.update(frame[None],m)
            head.action_delta(torch.ones(1,16),m)[0,0].backward()
            self.assertEqual(bool(x.grad[0].abs().sum()>0),mode!='LOCAL')
            self.assertTrue(torch.equal(head.reset(),torch.zeros(1,8,64)))
            self.assertEqual(head.recurrent.weight.grad is not None,mode=='RECURRENT')

    def test_fresh_actor_has_no_effect_and_parameter_storage_matches(self):
        heads=[UnitReadoutMemory(16,k) for k in ('LOCAL','EMA','RECURRENT')]
        initial=heads[0].state_dict()
        for h in heads:
            h.load_state_dict(initial)
            m=h.update(torch.randn(1,16),h.reset())
            self.assertTrue(torch.equal(h.action_delta(torch.randn(1,16),m),torch.zeros(1,4)))
            self.assertEqual({k:v.shape for k,v in h.state_dict().items()},{k:v.shape for k,v in initial.items()})


if __name__=='__main__':unittest.main()
