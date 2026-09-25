"""Gradient and causality checks for dense writes and sparse actor queries."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import common as u
from memory_v2 import ExecutionMemory

torch.set_num_threads(2);torch.manual_seed(42)
model=ExecutionMemory(16)
# A trained readout must transmit a terminal action loss to an old write even
# when the actor receives a different feature stream.
with torch.no_grad():model.actor[-1].weight.normal_(std=.01)
history=torch.randn(1,45,16,requires_grad=True);actor=torch.randn(1,45,16)
out=model(history,torch.tensor([45]),actor)
loss=torch.nn.functional.cross_entropy(out['delta'][:,-1],torch.tensor([1]));loss.backward()
assert history.grad[0,0].abs().sum()>0
changed=history.detach().clone();changed[:,30:]+=20
with torch.no_grad():second=model(changed,torch.tensor([45]),actor)
assert torch.equal(out['delta'][:,:30].detach(),second['delta'][:,:30])
assert model.writer.weight.grad.abs().sum()>0
u.write(u.HERE/'DENSE_MEMORY_CPU_TEST_RESULT.json',dict(status='CPU_CONTRACT_CHECKED',old_write_receives_gradient=True,
    future_frame_change_preserves_earlier_actions=True,scope='Synthetic CPU causality/gradient test; no navigation effect'))
