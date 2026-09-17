"""CPU math, weighted distributed reduction, state restoration, and source tests."""
import ast,copy,hashlib,json,os,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
assert not torch.cuda.is_initialized()
a=runpy.run_path(str(HERE/'anchor.py'));checks=[]
torch.manual_seed(1209)
x=torch.randn(9,4,requires_grad=True);q=torch.randn(9,4).detach();w=torch.tensor([1.,3.2,1.,1.,3.2,1.,1.,3.2,1.])
kl=a['per_sample_kl'](x,q)
ref=torch.nn.functional.kl_div(x.log_softmax(-1),q.log_softmax(-1),reduction='none',log_target=True).sum(-1)
assert torch.allclose(kl,ref,atol=1e-7,rtol=1e-6);checks.append('kl_matches_installed_torch')
g=torch.autograd.grad((kl*w).sum()/w.sum(),x)[0]
expected=(x.detach().softmax(-1)-q.softmax(-1))*w[:,None]/w.sum()
assert torch.allclose(g,expected,atol=1e-7,rtol=1e-5);checks.append('kl_gradient_direction')
z=x.detach().clone();assert torch.equal(a['per_sample_kl'](z,z),torch.zeros(9));checks.append('identical_distribution_zero_kl')
p=a['first_parity'](z,z);assert p['all_argmax_identical'] and p['mean_abs_kl']==0;checks.append('initial_parity')
targets=torch.arange(9)%4
def loss(v,idx):
    return ((torch.nn.functional.cross_entropy(v,targets[idx],reduction='none')+a['per_sample_kl'](v,q[idx]))*w[idx]).sum()
full=x.detach().clone().requires_grad_(True);full_loss=loss(full,slice(None))/w.sum()
full_grad=torch.autograd.grad(full_loss,full)[0]
parts=[]
for idx in [slice(0,3),slice(3,6),slice(6,9)]:
    local=x.detach()[idx].clone().requires_grad_(True)
    parts.append(torch.autograd.grad(loss(local,idx)*3/w.sum(),local)[0]/3)
assert torch.allclose(torch.cat(parts),full_grad,atol=1e-7,rtol=1e-5);checks.append('three_rank_global_weighted_loss')
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head=torch.nn.Linear(3,4);self.fail=False
    def forward(self,input_ids):
        if self.fail:raise RuntimeError('injected_teacher_exception')
        return self.head(input_ids)
m=Tiny();opt=torch.optim.AdamW(m.parameters(),lr=.01);inp=torch.randn(5,3)
m(inp).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
source={n:p.detach().clone() for n,p in m.named_parameters()}
teacher=m(inp).detach().clone()
anchor=a['Anchor'](m,source)
identity=[id(p) for p in m.parameters()]
opt_before=copy.deepcopy(opt.state_dict())
with torch.no_grad():
    for p in m.parameters():p.add_(torch.randn_like(p)*.01)
student={n:p.detach().clone() for n,p in m.named_parameters()}
cpu_rng=torch.get_rng_state().clone()
y=anchor.forward(dict(input_ids=inp),verify=True)
assert torch.equal(y,teacher) and not y.requires_grad;checks.append('fixed_source_not_current_student')
assert all(torch.equal(p,student[n]) for n,p in m.named_parameters());checks.append('student_parameters_exactly_restored')
assert [id(p) for p in m.parameters()]==identity;checks.append('parameter_object_identity_preserved')
assert torch.equal(torch.get_rng_state(),cpu_rng);checks.append('no_teacher_rng_consumption')
for k,d in opt_before['state'].items():
    for n,v in d.items():assert torch.equal(v,opt.state_dict()['state'][k][n])
checks.append('optimizer_state_untouched')
assert anchor.calls==1 and anchor.decisions==5 and all(p.grad is None for p in m.parameters());checks.append('forward_accounting_no_teacher_grads')
m.fail=True
try:anchor.forward(dict(input_ids=inp),verify=True);raise AssertionError('injection_not_raised')
except RuntimeError as e:assert str(e)=='injected_teacher_exception'
assert all(torch.equal(p,student[n]) for n,p in m.named_parameters()) and anchor.calls==1;checks.append('exception_finally_restores_student')
m.fail=False
logits=m(inp);(torch.nn.functional.cross_entropy(logits,torch.arange(5)%4)+a['per_sample_kl'](logits,y).mean()).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());checks.append('student_backward_after_swap')
r=runpy.run_path(str(HERE/'reuse.py'))
for n in ('train_filestore.py','supervise_filestore.py','model.py','control.py','launcher.py','lease_run.py'):ast.parse(r['source'](n))
checks.append('compiled_sources_parse')
for directory in (HERE,LINE/'reviews/Q35N_ORDINARY_POLICY_PRESERVATION_V9',LINE/'closed_loop_bench/ordinary_policy_preservation_dev_v9'):
    for p in directory.glob('*.py'):ast.parse(p.read_text())
checks.append('all_wrapper_sources_parse')
assert not torch.cuda.is_initialized()
result=dict(status='PASS',unix=time.time(),checks=checks,count=len(checks),cuda_initialized=False,
    navigation_gain_verified=False,source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['anchor.py','reuse.py','tests.py']})
with (HERE/'CPU_TEST_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
