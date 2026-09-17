"""Fixed source policy distillation with no persistent teacher backbone."""
import torch
F=torch.nn.functional
class Anchor:
    def __init__(self,policy,source):
        self.policy=policy
        self.params={n:p for n,p in policy.named_parameters() if p.requires_grad}
        assert set(source)==set(self.params)
        self.reference={n:source[n].detach().to(device=p.device,dtype=p.dtype).clone() for n,p in self.params.items()}
        self.original_cpu={n:x.detach().cpu().clone() for n,x in self.reference.items()}
        self.identities={n:id(p) for n,p in self.params.items()}
        self.calls=0;self.decisions=0
    def forward(self,kwargs,verify=False):
        assert all(id(p)==self.identities[n] for n,p in self.params.items())
        assert all(p.grad is None for p in self.params.values()),'TEACHER_AFTER_STUDENT_GRAPH_FORBIDDEN'
        saved={n:p.detach().clone() for n,p in self.params.items()}
        cpu_rng=torch.get_rng_state()
        cuda_rng=torch.cuda.get_rng_state() if next(iter(self.params.values())).is_cuda else None
        try:
            with torch.no_grad():
                for n,p in self.params.items():p.copy_(self.reference[n])
                logits=self.policy(**kwargs).detach()
                assert not logits.requires_grad and torch.isfinite(logits).all()
        finally:
            with torch.no_grad():
                for n,p in self.params.items():p.copy_(saved[n])
            if verify:
                assert all(torch.equal(p,saved[n]) for n,p in self.params.items()),'STUDENT_RESTORE_CHANGED'
                assert all(torch.equal(x.cpu(),self.original_cpu[n]) for n,x in self.reference.items()),'REFERENCE_DRIFT'
                assert torch.equal(torch.get_rng_state(),cpu_rng),'TEACHER_CPU_RNG_CHANGED'
                if cuda_rng is not None:assert torch.equal(torch.cuda.get_rng_state(),cuda_rng),'TEACHER_CUDA_RNG_CHANGED'
        assert all(p.grad is None for p in self.params.values()),'TEACHER_GRAD_LEAK'
        self.calls+=1;self.decisions+=len(logits)
        return logits
def per_sample_kl(student,teacher):
    assert not teacher.requires_grad
    logp=F.log_softmax(student.float(),dim=-1)
    logq=F.log_softmax(teacher.float(),dim=-1)
    kl=(logq.exp()*(logq-logp)).sum(-1)
    assert torch.isfinite(kl).all() and bool((kl>=-2e-5).all()),'INVALID_KL'
    return kl
def first_parity(student,teacher):
    x=student.detach().float();y=teacher.detach().float()
    delta=x-y;absolute=float(delta.abs().max())
    relative=float(delta.norm()/y.norm().clamp_min(1e-12))
    argmax=bool((x.argmax(-1)==y.argmax(-1)).all())
    kl=float(per_sample_kl(x,y).abs().mean())
    assert absolute<=.15 and relative<=.03 and argmax and kl<=1e-3,'INITIAL_REFERENCE_FORWARD_MISMATCH'
    return dict(max_abs=absolute,relative_l2=relative,all_argmax_identical=argmax,mean_abs_kl=kl)
