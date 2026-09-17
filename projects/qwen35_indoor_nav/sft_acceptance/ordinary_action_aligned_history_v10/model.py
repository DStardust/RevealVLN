import importlib.util as _iu
from pathlib import Path as _P
_s = _iu.spec_from_file_location('expanded_reuse_model', _P(__file__).with_name('reuse.py'))
_r = _iu.module_from_spec(_s); _s.loader.exec_module(_r)
_r.execute('model.py', globals())

_original_build_policy=build_policy
def build_policy(seed=1109):
    p=_original_build_policy(seed)
    # Convert after native initialization: no seed/RNG consumption change.
    p.exec_embed.float()
    p.action_query=nn.Parameter(p.action_query.detach().float())
    return p
def assert_master_precision(policy,opt,source_state):
    names=dict(policy.named_parameters())
    for name in ('action_query','exec_embed.weight'):
        p=names[name];assert p.dtype==torch.float32 and p.requires_grad
        assert torch.equal(p.detach().to(torch.bfloat16).cpu(),source_state['trainable'][name].to(torch.bfloat16))
        slot=opt.state[p];assert slot['exp_avg'].dtype==slot['exp_avg_sq'].dtype==torch.float32
        assert torch.isfinite(slot['exp_avg']).all() and torch.isfinite(slot['exp_avg_sq']).all()
    assert all(p.dtype==torch.float32 for p in policy.parameters() if p.requires_grad)
