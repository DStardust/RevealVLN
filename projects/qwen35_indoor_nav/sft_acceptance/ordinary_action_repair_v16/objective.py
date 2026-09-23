"""Frozen-feature four-row correction; no backbone or policy architecture change."""
import time
import torch
def logits(theta,x,base):return base+x@theta[:,:-1].T+theta[:,-1]
def loss(theta,x,z,y,stop_w,motion_y,motion_w):
    pred=logits(theta,x,z);margin=pred[:,3]-pred[:,:3].max(1).values
    stop=(stop_w*torch.nn.functional.binary_cross_entropy_with_logits(margin,y,reduction='none')).sum()
    motion=(motion_w*torch.nn.functional.cross_entropy(pred[:,:3],motion_y.clamp(min=0),reduction='none')).sum()
    reg=.0005*theta.square().sum()
    return stop+motion+reg,dict(stop=stop,motion=motion,regularization=reg)
def optimize(h,z,y,stop_w,motion_y,motion_w,progress):
    scale=(stop_w[:,None]*h.square()).sum().sqrt();x=h/scale;theta=torch.zeros((4,h.shape[1]+1),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([theta],lr=1,max_iter=200,max_eval=250,history_size=50,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe');began=time.monotonic();calls=0;first_gradient=None
    def closure():
        nonlocal calls,first_gradient
        calls+=1;assert calls<=250 and time.monotonic()-began<1200,'FIT_BOUND'
        opt.zero_grad();value,parts=loss(theta,x,z,y,stop_w,motion_y,motion_w);assert torch.isfinite(value);value.backward();assert torch.isfinite(theta.grad).all(),'NONFINITE_GRADIENT'
        if first_gradient is None:first_gradient=float(theta.grad.norm())
        progress(dict(closure=calls,loss=float(value.detach()),parts={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(theta.grad.norm()),seconds=time.monotonic()-began))
        return value
    opt.step(closure);assert torch.isfinite(theta).all();return theta.detach(),scale,dict(closure_calls=calls,iterations=int(opt.state[theta]['n_iter']),wall_seconds=time.monotonic()-began,first_gradient_norm=first_gradient,update_norm=float(theta.detach().norm()),changed=True if theta.norm()>0 else False)
