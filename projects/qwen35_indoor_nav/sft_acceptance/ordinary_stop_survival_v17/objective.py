"""Trajectory-level false-stop surrogate with fixed motion scores and V13 anchor."""
import time
import torch
def margins(theta,x,reference):return reference+x@theta[:-1]+theta[-1]
def losses(theta,x,reference,negative_indices,negative_mask,positive_indices,positive_mask,trajectory_weights):
    m=margins(theta,x,reference)
    values=m[negative_indices].masked_fill(~negative_mask,-1e9)
    # Smooth union of bad stopping opportunities, not an estimated closed-loop SR.
    negative=torch.nn.functional.softplus(torch.logsumexp(values,dim=1))
    positive=torch.nn.functional.softplus(-m[positive_indices])*positive_mask
    neg=(trajectory_weights*negative).sum();pos=(trajectory_weights*positive).sum();reg=.0005*theta.square().sum()
    return neg+pos+reg,dict(negative_bag=neg,first_legal_stop=pos,regularization=reg)
def optimize(h,reference,scale,groups,progress):
    x=h/scale;theta=torch.zeros(h.shape[1]+1,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([theta],lr=1,max_iter=200,max_eval=250,history_size=50,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe');calls=0;start=time.monotonic();first=None
    def closure():
        nonlocal calls,first
        calls+=1;assert calls<=250 and time.monotonic()-start<1200,'FIT_BOUND'
        opt.zero_grad();value,parts=losses(theta,x,reference,**groups);assert torch.isfinite(value)
        value.backward();assert torch.isfinite(theta.grad).all(),'NONFINITE_GRADIENT'
        if first is None:first=float(theta.grad.norm())
        progress(dict(closure=calls,loss=float(value.detach()),parts={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(theta.grad.norm()),seconds=time.monotonic()-start))
        return value
    opt.step(closure)
    return theta.detach(),dict(iterations=int(opt.state[theta]['n_iter']),closures=calls,seconds=time.monotonic()-start,first_gradient_norm=first,update_norm=float(theta.detach().norm()))
