"""Matched terminal ranking plus protection of individual V13 continue decisions."""
import time
import torch
import torch.nn.functional as F

def margins(theta,x,reference):return reference+x@theta[:-1]+theta[-1]

def losses(theta,x,reference,g):
    m=margins(theta,x,reference)
    neg=m[g['negative_indices']];mask=g['negative_mask'];w=g['trajectory_weights']
    count=mask.sum(1).clamp(min=1)
    negative=(w*(F.softplus(neg)*mask).sum(1)/count).sum()
    pm=g['positive_mask'];pw=w*pm
    positive=(pw*F.softplus(-m[g['positive_indices']])).sum()/pw.sum().clamp(min=1e-12)
    pair_loss=F.softplus(1+m[g['pair_negative']]-m[g['pair_positive']])
    ranking=(g['pair_weights']*pair_loss).sum()
    # V19 preserved only an aggregate false-STOP budget. Here every observed
    # negative is tied to its own reference margin, with room up to -0.25.
    target=reference[g['negative_indices']].clamp(min=-.25)
    excess=F.relu(neg-target).square().masked_fill(~mask,0)
    protection=(w*excess.max(1).values).sum()
    regularization=.0005*theta.square().sum()
    parts=dict(terminal=positive,negative=negative,matched_ranking=ranking,
               reference_continue_protection=protection,regularization=regularization)
    return sum(parts.values()),parts

def optimize(h,reference,scale,g,progress):
    x=h.double()/scale;theta=torch.zeros(h.shape[1]+1,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([theta],lr=1,max_iter=200,max_eval=250,history_size=50,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    calls=0;start=time.monotonic();first=None
    def closure():
        nonlocal calls,first
        calls+=1
        if calls>300 or time.monotonic()-start>1200:raise RuntimeError('FIT_COMPUTE_BOUND')
        opt.zero_grad();loss,parts=losses(theta,x,reference.double(),g)
        if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
        loss.backward()
        if not torch.isfinite(theta.grad).all():raise ValueError('NONFINITE_GRADIENT')
        if first is None:first=float(theta.grad.norm())
        progress(dict(closure=calls,loss=float(loss.detach()),parts={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(theta.grad.norm()),seconds=time.monotonic()-start))
        return loss
    opt.step(closure)
    return theta.detach(),dict(iterations=int(opt.state[theta]['n_iter']),closures=calls,seconds=time.monotonic()-start,first_gradient_norm=first,update_norm=float(theta.detach().norm()))
