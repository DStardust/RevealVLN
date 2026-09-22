"""Convex boundary ranking plus the unchanged V13 anchored BCE objective."""
import time
import torch

def ranking_loss(margins,positive,negative,weights,gap):
    return (weights*torch.nn.functional.softplus(gap-margins[positive]+margins[negative])).sum()

def optimize(h,z,y,w,positive,negative,pair_w,cfg,log):
    scale=(w[:,None]*h.square()).sum().sqrt();assert scale>0
    x=h/scale;offset=z[:,3]-z[:,:3].max(1).values
    theta=torch.zeros(h.shape[1]+1,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([theta],lr=1,max_iter=200,max_eval=250,history_size=50,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    calls=0;began=time.monotonic()
    def closure():
        nonlocal calls
        calls+=1;assert calls<=250 and time.monotonic()-began<600,'OPTIMIZER_BOUND'
        optimizer.zero_grad();margin=offset+x@theta[:-1]+theta[-1]
        bce=(w*torch.nn.functional.binary_cross_entropy_with_logits(margin,y,reduction='none')).sum()
        ranking=ranking_loss(margin,positive,negative,pair_w,cfg['ranking_margin'])
        reg=.5*cfg['regularization']*theta.square().sum()
        loss=bce+cfg['ranking_weight']*ranking+reg
        assert torch.isfinite(loss),'NONFINITE_LOSS';loss.backward()
        assert theta.grad is not None and torch.isfinite(theta.grad).all(),'NONFINITE_GRADIENT'
        log(dict(closure=calls,loss=float(loss.detach()),bce=float(bce.detach()),ranking=float(ranking.detach()),regularization=float(reg.detach()),gradient_norm=float(theta.grad.norm()),parameter_norm=float(theta.detach().norm())))
        return loss
    optimizer.step(closure)
    assert torch.isfinite(theta).all(),'NONFINITE_PARAMETERS'
    return theta.detach(),scale,dict(closure_calls=calls,optimizer_iterations=int(optimizer.state[theta]['n_iter']),wall_seconds=time.monotonic()-began,scale=float(scale),residual_norm=float(theta.detach().norm()))
