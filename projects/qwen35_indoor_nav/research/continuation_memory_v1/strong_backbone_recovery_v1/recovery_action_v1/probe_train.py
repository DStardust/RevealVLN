"""Actual cached observations through the training objective, without efficacy claims."""
import argparse
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parent.parent)]
import common as u
import torch
from recovery_model import RecoveryMemory
from train import objective


def main(run):
    torch.set_num_threads(4)
    rp=run/'probe_features/evaluation/probe_001/episodes/0/CACHE.pt'
    op=run/'probe_preserve_features/evaluation/probe_001/episodes/174/CACHE.pt'
    recovery=torch.load(rp,map_location='cpu',weights_only=True)
    ordinary=torch.load(op,map_location='cpu',weights_only=True)
    assert recovery['partition']==ordinary['partition']=='FIT'
    assert recovery['kind']=='RECOVERY' and ordinary['kind']=='PRESERVATION'
    assert recovery['cutoff']==64 and not recovery['known'][0]
    out=run/'training_probe';out.mkdir(exist_ok=False);torch.manual_seed(42)
    initial=RecoveryMemory(3584,'CONCAT').state_dict();results={}
    for arm in ('CONCAT','EVIDENCE'):
        model=RecoveryMemory(3584,arm);model.load_state_dict(initial)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01)
        x=recovery['memory_features'].clone().requires_grad_();r=dict(recovery,memory_features=x);rows=[]
        for step in range(3):
            optimizer.zero_grad();x.grad=None
            loss,items=objective(model,r,ordinary,torch.ones(4));loss.backward()
            assert torch.isfinite(loss)
            gradient=float(model.writer.weight.grad.norm());early=float(x.grad[0].norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            rows.append(dict(step=step+1,loss=float(loss.detach()),writer_gradient=gradient,initial_observation_gradient=early,**items))
        assert gradient>0 and early>0
        changed=not torch.equal(model.writer.weight.detach(),initial['writer.weight']);assert changed
        folder=out/arm;folder.mkdir();torch.save(dict(model=model.state_dict(),arm=arm,step=3),folder/'FINAL.pt')
        results[arm]=dict(rows=rows,writer_changed=changed,weights_sha256=u.sha(folder/'FINAL.pt'))
    u.write(out/'RESULT.json',dict(status='REAL_FORWARD_BACKWARD_UPDATE_VERIFIED',gpu_hours=0,
        recovery_cache_sha256=u.sha(rp),ordinary_cache_sha256=u.sha(op),results=results,
        method_benefit='UNKNOWN',training_sample_selection='First registered FIT recovery and preservation, no score selection',
        scientific_scope='Debug only; no base updates, no claim of navigation efficacy'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
