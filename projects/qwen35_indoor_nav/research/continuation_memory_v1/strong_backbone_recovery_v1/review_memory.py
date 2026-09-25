"""Fixed final-checkpoint action diagnostics; these are not closed-loop SR."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from memory_v2 import ExecutionMemory


def main(data,training,output):
    torch.set_num_threads(4);results={}
    for split in ('FIT','CHECK'):
        pack=torch.load(data/(split+'.pt'),weights_only=True);rows=[]
        for arm in ('NATIVE','BC','B2','OURS'):
            if arm!='NATIVE':
                saved=torch.load(training/arm/'FINAL.pt',weights_only=True)
                model=ExecutionMemory(pack['groups'][0]['features'].shape[-1]).eval();model.load_state_dict(saved['model'])
            correct=total=0;details=[]
            with torch.inference_mode():
                for g in pack['groups']:
                    x=g['features'];nt,nh,ns,nf=x.shape;delta=torch.zeros_like(g['base_action_logits'])
                    if arm!='NATIVE':
                        out=model(x.reshape(nt*nh,ns,nf),g['lengths'].flatten(),g['actor_features'].reshape(nt*nh,ns,nf))
                        delta=out['delta'].reshape(nt,nh,ns,4)
                    logits=g['base_action_logits']+delta;mask=g['action_known'];pred=logits.argmax(-1)
                    correct+=int((pred[mask]==g['action_targets'][mask]).sum());total+=int(mask.sum())
                    for t in range(nt):
                        for h in range(nh):
                            step=int(g['lengths'][t,h]-1)
                            details.append(dict(family=g['family'],task=g['tasks'][t],history=g['histories'][h],
                                continuations=g['continuations'],action=int(pred[t,h,step]),target=int(g['action_targets'][t,h,step]),
                                action_known=bool(mask[t,h,step]),native_action=int(g['base_action_logits'][t,h,step].argmax())))
            rows.append(dict(arm=arm,correct=correct,total=total,teacher_action_accuracy=correct/total if total else None,details=details))
        results[split]=rows
    u.write(output,dict(status='FIXED_ACTION_DIAGNOSTIC_COMPLETE',results=results,
        scope='Exposed two-house actual-fork action diagnostic. Neither closed-loop continuation success nor unseen R2R SR.',
        method_benefit_closed_loop='NOT_MEASURED',selection='All registered final 1200-step heads; no best-step selection'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.data,a.training,a.output)
