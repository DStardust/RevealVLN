"""Same 8x64 architecture, data and action losses; B2Fix changes only conflicting supervised KL contributions."""
import sys
import torch
from torch.nn import functional as F
from shared import *
prior=load('ident_frozen_objective',V16/'objective.py')
sys.path.insert(0,str(HERE))
initialize=prior.initialize
batch=prior.batch
encode_masked=prior.encode_masked
class_weights=prior.class_weights
ordinary_loss=prior.ordinary_loss
compose_state_probability=prior.compose_state_probability
FIELDS=prior.FIELDS
average=prior.average

def preservation_loss(logits,native,action_mask,targets,mask,repair):
    per_step=F.kl_div(logits.log_softmax(-1),native.softmax(-1),reduction='none').sum(-1)
    conflicts=action_mask.bool() & (native.argmax(-1)!=targets) & mask.bool()
    effective=mask*(~conflicts) if repair else mask
    # Retain the original denominator: untouched locations keep their original weight.
    return (per_step*effective).sum()/mask.sum() if bool(mask.sum()>0) else per_step.sum()*0,conflicts


def losses(net,cache,b,arm,weights,retain_steps=()):
    features=cache['features'][b['indices']];native=cache['logits'][b['indices']]
    states,writes=encode_masked(net,features,b['alive'],retain_steps)
    logits=net.action_logits(states.flatten(0,1),native.flatten(0,1),features.flatten(0,1)).view(*b['indices'].shape,4)
    ce=F.cross_entropy(logits.flatten(0,1),b['targets'].flatten(),reduction='none').view_as(b['targets'])
    action=average(ce,b['action_mask'])
    cutoff_mask=torch.zeros_like(b['action_mask'])
    for i,row in enumerate(b['family']['sequences']):cutoff_mask[i,row['cutoff']]=b['action_mask'][i,row['cutoff']]
    # Equal task/history weighting at the certified decision; all three arms share it.
    fork=average(ce,cutoff_mask)
    preservation,conflicts=preservation_loss(logits,native,b['action_mask'],b['targets'],b['preservation_mask'],arm=='B2Fix')
    z=net.state_head(states.flatten(2));q=net.reader(states.flatten(0,1),b['query'].flatten(0,1)).view_as(b['y'])
    sw=torch.tensor(weights['state'],device=features.device)
    sm=torch.where(b['state'].bool(),sw[:,1],sw[:,0])*b['state_mask'][...,None]
    state=average(F.binary_cross_entropy_with_logits(z,b['state'],reduction='none'),sm)
    qw=weights['query'];qm=torch.where(b['y'].bool(),qw[1],qw[0])*b['query_mask']
    query=average(F.binary_cross_entropy_with_logits(q,b['y'],reduction='none'),qm)
    terminal=average(F.binary_cross_entropy_with_logits(z[...,2],b['state'][...,2],reduction='none'),sm[...,2])
    aux={'B1':action*0,'B2':state,'Terminal':terminal,'B2Fix':state}[arm]
    loss=action+fork+preservation+aux
    stats=dict(action_ce=float(action.detach()),cutoff_ce=float(fork.detach()),preservation_kl=float(preservation.detach()),
        auxiliary=float(aux.detach()),action_correct=int(((logits.argmax(-1)==b['targets'])*b['action_mask']).sum()),
        action_count=int(b['action_mask'].sum()),cutoff_correct=int(((logits.argmax(-1)==b['targets'])*cutoff_mask).sum()),cutoff_count=int(cutoff_mask.sum()),
        state_supervisions=int(b['state_mask'].sum())*(4 if arm in ('B2','B2Fix') else 1 if arm=='Terminal' else 0),query_supervisions=0,preservation_slots=int(b['preservation_mask'].sum()),
        preservation_conflicts=int(conflicts.sum()),preservation_removed=int(conflicts.sum()) if arm=='B2Fix' else 0)
    return loss,stats,dict(states=states,writes=writes,logits=logits,state_logits=z,query_logits=q,action_loss=action+fork,auxiliary_loss=aux)
