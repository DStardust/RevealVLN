"""Same 8x64 architecture; arm differences are only masked auxiliary supervision."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from select_action import tensor_indices
models=load('v16_frozen_memory_architecture',HERE.parent/'query_semantics_v11/model.py')

def initialize(seed):
    torch.manual_seed(seed)
    net=models.MemoryPolicy(2048,1,8,64,.99)
    net.state_head=nn.Sequential(nn.Linear(512,128),nn.Tanh(),nn.Linear(128,4))
    path=HERE.parent/'fork_balanced_v14/strong_state_run_001'/f'INITIAL_{seed}.pt'
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    return net

FIELDS=('stop_at_cutoff','stops','suffix_anchor_before_final','terminal_at_final')

def class_weights(families):
    positive=[0]*4;negative=[0]*4;qp=qn=0
    for family in families:
        if family['split']!='FIT':raise ValueError('TRAINING_STATISTICS_REQUIRE_FIT')
        for row in family['sequences']:
            for state,mask in zip(row['state_targets'],row['state_masks']):
                if mask:
                    for k,z in enumerate(state):positive[k]+=z;negative[k]+=1-z
            for y,mask in zip(row['y'],row['query_masks']):
                if mask:qp+=y;qn+=1-y
    def weights(p,n):
        if not p or not n:raise ValueError('MISSING_AUXILIARY_CLASS_SUPPORT')
        return [(p+n)/(2*n),(p+n)/(2*p)]
    return dict(state=[weights(p,n) for p,n in zip(positive,negative)],query=weights(qp,qn),
                counts=dict(state_positive=positive,state_negative=negative,query_positive=qp,query_negative=qn))

def batch(family,device):
    rows=family['sequences'];maximum=max(len(r['features']) for r in rows)
    def pad(key,zero):return [r[key]+[zero]*(maximum-len(r[key])) for r in rows]
    result=dict(family=family,indices=torch.tensor(pad('features',0),device=device),
        targets=torch.tensor(pad('targets',0),device=device),
        action_mask=torch.tensor(pad('action_masks',0),device=device,dtype=torch.float32),
        state=torch.tensor(pad('state_targets',[0]*4),device=device,dtype=torch.float32),
        state_mask=torch.tensor(pad('state_masks',0),device=device,dtype=torch.float32),
        y=torch.tensor(pad('y',0),device=device,dtype=torch.float32),
        query_mask=torch.tensor(pad('query_masks',0),device=device,dtype=torch.float32),
        alive=torch.tensor([[True]*len(r['features'])+[False]*(maximum-len(r['features'])) for r in rows],device=device),
        preservation_mask=torch.tensor([[int(t<min(156,r['cutoff']+1)) for t in range(maximum)] for r in rows],device=device,dtype=torch.float32),
        query=torch.tensor([[[float(q[k]) for k in FIELDS] for q in r['query_contexts']]+[[0]*4]*(maximum-len(r['features'])) for r in rows],device=device))
    return result

def encode_masked(net,features,alive,retain_steps=()):
    memory=net.reset(features.shape[0],features.device);states=[];writes={}
    for t in range(features.shape[1]):
        updated,write=net.update(features[:,t],memory,t in retain_steps)
        memory=torch.where(alive[:,t,None,None],updated,memory)
        states.append(memory)
        if t in retain_steps:writes[t]=write
    return torch.stack(states,1),writes

def average(values,mask):
    if not bool(mask.sum()>0):return values.sum()*0
    return (values*mask).sum()/mask.sum()

def losses(net,cache,b,arm,weights,retain_steps=()):
    features=cache['features'][b['indices']];native=cache['logits'][b['indices']]
    states,writes=encode_masked(net,features,b['alive'],retain_steps)
    logits=net.action_logits(states.flatten(0,1),native.flatten(0,1),features.flatten(0,1)).view(*b['indices'].shape,4)
    ce=F.cross_entropy(logits.flatten(0,1),b['targets'].flatten(),reduction='none').view_as(b['targets'])
    action=average(ce,b['action_mask'])
    kl=F.kl_div(logits.log_softmax(-1),native.softmax(-1),reduction='none').sum(-1)
    preservation=average(kl,b['preservation_mask'])
    state_logits=net.state_head(states.flatten(2))
    sw=torch.tensor(weights['state'],device=features.device)
    state_weight=torch.where(b['state'].bool(),sw[:,1],sw[:,0])*b['state_mask'][...,None]
    state_loss=average(F.binary_cross_entropy_with_logits(state_logits,b['state'],reduction='none'),state_weight)
    query_logits=net.reader(states.flatten(0,1),b['query'].flatten(0,1)).view_as(b['y'])
    qw=weights['query'];query_weight=torch.where(b['y'].bool(),qw[1],qw[0])*b['query_mask']
    query_loss=average(F.binary_cross_entropy_with_logits(query_logits,b['y'],reduction='none'),query_weight)
    # Additional equal-weight fork CE only for actually identical current inputs.
    fork_terms=[];forks=[];rows=b['family']['sequences'];admission=b['family']['teacher_admission']
    cells=b['family']['cells'];prefixes=b['family']['prefixes']
    lookup={(p['history_id'],p['task_id']):i for i,p in enumerate(prefixes)}
    for task in ('task_A','task_B','task_T'):
        for suffix in ('','_R'):
            choices=[admission['selected_cells'].get(str(lookup[(h+suffix,task)]),[]) for h in ('H_A','H_B')]
            if any(len(x)!=1 for x in choices):continue
            left,right=[cells[x[0]]['sequence'] for x in choices];l,r=rows[left],rows[right]
            for offset in range(min(len(l['features'])-l['cutoff'],len(r['features'])-r['cutoff'])):
                tl,tr=l['cutoff']+offset,r['cutoff']+offset
                if l['features'][tl]!=r['features'][tr]:break
                if l['targets'][tl]==r['targets'][tr]:continue
                if l['action_masks'][tl] and r['action_masks'][tr]:
                    fork_terms.append((ce[left,tl]+ce[right,tr])/2)
                    forks.append(dict(left=left,right=right,left_step=tl,right_step=tr,task=task,stratum=suffix))
                break
    fork=torch.stack(fork_terms).mean() if fork_terms else action*0
    auxiliary={'B1':action*0,'B2':state_loss,'Ours':query_loss}[arm]
    predictions=tensor_indices(logits)
    query_probability=compose_state_probability(state_logits.sigmoid(),b['query']) if arm=='B2' else query_logits.sigmoid() if arm=='Ours' else None
    diagnosed_query_bce=None if query_probability is None else float(average(F.binary_cross_entropy(query_probability.clamp(1e-7,1-1e-7),b['y'],reduction='none'),b['query_mask']).detach())
    stats=dict(action_ce=float(action.detach()),fork_ce=float(fork.detach()),fork_pairs=len(forks),
        exact_state_bce=float(state_loss.detach()) if arm=='B2' else None,crossed_bce=float(query_loss.detach()) if arm=='Ours' else None,
        query_diagnostic_bce=diagnosed_query_bce,query_diagnostic_rule='analytic exact-state composition' if arm=='B2' else 'trained result reader' if arm=='Ours' else 'not trained',
        preservation_kl=float(preservation.detach()),
        action_owners=int(b['action_mask'].sum()),action_correct=int(((predictions==b['targets'])*b['action_mask']).sum()),
        state_scalar_supervisions=int(b['state_mask'].sum())*4,query_supervisions=int(b['query_mask'].sum()),
        native_teacher_conflicts=int(((tensor_indices(native)!=b['targets'])*b['action_mask']).sum()),
        native_stop_teacher_continue=int(((tensor_indices(native)==3)&(b['targets']!=3)&b['action_mask'].bool()).sum()),
        valid_causal_steps=int(b['alive'].sum()),maximum_unroll=int(b['alive'].shape[1]))
    return action+fork+preservation+auxiliary,stats,dict(states=states,writes=writes,logits=logits,state_logits=state_logits,query_logits=query_logits,forks=forks)

def ordinary_loss(net,cache,records):
    maximum=max(len(r['features']) for r in records);device=cache['features'].device
    indices=torch.tensor([r['features']+[0]*(maximum-len(r['features'])) for r in records],device=device)
    targets=torch.tensor([r['targets']+[0]*(maximum-len(r['targets'])) for r in records],device=device)
    alive=torch.tensor([[True]*len(r['features'])+[False]*(maximum-len(r['features'])) for r in records],device=device)
    features=cache['features'][indices];states,_=encode_masked(net,features,alive)
    logits=net.action_logits(states.flatten(0,1),cache['logits'][indices].flatten(0,1),features.flatten(0,1))
    ce=F.cross_entropy(logits,targets.flatten(),reduction='none').view_as(alive)
    return ((ce*alive).sum(-1)/alive.sum(-1)).mean()

def compose_state_probability(z,query):
    stop,stops,anchor,terminal=query.unbind(-1)
    value=torch.where(stop.bool(),z[...,3],torch.where(anchor.bool(),torch.ones_like(z[...,1]),z[...,1]))
    return value*stops*torch.where(stop.bool(),torch.ones_like(terminal),terminal)
