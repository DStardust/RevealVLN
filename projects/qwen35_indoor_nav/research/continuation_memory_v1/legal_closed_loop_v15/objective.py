"""V14 memory/actor and objectives, with correctly masked variable-length prefixes."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c
models=c.load('v15_frozen_memory_architecture',HERE.parent/'query_semantics_v11/model.py')
teacher=c.load('v15_frozen_teacher_rule',HERE.parent/'cost_teacher_v13/teacher.py')


def initialize(seed):
    torch.manual_seed(seed)
    net=models.MemoryPolicy(2048,1,8,64,.99)
    net.state_head=nn.Sequential(nn.Linear(512,128),nn.Tanh(),nn.Linear(128,4))
    initial=torch.load(HERE.parent/'fork_balanced_v14/strong_state_run_001'/f'INITIAL_{seed}.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(initial,strict=True)
    return net


def batch(family,device='cpu'):
    prefixes=family['prefixes'];maximum=max(len(p['features']) for p in prefixes)
    indices=[];targets=[];mask=[];lengths=[]
    for p in prefixes:
        n=len(p['features']);lengths.append(n)
        indices.append(p['features']+[0]*(maximum-n))
        targets.append(p['state_targets']+[[0]*4]*(maximum-n))
        mask.append([True]*n+[False]*(maximum-n))
    admission=teacher.admit(family)
    forks=[];excluded=[]
    lookup={(p['history_id'],p['task_id']):i for i,p in enumerate(prefixes)}
    for task in ('task_A','task_B','task_T'):
        for suffix in ('','_N'):
            left,right=[lookup[(h+suffix,task)] for h in ('H_A','H_B')]
            choices=[admission['selected_cells'][str(p)] for p in (left,right)]
            if any(len(ids)!=1 for ids in choices):
                excluded.append(dict(task=task,stratum=suffix,reason='TIED_MINIMUM_TESTED_PASS'));continue
            il,ir=[x[0] for x in choices];found=False
            for t,(a,b) in enumerate(zip(family['cells'][il]['tail'],family['cells'][ir]['tail'])):
                if a['feature']!=b['feature']:
                    excluded.append(dict(task=task,stratum=suffix,reason='INPUT_DIFFERS_BEFORE_EXPERT_FORK'));found=True;break
                if a['target']==b['target']:continue
                if not admission['masks'][il][t] or not admission['masks'][ir][t]:
                    excluded.append(dict(task=task,stratum=suffix,reason='AMBIGUOUS_OR_DUPLICATE_OWNER'));found=True;break
                sham_suffix='' if suffix else '_N'
                forks.append(dict(left=left,right=right,sham_left=lookup[('H_A'+sham_suffix,task)],
                    sham_right=lookup[('H_B'+sham_suffix,task)],left_cell=il,right_cell=ir,step=t,
                    left_action=a['target'],right_action=b['target'],common_feature=a['feature'],task=task,stratum=suffix))
                found=True;break
            if not found:excluded.append(dict(task=task,stratum=suffix,reason='NO_EXPERT_ACTION_CONFLICT'))
    selected=[i for i,m in enumerate(admission['masks']) if any(m)]
    tail_length=max(len(family['cells'][i]['tail']) for i in selected)
    tails=[family['cells'][i] for i in selected]
    return dict(family=family,indices=torch.tensor(indices,device=device),state_targets=torch.tensor(targets,dtype=torch.float32,device=device),
        mask=torch.tensor(mask,device=device),lengths=torch.tensor(lengths,device=device),admission=admission,forks=forks,excluded=excluded,
        tail_prefix=torch.tensor([r['prefix'] for r in tails],device=device),
        tail_indices=torch.tensor([[x['feature'] for x in r['tail']]+[0]*(tail_length-len(r['tail'])) for r in tails],device=device),
        tail_targets=torch.tensor([[x['target'] for x in r['tail']]+[0]*(tail_length-len(r['tail'])) for r in tails],device=device),
        tail_mask=torch.tensor([admission['masks'][i]+[0]*(tail_length-len(family['cells'][i]['tail'])) for i in selected],device=device,dtype=torch.float32),
        tail_alive=torch.tensor([[True]*len(r['tail'])+[False]*(tail_length-len(r['tail'])) for r in tails],device=device))


def rollout_tail(net,cache,final,cell):
    memory=final.unsqueeze(0);outputs=[]
    for t,row in enumerate(cell['tail']):
        feature=cache['features'][row['feature']].unsqueeze(0)
        if t:memory,_=net.update(feature,memory)
        outputs.append(net.action_logits(memory,cache['logits'][row['feature']].unsqueeze(0),feature)[0])
    return torch.stack(outputs)


def fork_logits(net,cache,b,final,case,donor='correct'):
    selected=[case['left'],case['right']]
    if donor=='wrong':selected.reverse()
    if donor=='sham':selected=[case['sham_left'],case['sham_right']]
    memory=final[selected]
    tail=b['family']['cells'][case['left_cell']]['tail']
    for t in range(1,case['step']+1):
        current=cache['features'][tail[t]['feature']].unsqueeze(0).expand(2,-1)
        memory,_=net.update(current,memory)
    current=cache['features'][case['common_feature']].unsqueeze(0).expand(2,-1)
    native=cache['logits'][case['common_feature']].unsqueeze(0).expand(2,-1)
    return net.action_logits(memory,native,current)


def losses(net,cache,b,arm,retain_steps=()):
    device=cache['features'].device
    features=cache['features'][b['indices']]
    states,writes=net.encode(features,retain_steps=retain_steps)
    final=states[torch.arange(len(b['lengths']),device=device),b['lengths']-1]
    cells=b['family']['cells'];prefix=torch.tensor([r['prefix'] for r in cells],device=device)
    fields=('stop_at_cutoff','stops','suffix_anchor_before_final','terminal_at_final')
    query=torch.tensor([[float(bool(r['query_context'][k])) for k in fields] for r in cells],device=device)
    y=torch.tensor([r['y'] for r in cells],dtype=torch.float32,device=device)
    query_logits=net.reader(final[prefix],query)
    query_loss=F.binary_cross_entropy_with_logits(query_logits,y)
    state_logits=net.state_head(states.flatten(2))
    state_ce=F.binary_cross_entropy_with_logits(state_logits,b['state_targets'],reduction='none').mean(-1)
    state_loss=(state_ce*b['mask']).sum()/b['mask'].sum()
    memory=final[b['tail_prefix']];outputs=[]
    for t in range(b['tail_indices'].shape[1]):
        indices=b['tail_indices'][:,t];current=cache['features'][indices]
        if t:
            updated,_=net.update(current,memory)
            memory=torch.where(b['tail_alive'][:,t,None,None],updated,memory)
        outputs.append(net.action_logits(memory,cache['logits'][indices],current))
    logits=torch.stack(outputs,1);mask=b['tail_mask'];targets=b['tail_targets']
    per_action=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view_as(mask)
    owners=int(mask.sum());correct=int(((logits.argmax(-1)==targets)*mask).sum())
    action_loss=(per_action*mask).sum()/mask.sum()
    fork_terms=[];fork_correct=0
    for case in b['forks']:
        logits=fork_logits(net,cache,b,final,case);targets=torch.tensor([case['left_action'],case['right_action']],device=device)
        fork_terms.append(F.cross_entropy(logits,targets));fork_correct+=int((logits.argmax(-1)==targets).all())
    fork_loss=torch.stack(fork_terms).mean() if fork_terms else action_loss*0
    count=min(156,states.shape[1]);native=cache['logits'][b['indices'][:,:count]]
    predicted=net.action_logits(states[:,:count].flatten(0,1),native.flatten(0,1),features[:,:count].flatten(0,1))
    kl=F.kl_div(F.log_softmax(predicted,dim=-1),F.softmax(native.flatten(0,1).detach(),dim=-1),reduction='none').sum(-1).view_as(b['mask'][:,:count])
    preservation=(kl*b['mask'][:,:count]).sum()/b['mask'][:,:count].sum()
    auxiliary={'B1':action_loss*0,'B2':state_loss,'Ours':query_loss}[arm]
    stats=dict(action_ce=float(action_loss.detach()),action_correct=correct,action_owners=owners,
        fork_ce=float(fork_loss.detach()),fork_both_correct=fork_correct,fork_pairs=len(b['forks']),
        exact_state_bce=float(state_loss.detach()),crossed_bce=float(query_loss.detach()),preservation_kl=float(preservation.detach()))
    return action_loss+fork_loss+auxiliary+preservation,stats,dict(final=final,states=states,writes=writes,query_logits=query_logits,state_logits=state_logits)


def ordinary_loss(net,cache,records):
    device=cache['features'].device
    maximum=max(len(r['features']) for r in records)
    indices=torch.tensor([r['features']+[0]*(maximum-len(r['features'])) for r in records],device=device)
    targets=torch.tensor([r['targets']+[0]*(maximum-len(r['targets'])) for r in records],device=device)
    mask=torch.tensor([[True]*len(r['features'])+[False]*(maximum-len(r['features'])) for r in records],device=device)
    features=cache['features'][indices];states,_=net.encode(features)
    logits=net.action_logits(states.flatten(0,1),cache['logits'][indices].flatten(0,1),features.flatten(0,1))
    ce=F.cross_entropy(logits,targets.flatten(),reduction='none').view_as(mask)
    return ((ce*mask).sum(-1)/mask.sum(-1)).mean()


def compose_state_probability(state,cells):
    rows=[]
    for cell in cells:
        z=state[cell['prefix']];ctx=cell['query_context']
        value=z[3] if ctx['stop_at_cutoff'] else z[1]
        if not ctx['stop_at_cutoff'] and ctx['suffix_anchor_before_final']:value=torch.ones_like(value)
        if not ctx['stops'] or (not ctx['stop_at_cutoff'] and not ctx['terminal_at_final']):value=torch.zeros_like(value)
        rows.append(value)
    return torch.stack(rows)


@torch.no_grad()
def evaluate(net,cache,batches,arm):
    rows=[]
    for b in batches:
        _,stats,parts=losses(net,cache,b,arm);cases=[]
        for case in b['forks']:
            target=torch.tensor([case['left_action'],case['right_action']])
            predictions={mode:fork_logits(net,cache,b,parts['final'],case,mode).argmax(-1).tolist() for mode in ('correct','wrong','sham')}
            cases.append(dict(case=case,targets=target.tolist(),predictions=predictions,
                both_correct={k:v==target.tolist() for k,v in predictions.items()}))
        family=b['family']
        probability=(compose_state_probability(net.state_head(parts['final'].flatten(1)).sigmoid(),family['cells'])
                     if arm=='B2' else parts['query_logits'].sigmoid())
        targets=torch.tensor([cell['y'] for cell in family['cells']],device=probability.device)
        rows.append(dict(family_id=family['family_id'],house=family['house'],split=family['split'],**stats,
            fork_cases=cases,excluded=b['excluded'],query_reader_untrained=arm!='Ours',
            query_correct=int(((probability>.5)==targets.bool()).sum()) if arm!='B1' else None,
            query_cells=len(family['cells']),query_probabilities=probability.tolist() if arm!='B1' else None,
            query_rule='composed predicted exact state' if arm=='B2' else 'trained result reader' if arm=='Ours' else 'not trained'))
    return rows
