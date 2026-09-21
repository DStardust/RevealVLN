"""Fixed final-head diagnostics, memory interventions and FIT-only shortcut lookup."""
from collections import Counter, defaultdict
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
import objective as o

def main(run):
    import torch
    cfg=config(run);data=read(run/'DATA.json');cache={k:v.float().cuda() for k,v in torch.load(run/'features/FEATURES.pt',weights_only=True).items()}
    allrows=[r for f in data['families'] for r in f['sequences']]
    # Full causal action sequence only. No DEV labels fit the lookup.
    raw={f['family_id']:f for f in data['raw_families']};patterns=defaultdict(Counter)
    for f in raw.values():
        if f['split']=='FIT':
            for h,actions in f['histories'].items():patterns[tuple(actions)][int(h.startswith('seen'))]+=1
    shortcut=[]
    for f in raw.values():
        if f['split']=='DEV':
            for h,actions in f['histories'].items():
                dist=patterns.get(tuple(actions));prediction=max((0,1),key=lambda z:(dist[z],-z)) if dist else 0
                shortcut.append(dict(family=f['family_id'],history=h,truth=int(h.startswith('seen')),prediction=prediction,seen_in_FIT=bool(dist)))
    immutable(run/'ACTION_ONLY_LOOKUP.json',dict(rows=shortcut,correct=sum(x['prediction']==x['truth'] for x in shortcut),total=len(shortcut),
        unseen_FIT_patterns=sum(not x['seen_in_FIT'] for x in shortcut),fallback='predict missing; fixed before DEV results',
        scope='Nuisance lower bound: exact causal action lookup, not matched-capacity learned ablation'))
    summaries=[];interventions=[];events=[]
    with torch.inference_mode():
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                tag=f'{arm}_{seed}';net=o.initialize(seed).cuda().eval();net.load_state_dict(torch.load(run/'train'/tag/'FINAL.pt',weights_only=True))
                for family in data['families']:
                    b=o.batch(family,'cuda');x=cache['features'][b['indices']];native=cache['logits'][b['indices']]
                    states,_=o.encode_masked(net,x,b['alive']);z=net.state_head(states.flatten(2))
                    logits=net.action_logits(states.flatten(0,1),native.flatten(0,1),x.flatten(0,1)).view(*b['indices'].shape,4)
                    q=net.reader(states.flatten(0,1),b['query'].flatten(0,1)).view_as(b['y'])
                    probability=o.compose_state_probability(z.sigmoid(),b['query']) if arm=='B2' else q.sigmoid() if arm=='Ours' else None
                    seq=family['sequences'];lookup={(r['history'],r['task'],r['continuation']):i for i,r in enumerate(seq)}
                    for i,r in enumerate(seq):
                        t=r['cutoff'];eligible=bool(r['action_masks'][t]);prediction=int(logits[i,t].argmax())
                        if eligible:
                            summaries.append(dict(model=tag,arm=arm,seed=seed,split=family['split'],family=family['family_id'],task=r['task'],history=r['history'],
                                target=r['targets'][t],prediction=prediction,stop_correct=(prediction==3)==(r['targets'][t]==3),exact_action_correct=prediction==r['targets'][t],
                                state_correct=None if arm!='B2' else bool(torch.equal((z[i,t]>0).float(),b['state'][i,t]))))
                        if probability is not None:
                            events.append(dict(model=tag,arm=arm,split=family['split'],family=family['family_id'],history=r['history'],task=r['task'],query=r['continuation'],
                                label=r['y'][t],probability=float(probability[i,t]),correct=bool((probability[i,t]>.5)==b['y'][i,t])))
                    if family['split']!='DEV':continue
                    for h,other,sham in (('seen','missing','seen_sham'),('missing','seen','missing_sham')):
                        for task in ('task_A','task_T'):
                            def teacher(history):return 'direct_stop' if history.startswith('seen') or task=='task_T' else 'acquire_anchor'
                            i=lookup[(h,task,teacher(h))];j=lookup[(other,task,teacher(other))];k=lookup[(sham,task,teacher(sham))];t=seq[i]['cutoff']
                            if len({seq[n]['features'][t] for n in (i,j,k)})!=1:raise ValueError('MECHANISM_INPUT_CHANGED')
                            memory=states[i,t].unsqueeze(0);n=native[i,t].unsqueeze(0);feature=x[i,t].unsqueeze(0)
                            predictions={}
                            for mode,m in dict(correct=memory,wrong=states[j,t].unsqueeze(0),sham=states[k,t].unsqueeze(0),zero=torch.zeros_like(memory)).items():
                                values=net.action_logits(m,n,feature)[0];predictions[mode]=dict(action=int(values.argmax()),logits=values.cpu().tolist())
                            interventions.append(dict(model=tag,arm=arm,seed=seed,family=family['family_id'],task=task,history=h,target=seq[i]['targets'][t],predictions=predictions))
                del net
    immutable(run/'TEACHER_DIAGNOSTICS.json',dict(rows=summaries,query_rows=events,scope='Final1200, forced true history; not autonomous success'))
    immutable(run/'MEMORY_INTERVENTIONS.json',dict(rows=interventions,scope='Read-only action interventions at certified matching input; no oracle environment takeover'))
    immutable(run/'DIAGNOSTICS_COMPLETE.json',dict(models=9,teacher_rows=len(summaries),intervention_rows=len(interventions)))

if __name__=='__main__':main(Path(sys.argv[1]))
