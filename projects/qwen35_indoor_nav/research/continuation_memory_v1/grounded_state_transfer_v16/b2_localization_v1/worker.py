"""Fixed FIT-only probes plus read-only teacher-path diagnosis of trained policies."""
import argparse
from collections import defaultdict
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import torch
from torch import nn
from torch.nn import functional as F


def progress(seed,phase,**fields):
    write(OUT/f'PROGRESS_{seed}.json',dict(status='RUNNING',phase=phase,seed=seed,unix=time.time(),**fields))


def probe(x,rows,y,kind,seed,name,cfg):
    train=[i for i,r in enumerate(rows) if r['split']=='FIT'];ids=torch.tensor(train)
    weights=fit_weights([rows[i] for i in train],y[ids]);x=F.layer_norm(x,(x.shape[-1],))
    torch.manual_seed(seed)
    net=nn.Linear(x.shape[-1],y.shape[-1]) if kind=='linear' else nn.Sequential(nn.Linear(x.shape[-1],128),nn.Tanh(),nn.Linear(128,y.shape[-1]))
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    gen=torch.Generator().manual_seed(seed);initial=c.model_identity(net)['sha256'];logs=[]
    for step in range(cfg['probe_steps']):
        sel=torch.randint(len(ids),(cfg['batch_size'],),generator=gen);batch=ids[sel]
        optimizer.zero_grad(set_to_none=True);logits=net(x[batch]);w=weights[sel]
        loss=(F.binary_cross_entropy_with_logits(logits,y[batch],reduction='none')*w).sum()/w.sum()
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('NONFINITE_PROBE')
        optimizer.step()
        if (step+1)%50==0:logs.append(dict(step=step+1,loss=float(loss.detach())));progress(seed,name,step=step+1)
    with torch.inference_mode():prediction=torch.cat([net(z).sigmoid() for z in x.split(1024)]).tolist()
    folder=OUT/f'seed_{seed}';atomic_torch(folder/(name+'.pt'),net.state_dict())
    immutable(folder/(name+'_TRAIN.json'),dict(steps=cfg['probe_steps'],fit_samples=len(train),dev_used_in_updates=False,initial_sha256=initial,final_sha256=c.model_identity(net)['sha256'],losses=logs))
    return prediction


def binary_tables(rows,labels,probabilities,names):
    tables=[]
    for split in ('FIT','DEV'):
        selected=[i for i,r in enumerate(rows) if r['split']==split]
        for house in [None]+sorted({rows[i]['house'] for i in selected}):
            ids=[i for i in selected if house is None or rows[i]['house']==house]
            for b,name in enumerate(names):
                tables.append(dict(split=split,house=house,target=name,**metrics([labels[i][b] for i in ids],[probabilities[i][b] for i in ids])))
    return tables


def gradient_diagnosis(net,data,cfg):
    houses=sorted({f['house'] for f in data['families'] if f['split']=='FIT'});chosen=[]
    for house in houses:
        variants=[f for f in data['families'] if f['house']==house and f['stratum']=='terminal_present']
        chosen+=sorted(variants,key=lambda f:digest(f['parent_family_id']))[:cfg['gradient_families_per_house']]
    weights=o.class_weights([f for f in data['families'] if f['split']=='FIT']);cache=torch.load(RUN/'features/FEATURES.pt',weights_only=True)
    params=list(net.writer.parameters())+list(net.recurrent.parameters());rows=[]
    for f in chosen:
        b=o.batch(f,'cpu');_,_,detail=o.losses(net,cache,b,'B2',weights)
        logits=detail['logits'];native=cache['logits'][b['indices']]
        kl=o.average(F.kl_div(logits.log_softmax(-1),native.softmax(-1),reduction='none').sum(-1),b['preservation_mask'])
        terms=dict(action=detail['action_loss'],auxiliary=detail['auxiliary_loss'],preservation=kl);grads={}
        for name,loss in terms.items():
            g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            grads[name]=torch.cat([(torch.zeros_like(p) if x is None else x).flatten() for p,x in zip(params,g)])
        comparisons={}
        for left,right in (('action','auxiliary'),('action','preservation'),('auxiliary','preservation')):
            a,z=grads[left],grads[right];den=a.norm()*z.norm()
            comparisons[left+'_vs_'+right]=float(torch.dot(a,z)/den) if den else None
        rows.append(dict(family=f['family_id'],house=f['house'],norms={k:float(v.norm()) for k,v in grads.items()},cosines=comparisons))
    return dict(rows=rows,scope='Fixed two FIT parent families per house; shared writer/recurrent parameters; gradients only, zero updates. Does not include ordinary BC or prove a counterfactual repair benefit.')


def main(seed):
    cfg=read(HERE/'PROTOCOL.json');torch.set_num_threads(cfg['threads_per_worker']);torch.use_deterministic_algorithms(True)
    folder=OUT/f'seed_{seed}';folder.mkdir();started=time.monotonic()
    binding=read(OUT/'BINDING.json')
    for path in [RUN/'DATA.json',RUN/'features/FEATURES.pt']:
        if sha(path)!=binding['sources'][str(path)]:raise ValueError('INPUT_CHANGED')
    diag=read(OUT/'DATA.json');data=read(RUN/'DATA.json');families={f['family_id']:f for f in data['families']};cache=torch.load(RUN/'features/FEATURES.pt',weights_only=True)
    events=diag['events'];ids=torch.tensor([r['index'] for r in events]);labels=torch.tensor([r['event'] for r in events],dtype=torch.float32);event_results={};event_predictions={}
    for kind in cfg['probe_kinds']:
        name='event_'+kind;prediction=probe(cache['features'][ids].float(),events,labels,kind,seed,name,cfg)
        tables=binary_tables(events,labels.tolist(),prediction,['current_anchor_SEE2','current_terminal_SEE2'])
        scales=[]
        for b,target in enumerate(('anchor','terminal')):
            for bucket in ('absent','subthreshold','near_threshold','clear'):
                q=[i for i,r in enumerate(events) if r['split']=='DEV' and ('absent' if r['scale'][b]==0 else 'subthreshold' if r['scale'][b]<256 else 'near_threshold' if r['scale'][b]<512 else 'clear')==bucket]
                scales.append(dict(target=target,bucket=bucket,**metrics([events[i]['event'][b] for i in q],[prediction[i][b] for i in q])))
        event_results[kind]=dict(tables=tables,DEV_scale=scales);event_predictions[kind]=prediction
    immutable(folder/'EVENT_PROBE.json',event_results);immutable(folder/'EVENT_PREDICTIONS.json',event_predictions)
    visits=diag['visits'];memory_results={};model_seals={}
    for arm in ('B1','B2','Terminal'):
        progress(seed,'policy_readback',arm=arm)
        tag=f'{arm}_{seed}';path=RUN/'train'/tag/'FINAL.pt'
        if sha(path)!=binding['sources'][str(path)]:raise ValueError('HEAD_CHANGED')
        net=o.initialize(seed);net.load_state_dict(torch.load(path,weights_only=True));net.eval();initial=c.model_identity(net)['sha256']
        memory=torch.empty(len(visits),512);records=[None]*len(visits)
        with torch.inference_mode():
            for si,sequence in enumerate(diag['sequences']):
                if not sequence['selected']:continue
                family=families[sequence['family']];row=family['sequences'][sequence['sequence']];indices=torch.tensor(row['features'])
                x=cache['features'][indices].float().unsqueeze(0);native=cache['logits'][indices].float()
                states,_=net.encode(x);states=states[0];z=net.state_head(states.flatten(1)).sigmoid()
                logits=net.action_logits(states,native,x[0]);pred=logits.argmax(-1)
                for t,index in sequence['selected']:
                    v=visits[index];memory[index]=states[t].flatten()
                    records[index]=dict(row_id=index,prediction=int(pred[t]),native=int(native[t].argmax()),probability=z[t].tolist(),
                        teacher_agreement=int(pred[t]) in v['teacher_targets'],ready_stop=int(pred[t])==3,state_correct=bool(torch.equal((z[t]>=.5).int(),torch.tensor(v['state']))))
                if si%80==0:progress(seed,'policy_readback',arm=arm,sequences=si,total=len(diag['sequences']))
        if any(r is None for r in records):raise ValueError('MISSING_STATE_ROW')
        immutable(folder/(arm+'_POLICY_ROWS.json'),records)
        if arm in cfg['memory_probes']:
            selected=[i for i,v in enumerate(visits) if v['task']=='task_A'];rows=[visits[i] for i in selected]
            y=torch.tensor([[v['state'][1]] for v in rows],dtype=torch.float32)
            pr=probe(memory[selected],rows,y,'linear',seed,'memory_'+arm,cfg)
            memory_results[arm]=dict(tables=binary_tables(rows,y.tolist(),pr,['anchor_seen']),by_age=[])
            for split in ('FIT','DEV'):
                for age in ('never','current','1-8','9-16','17+'):
                    q=[i for i,v in enumerate(rows) if v['split']==split and age_bucket(v['age'])==age]
                    memory_results[arm]['by_age'].append(dict(split=split,age=age,**metrics([rows[i]['state'][1] for i in q],[pr[i][0] for i in q])))
            immutable(folder/(arm+'_MEMORY_PREDICTIONS.json'),[dict(row_id=selected[i],probability=x[0]) for i,x in enumerate(pr)])
        if arm=='B2':immutable(folder/'GRADIENT_CONFLICT.json',gradient_diagnosis(net,data,cfg))
        final=c.model_identity(net)['sha256']
        if initial!=final:raise ValueError('POLICY_PARAMETER_CHANGED')
        model_seals[tag]=dict(before=initial,after=final,unchanged=True)
        del net,memory
    immutable(folder/'MEMORY_PROBE.json',memory_results)
    immutable(folder/'RESULT.json',dict(status='COMPLETE_DIAGNOSTIC',seed=seed,model_seals=model_seals,policy_updates=0,base_updates=0,new_episodes=0,seconds=time.monotonic()-started))
    write(OUT/f'PROGRESS_{seed}.json',dict(status='COMPLETE',seed=seed,unix=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);main(parser.parse_args().seed)
