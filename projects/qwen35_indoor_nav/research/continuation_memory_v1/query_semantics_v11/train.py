"""Matched B2/Ours CPU training; only future-query semantics changes from V10."""
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(HERE))
import losses as special
c=special.c
models=special.models

def ordinary_batch(records, device):
    length = max(len(row['features']) for row in records)
    return dict(indices=torch.tensor([r['features'] + [0] * (length-len(r['features'])) for r in records], device=device),
        targets=torch.tensor([r['targets'] + [0] * (length-len(r['targets'])) for r in records], device=device),
        mask=torch.tensor([[True]*len(r['features']) + [False]*(length-len(r['features'])) for r in records], device=device))


def ordinary_loss(net, cache, batch):
    states, _ = net.encode(cache['features'][batch['indices']])
    logits = net.action_logits(states.flatten(0, 1), cache['logits'][batch['indices']].flatten(0, 1), cache['features'][batch['indices']].flatten(0, 1))
    per_step = F.cross_entropy(logits, batch['targets'].flatten(), reduction='none').view_as(batch['mask'])
    per_route = (per_step * batch['mask']).sum(-1) / batch['mask'].sum(-1)
    return per_route.mean(), logits.view(*batch['indices'].shape, 4)


@torch.no_grad()
def ordinary_evaluate(net, cache, records):
    rows = []
    for start in range(0, len(records), 8):
        selected = records[start:start+8]
        batch = ordinary_batch(selected, cache['features'].device)
        _, logits = ordinary_loss(net, cache, batch)
        for i, row in enumerate(selected):
            n = len(row['features'])
            target = batch['targets'][i, :n]
            base = cache['logits'][batch['indices'][i, :n]]
            current = logits[i, :n]
            rows.append(dict(record_id=row['row']['record_id'], house=row['row']['scene_group'],
                partition=row['partition'], decisions=n,
                ce=float(F.cross_entropy(current, target)), base_ce=float(F.cross_entropy(base, target)),
                correct=int((current.argmax(-1)==target).sum()), base_correct=int((base.argmax(-1)==target).sum()),
                native_action_changes=int((current.argmax(-1)!=base.argmax(-1)).sum()),
                terminal_stop_correct=bool(current[-1].argmax()==3), base_terminal_stop_correct=bool(base[-1].argmax()==3),
                premature_stops=int((current[:-1].argmax(-1)==3).sum()), base_premature_stops=int((base[:-1].argmax(-1)==3).sum())))
    summaries = {}
    for partition in ('fit', 'check'):
        selected = [r for r in rows if r['partition']==partition]
        decisions = sum(r['decisions'] for r in selected)
        summaries[partition] = dict(routes=len(selected), decisions=decisions,
            houses=sorted({r['house'] for r in selected}),
            macro_ce=sum(r['ce'] for r in selected)/len(selected),
            base_macro_ce=sum(r['base_ce'] for r in selected)/len(selected),
            accuracy=sum(r['correct'] for r in selected)/decisions,
            base_accuracy=sum(r['base_correct'] for r in selected)/decisions,
            terminal_stop_correct=sum(r['terminal_stop_correct'] for r in selected),
            base_terminal_stop_correct=sum(r['base_terminal_stop_correct'] for r in selected),
            premature_stops=sum(r['premature_stops'] for r in selected),
            base_premature_stops=sum(r['base_premature_stops'] for r in selected))
    return dict(rows=rows, summaries=summaries, scope='Teacher trajectories, not closed-loop success; current actions/labels are absent from feature input')


def gradient_audit(net, cache, batch, mode):
    prefixes=batch['family']['prefixes']
    index=next(i for i,p in enumerate(prefixes) if p['history_id']=='H_B' and p['task_id']=='task_B')
    critical=next(t for t,state in enumerate(prefixes[index]['state_targets']) if state[1])
    length=len(prefixes[index]['features'])
    assert length-1-critical>8
    features=cache['features'][batch['prefix_indices']]
    states,writes=net.encode(features,retain_steps=[critical])
    final=states[:,-1]
    action=net.action_logits(final,cache['logits'][batch['prefix_indices'][:,-1]],features[:,-1])
    grad=torch.autograd.grad(action.square().sum(),final,retain_graph=True,allow_unused=True)[0]
    action_norm=float(grad.norm()) if grad is not None else 0.
    auxiliary_norm=None
    if mode in ('B2','Ours'):
        if mode=='Ours':
            logits=special.reader_logits(net,final,batch)
            loss=(F.binary_cross_entropy_with_logits(logits,batch['y'].float(),reduction='none')*batch['masks']).sum()/batch['masks'].sum()
        else:
            logits=net.state_head(final.flatten(1))
            loss=F.binary_cross_entropy_with_logits(logits,batch['state_targets'][:,-1].float())
        gradient=torch.autograd.grad(loss,writes[critical])[0][index]
        assert bool(torch.isfinite(gradient).all()) and float(gradient.norm())>0
        auxiliary_norm=float(gradient.norm())
    assert action_norm>0 if mode!='N0' else action_norm==0
    return dict(family_id=batch['family']['family_id'],critical_step=critical,supervision_step=length-1,
        critical_writer_auxiliary_gradient_norm=auxiliary_norm,action_memory_gradient_norm=action_norm,
        after_optimizer_updates=2,encoder_trained=False,full_prefix_bptt=True,per_step_memory_detach=False,
        no_memory_control=mode=='N0')


def train(run,config,special_cache,special_data,natural_cache,natural_data):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    special_cache={k:v.float() for k,v in special_cache.items()}
    natural_cache={k:v.float() for k,v in natural_cache.items()}
    batches=[special.tensors(f,'cpu') for f in special_data['families']]
    result={}
    for seed in config['seeds']:
        torch.manual_seed(seed)
        net=models.MemoryPolicy(2048,len(special_data['query_vocabulary']),8,64,.99).to('cpu')
        initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        prior=torch.load(HERE.parent/'contextual_readout_v10/run_001'/f'INITIAL_{seed}.pt',map_location='cpu',weights_only=True)
        for name,value in initial.items():
            if not name.startswith('result_head.'):
                assert torch.equal(value,prior[name]),'RUNTIME_INITIALIZATION_CHANGED:'+name
        initial_sha=c.model_identity(net)['sha256']
        torch.save(initial,run/f'INITIAL_{seed}.pt')
        schedule=c.read(HERE.parent/'natural_transfer_v9/run_001'/f'SCHEDULE_{seed}.json')
        assert len(schedule['family_indices'])==len(schedule['ordinary_indices'])==config['steps_per_arm']
        c.write(run/f'SCHEDULE_{seed}.json',dict(family_indices=schedule['family_indices'],ordinary_indices=schedule['ordinary_indices'],
            initial_state_sha256=initial_sha,source_schedule_sha256=c.sha(HERE.parent/'natural_transfer_v9/run_001'/f'SCHEDULE_{seed}.json')),True)
        for arm in config['arms']:
            net.load_state_dict(initial);net.no_memory=arm=='N0'
            assert c.model_identity(net)['sha256']==initial_sha
            optimizer=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            began=time.monotonic()
            for step,(family_index,ordinary_indices) in enumerate(zip(schedule['family_indices'],schedule['ordinary_indices']),1):
                net.zero_grad(set_to_none=True)
                special_loss,stats,_=special.losses(net,special_cache,batches[family_index],arm)
                batch=ordinary_batch([natural_data['records'][i] for i in ordinary_indices],'cpu')
                natural_loss,_=ordinary_loss(net,natural_cache,batch)
                loss=special_loss+natural_loss
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
                optimizer.step()
                c.append(run/f'{arm}_{seed}_STEPS.jsonl',dict(step=step,family_index=family_index,ordinary_indices=ordinary_indices,
                    loss=float(loss.detach()),ordinary_ce=float(natural_loss.detach()),**stats))
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='TRAIN',seed=seed,arm=arm,step=step,total=config['steps_per_arm']))
                if step==2:
                    c.write(run/f'{arm}_{seed}_GRADIENT.json',gradient_audit(net,special_cache,batches[schedule['family_indices'][0]],arm),True)
            measured=special.evaluate(net,special_cache,batches,arm)
            measured['ordinary']=ordinary_evaluate(net,natural_cache,natural_data['records'])
            final={k:v.detach().cpu() for k,v in net.state_dict().items()}
            changes={name:bool(not torch.equal(initial[name],value)) for name,value in final.items()}
            assert changes['action.0.weight'] and changes['action.2.weight']
            assert changes['writer.weight']==(arm!='N0') and changes['recurrent.weight']==(arm!='N0')
            final_sha=c.model_identity(net)['sha256']
            torch.save(final,run/f'{arm}_{seed}_MEMORY.pt')
            measured.update(seed=seed,arm=arm,updates=config['steps_per_arm'],initial_state_sha256=initial_sha,
                final_state_sha256=final_sha,parameter_changed=changes,seconds=time.monotonic()-began,no_memory_control=arm=='N0')
            c.write(run/f'{arm}_{seed}_RESULT.json',measured,True)
            result[f'{arm}_{seed}']=measured
            del optimizer,loss,special_loss,natural_loss,batch

        del net
    c.write(run/'RESULT.json',dict(status='MATCHED_STRUCTURED_QUERY_COMPLETE',runs=result,
        base_encoder_updates=0,new_qwen_forwards=0,optimizer_updates=len(config['seeds'])*len(config['arms'])*config['steps_per_arm'],
        original_special_training_admission=False,ordinary_labels_unchanged=True,selected_by_final_check_results=False,
        independent_blind_test=False,closed_loop_memory_gain_tested=False),True)
