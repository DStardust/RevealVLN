"""Real frozen-Qwen features, recurrent long BPTT and matched debug B2/Ours updates."""
import gc
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
V5=LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'
sys.path.insert(0,str(V5))
import common as c
sys.path.insert(0,str(HERE))
from data import prepare
from model import MemoryPolicy


def features(run,config):
    from evaluate import load_policy
    import torch
    from PIL import Image
    loader,records,prepared=prepare()
    recorded=c.read(HERE/'DATA.json')
    assert prepared==recorded,'DEBUG_ASSET_CHANGED'
    p=c.read(run/'CONFIG.json')
    model,policy,initial=load_policy(run,p)
    engineering_initial=next((V5/'sessions').glob('session_*/STATE_INITIAL.json'))
    assert initial['sha256']==c.read(engineering_initial)['sha256'],'PILOT_BASE_DIFFERS_FROM_ENGINEERING'
    for parameter in policy.parameters(): parameter.requires_grad_(False)
    collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
    class Store:
        def get(self,i,t):
            value=loader.policy_payload(records[i])
            return dict(instruction=value['instruction'],images=[Image.frombytes('RGB',(224,224),x) for x in value['rgb']],
                        executed=[a.lower() for a in value['executed_actions']])
    dataset=model.DecisionDataset([dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(records))],Store(),policy.processor)
    capture=[]
    def hook(module,args): capture.append(args[0].detach().cpu().clone())
    handle=policy.action_head.register_forward_pre_hook(hook)
    feature_rows=[];logits_rows=[]
    try:
        with torch.inference_mode():
            for i in range(len(records)):
                batch=collate([dataset[i]]);batch.pop('targets');batch.pop('weights')
                hashes={k:c.tensor_identity(v) for k,v in batch.items()}
                batch={k:v.to('cuda:0') for k,v in batch.items()}
                logits=policy.forward_batch(**batch)
                torch.cuda.synchronize()
                assert len(capture)==1
                feature_rows.append(capture.pop())
                logits_rows.append(logits.detach().float().cpu())
                c.append(run/'FEATURE_INPUTS.jsonl',dict(index=i,binding=prepared['audit']['feature_bindings'][i],processed=hashes))
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='CAUSAL_FEATURE_EXTRACTION',completed=i+1,total=len(records)))
        final=c.model_identity(policy)
        assert final['sha256']==initial['sha256'],'FROZEN_ENCODER_CHANGED'
        cache=dict(features=torch.cat(feature_rows),logits=torch.cat(logits_rows))
        torch.save(cache,run/'FEATURES.pt')
        c.write(run/'FEATURE_RESULT.json',dict(real_model_forward_count=len(records),feature_shape=list(cache['features'].shape),
            features_sha256=c.sha(run/'FEATURES.pt'),encoder_state_unchanged=True,initial_sha256=initial['sha256'],
            final_sha256=final['sha256'],future_query_in_encoder=False,base_parameter_updates=0),True)
    finally: handle.remove()
    del policy,model,initial,final,loader,dataset
    gc.collect();torch.cuda.empty_cache()
    return cache,prepared


def losses(net,cache,data,mode,retain=False):
    import torch
    import torch.nn.functional as F
    prefix_features=torch.stack([cache['features'][p['features']] for p in data['prefixes']])
    states,writes=net.encode(prefix_features,retain_steps=[14,17,237] if retain else [])
    final=states[:,-1]
    q_logits=[]
    for cell in data['cells']:
        query=torch.tensor([data['queries'][cell['query']]],device=final.device)
        q_logits.append(net.reader(final[cell['prefix']:cell['prefix']+1],query)[0])
    query_logits=torch.stack(q_logits)
    y=torch.tensor([cell['y'] for cell in data['cells']],device=final.device,dtype=torch.float32)
    masks=torch.tensor([cell['mask'] for cell in data['cells']],device=final.device,dtype=torch.float32)
    result_loss=(F.binary_cross_entropy_with_logits(query_logits,y,reduction='none')*masks).sum()/masks.sum()
    target=torch.tensor([p['state_targets'] for p in data['prefixes']],device=final.device,dtype=torch.float32)
    state_masks=torch.tensor([p['state_masks'] for p in data['prefixes']],device=final.device,dtype=torch.float32)
    state_logits=net.state_head(states.flatten(2))
    state_loss=(F.binary_cross_entropy_with_logits(state_logits,target,reduction='none').mean(-1)*state_masks).sum()/state_masks.sum()
    # Strong B2 diagnostic: apply the exact checker to predicted state plus the legal
    # query. Do not rank B2 by its untrained crossed-result neural head.
    state_query=[]
    for cell in data['cells']:
        context=cell['query_context'];z=state_logits[cell['prefix'],-1].sigmoid()
        probability=z[3] if context['stop_at_cutoff'] else z[1]
        if not context['stop_at_cutoff'] and context['suffix_anchor_before_final']: probability=torch.ones_like(probability)
        if not context['stops'] or (not context['stop_at_cutoff'] and not context['terminal_at_final']): probability=torch.zeros_like(probability)
        state_query.append(probability)
    state_query=torch.stack(state_query)
    actions=[];targets=[]
    for cell in data['cells']:
        if not cell['tail']: continue
        memory=final[cell['prefix']:cell['prefix']+1]
        for offset,entry in enumerate(cell['tail']):
            feature=cache['features'][entry['feature']:entry['feature']+1]
            if offset:
                memory,_=net.update(feature,memory)
            if entry['mask']:
                actions.append(net.action_logits(memory,cache['logits'][entry['feature']:entry['feature']+1])[0])
                targets.append(entry['target'])
    action_logits=torch.stack(actions)
    targets=torch.tensor(targets,device=final.device)
    action_loss=F.cross_entropy(action_logits,targets)
    aux=state_loss if mode=='B2' else result_loss
    stats=dict(action_ce=float(action_loss.detach()),exact_state_bce=float(state_loss.detach()),
        crossed_result_bce=float(result_loss.detach()),action_accuracy=float((action_logits.argmax(-1)==targets).float().mean()),
        query_accuracy=float((((query_logits>0)==y.bool()).float()*masks).sum()/masks.sum()),action_owners=len(targets),
        state_derived_query_accuracy=float((((state_query>.5)==y.bool()).float()*masks).sum()/masks.sum()),
        effective_query_accuracy=float((((state_query>.5 if mode=='B2' else query_logits>0)==y.bool()).float()*masks).sum()/masks.sum()),
        query_cells=int(masks.sum()),state_known_steps=int(state_masks.sum()))
    return action_loss+aux,stats,dict(states=states,writes=writes,action_loss=action_loss,result_loss=result_loss,
        state_loss=state_loss,final=final,query_logits=query_logits)


def train(run,config,cache,data):
    import torch
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    cache={k:v.to('cuda:0') for k,v in cache.items()}
    vocab=max(x for q in data['queries'] for x in q)+1
    net=MemoryPolicy(cache['features'].shape[1],vocab,config['memory_slots'],config['slot_width'],config['retention']).cuda()
    initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    torch.save(initial,run/'INITIAL_MEMORY.pt')
    initial_sha=c.model_identity(net)['sha256']
    # Real critical-event write VJP, not merely a later shared-parameter gradient.
    loss,before,debug=losses(net,cache,data,'Ours',retain=True)
    hb=next(i for i,p in enumerate(data['prefixes']) if p['history_id']=='H_B' and p['task_id']=='task_B')
    debug['result_loss'].backward(retain_graph=True)
    event_grad=float(debug['writes'][14].grad[hb].norm())
    assert event_grad>0 and torch.isfinite(debug['writes'][14].grad).all(),'NO_CRITICAL_EVENT_GRADIENT'
    action_memory_grad=torch.autograd.grad(debug['action_loss'],debug['final'],retain_graph=True)[0]
    assert float(action_memory_grad.norm())>0,'ACTION_DOES_NOT_READ_MEMORY'
    memory=debug['final'].detach()
    base=cache['logits'][data['prefixes'][0]['features'][-1]].unsqueeze(0)
    with torch.no_grad():
        changed=float((net.action_logits(memory[:1],base)-net.action_logits(torch.zeros_like(memory[:1]),base)).abs().max())
        saved=memory.clone()
        for q in data['queries']: net.reader(memory[:1],torch.tensor([q],device='cuda:0'))
        assert torch.equal(saved,memory),'QUERY_MUTATED_RUNTIME_MEMORY'
    assert changed>0
    # One real update is a separate diagnostic; both comparison arms restart from initial.
    net.zero_grad(set_to_none=True)
    loss.backward()
    opt=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    opt.step()
    updated=c.model_identity(net)['sha256']
    assert updated!=initial_sha,'NO_PARAMETER_UPDATE'
    c.write(run/'GRADIENT_UPDATE_RESULT.json',dict(real_features=True,full_prefix_observations=249,
        critical_event=dict(history='H_B',task='task_B',step=14,supervision_step=248,writer_output_gradient_norm=event_grad),
        action_memory_gradient_norm=float(action_memory_grad.norm()),action_memory_effect_max_abs=changed,
        future_query_mutates_memory=False,initial_state_sha256=initial_sha,updated_state_sha256=updated,
        diagnostic_optimizer_updates=1,encoder_trained=False,closed_loop_memory_gain_tested=False),True)
    del loss,debug,action_memory_grad,memory,opt
    result={}
    for arm in config['arms']:
        net.load_state_dict(initial)
        assert c.model_identity(net)['sha256']==initial_sha
        opt=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        started=time.monotonic()
        with torch.no_grad(): _,initial_metrics,_=losses(net,cache,data,arm)
        for step in range(config['steps_per_arm']):
            net.zero_grad(set_to_none=True)
            loss,metrics,_=losses(net,cache,data,arm)
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
            opt.step()
            c.append(run/f'{arm}_STEPS.jsonl',dict(step=step+1,loss=float(loss.detach()),**metrics))
            c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='MEMORY_UPDATES',arm=arm,step=step+1,total=config['steps_per_arm']))
        with torch.no_grad(): _,after,_=losses(net,cache,data,arm)
        torch.save(net.state_dict(),run/f'{arm}_MEMORY.pt')
        result[arm]=dict(initial=initial_metrics,final=after,updates=config['steps_per_arm'],
            initial_state_sha256=initial_sha,final_state_sha256=c.model_identity(net)['sha256'],seconds=time.monotonic()-started)
    c.write(run/'RESULT.json',dict(status='DEBUG_FORWARD_BACKWARD_UPDATES_COMPLETE',arms=result,
        gradient_update=c.read(run/'GRADIENT_UPDATE_RESULT.json'),shared_data_sha256=c.sha(HERE/'DATA.json'),
        shared_feature_sha256=c.sha(run/'FEATURES.pt'),parameters=sum(p.numel() for p in net.parameters()),
        diagnostic_updates=1,comparison_updates=2*config['steps_per_arm'],seeds=1,
        independent_test=False,closed_loop_memory_evaluation=False,scientific_superiority='UNTESTED',
        scope='Single exposed SEE2 family; auxiliary fitting and implementation diagnostic only; neither B2 superiority nor Ours superiority established'),True)


def main():
    run=Path(sys.argv[1]);config=c.read(HERE/'CONFIG.json')
    try:
        cache,data=features(run,config)
        train(run,config,cache,data)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise


if __name__=='__main__':main()
