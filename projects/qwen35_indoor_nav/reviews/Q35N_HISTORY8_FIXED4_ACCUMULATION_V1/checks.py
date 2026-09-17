import gc
import itertools
import json
import time
def run(model,policy,collate,forward,c,fingerprint,initial_sha):
    import torch
    pair=c.LINE/'sft_acceptance/ordinary_history8_paired_train_v1'
    old=c.load('original32_fixed4',c.LINE/'reviews/Q35N_STOP_FEATURE_PREFIX_DIAGNOSTIC_V2/checks.py').check(model,policy,collate,forward,c,'entry')
    assert old['max_abs_old']<=1e-5 and old['original_argmax_same']
    runtime=c.load('pair_runtime_fixed4',pair/'runtime.py')
    policy.exec_embed.float();policy.action_query=torch.nn.Parameter(policy.action_query.detach().float())
    assert runtime.sha(runtime.BRIDGE)==runtime.BRIDGE_SHA
    state=torch.load(runtime.BRIDGE,map_location='cpu',weights_only=True);model.load_trainable(policy,state['trainable'])
    for name,p in policy.named_parameters():
        if p.requires_grad:assert p.dtype==torch.float32 and torch.equal(p.detach().cpu(),state['trainable'][name])
    initial=runtime.fingerprint(policy)
    with (pair/'SELECTED_SAMPLES.jsonl').open() as stream:chosen=[json.loads(x) for x in itertools.islice(stream,96)]
    samples=[dict(record_idx=x['entry'][0],t=x['entry'][1],target=x['entry'][2],weight=x['entry'][3]) for x in chosen]
    weight_sum=sum(float(torch.tensor(x['weight'],dtype=torch.float32)) for x in samples)
    data=runtime.ordinary();rows,_=data.load_rows()
    loss_fn=c.load('paired_loss_fixed4',pair/'loss.py').local_micro_loss
    results={}
    for mode,store in runtime.stores(rows).items():
        dataset=model.DecisionDataset(samples,store,policy.processor)
        items=[dataset[i] for i in range(32)]
        values=[];reference=None;timing=[];trainable=[p for p in policy.parameters() if p.requires_grad]
        for method in ('independent_then_sum','inplace_accumulate'):
            policy.train();policy.zero_grad(set_to_none=True)
            logits=[];began=time.perf_counter()
            for start in range(0,32,4):
                if method=='independent_then_sum':policy.zero_grad(set_to_none=True)
                batch=collate(items[start:start+4]);y=batch.pop('targets').to('cuda:0');w=batch.pop('weights').to('cuda:0')
                out=policy.forward_batch(**{k:v.to('cuda:0') for k,v in batch.items()})
                loss_fn(out,y,w,weight_sum,3).backward();logits.append(out.detach().cpu())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
                if method=='independent_then_sum':
                    grad=torch.cat([p.grad.detach().float().reshape(-1).cpu() for p in trainable]).double()
                    reference=grad if reference is None else reference+grad
                del out,batch,y,w
            torch.cuda.synchronize();timing.append(time.perf_counter()-began);values.append(torch.cat(logits))
        actual=torch.cat([p.grad.detach().float().reshape(-1).cpu() for p in trainable]).double()
        relative=float((actual-reference).norm()/reference.norm().clamp_min(1e-12))
        cosine=float(torch.nn.functional.cosine_similarity(actual,reference,dim=0))
        difference=float((values[0]-values[1]).abs().max())
        action_match=bool(torch.equal(values[0].argmax(1),values[1].argmax(1)))
        passed=relative<=1e-5 and cosine>=.999999 and difference<=1e-5 and action_match
        results[mode]=dict(passed=passed,gradient_relative_l2=relative,gradient_cosine=cosine,
            logits_max_abs=difference,argmax_equal=action_match,wall_seconds=timing,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        c.write(c.HERE/'PROGRESS.json',dict(unix=time.time(),results=results),False)
        policy.zero_grad(set_to_none=True);policy.eval();del reference,actual,values,items;reference=None;gc.collect()
        assert runtime.fingerprint(policy)==initial
    passed=all(x['passed'] for x in results.values())
    c.write(c.HERE/'RESULT.json',dict(status='PASS_INTERFACE_ONLY' if passed else 'FAIL_FIXED4_ACCUMULATION',
        unix=time.time(),results=results,old32_max_abs=old['max_abs_old'],source_bridge_sha256=runtime.BRIDGE_SHA,
        trainable_fingerprint=initial,trainable_exact_unchanged=True,forward_decisions=192,backward_calls=32,
        optimizer_steps=0,simulator_actions=0,navigation_gain=False,recommended_microbatch=4 if passed else None),True)
    print(json.dumps(dict(status='PASS_INTERFACE_ONLY' if passed else 'FAIL_FIXED4_ACCUMULATION',results=results)),flush=True)

