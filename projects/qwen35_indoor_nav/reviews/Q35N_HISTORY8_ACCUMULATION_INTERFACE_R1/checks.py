import gc
import json
import time
def run(model,policy,collate,forward,c,fingerprint,initial_sha):
    import torch
    pair=c.LINE/'sft_acceptance/ordinary_history8_paired_train_v1'
    old=c.load('original32',c.LINE/'reviews/Q35N_STOP_FEATURE_PREFIX_DIAGNOSTIC_V2/checks.py').check(model,policy,collate,forward,c,'entry')
    assert old['max_abs_old']<=1e-5 and old['original_argmax_same']
    runtime=c.load('pair_runtime',pair/'runtime.py')
    # Change only parameter storage precision after the original entry regression.
    policy.exec_embed.float();policy.action_query=torch.nn.Parameter(policy.action_query.detach().float())
    assert runtime.sha(runtime.BRIDGE)==runtime.BRIDGE_SHA
    state=torch.load(runtime.BRIDGE,map_location='cpu',weights_only=True)
    model.load_trainable(policy,state['trainable'])
    for name,p in policy.named_parameters():
        if p.requires_grad:assert p.dtype==torch.float32 and torch.equal(p.detach().cpu(),state['trainable'][name])
    initial=runtime.fingerprint(policy)
    data=runtime.ordinary();rows,_=data.load_rows()
    samples=json.loads((pair.parent/'ordinary_prefix_history8_v1/DIAGNOSTIC_SAMPLES.json').read_text())[:8]
    loss_fn=c.load('paired_loss',pair/'loss.py').local_micro_loss
    results={}
    for mode,store in runtime.stores(rows).items():
        dataset=model.DecisionDataset(samples,store,policy.processor)
        items=[dataset[i] for i in range(8)]
        allweights=sum(item['weight'] for item in items)
        grads=[];values=[];cost=[]
        for micro in (8,4):
            policy.train();policy.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            logits=[];began=time.perf_counter()
            for start in range(0,8,micro):
                batch=collate(items[start:start+micro])
                y=batch.pop('targets').to('cuda:0');w=batch.pop('weights').to('cuda:0')
                output=policy.forward_batch(**{k:v.to('cuda:0') for k,v in batch.items()})
                loss=loss_fn(output,y,w,allweights,1);loss.backward();logits.append(output.detach().cpu())
                del output,loss,batch
            torch.cuda.synchronize()
            trainable=[p for p in policy.parameters() if p.requires_grad]
            assert len(trainable)==28 and all(p.dtype==torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
            grads.append(torch.cat([p.grad.detach().float().reshape(-1).cpu() for p in trainable]))
            values.append(torch.cat(logits))
            cost.append(dict(microbatch=micro,wall_seconds=time.perf_counter()-began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        reference=grads[0];actual=grads[1]
        relative=float((reference-actual).norm()/reference.norm().clamp_min(1e-12))
        cosine=float(torch.nn.functional.cosine_similarity(reference,actual,dim=0))
        logit_relative=float((values[0]-values[1]).norm()/values[0].norm().clamp_min(1e-12))
        action_match=bool(torch.equal(values[0].argmax(1),values[1].argmax(1)))
        passed=relative<=.05 and cosine>=.999 and logit_relative<=.01 and action_match and max(v['peak_reserved_bytes'] for v in cost)<=25*1024**3
        results[mode]=dict(passed=passed,gradient_relative_l2=relative,gradient_cosine=cosine,
            logit_relative_l2=logit_relative,argmax_equal=action_match,cost=cost)
        c.write(c.HERE/'PROGRESS.json',dict(unix=time.time(),mode=mode,results=results),False)
        policy.zero_grad(set_to_none=True);policy.eval();del grads,values,reference,actual;gc.collect()
        assert runtime.fingerprint(policy)==initial
    passed=all(x['passed'] for x in results.values())
    c.write(c.HERE/'RESULT.json',dict(status='PASS_INTERFACE_ONLY' if passed else 'FAIL_NUMERICAL_INTERFACE',
        unix=time.time(),results=results,old32_max_abs=old['max_abs_old'],
        source_bridge_sha256=runtime.BRIDGE_SHA,trainable_fingerprint=initial,trainable_exact_unchanged=True,
        forward_decisions=96,backward_calls=6,optimizer_steps=0,simulator_actions=0,
        navigation_gain=False,recommended_microbatch=8 if passed else None),True)
    print(json.dumps(dict(status='PASS_INTERFACE_ONLY' if passed else 'FAIL_NUMERICAL_INTERFACE',results=results)),flush=True)


