"""One frozen action-only SFT configuration; no automatic retry or selection."""
import base64
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from policy import OUT,LINE,ROOT,DEVICE,ACTIONS,build,zero_memory,load_record,payload,policy_forward,trainable_state,load_trainable

def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def append(name,obj):
    with (OUT/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')

def event(name,**kw):
    obj=dict(event=name,unix=time.time(),**kw);append('EVENTS.jsonl',obj);print(json.dumps(obj),flush=True)

def check_lock():
    for name,h in json.loads((OUT/'PREREGISTRATION_LOCK.json').read_text()).items():
        if isinstance(h,str):assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h
    for name,h in json.loads((OUT/'CODE_LOCK.json').read_text()).items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h

def grad(p):
    return dict(present=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None,norm=float(p.grad.float().norm()) if p.grad is not None else None)

def preflight(policy,row):
    policy.eval();instruction,images,actions=load_record(row)
    rejected=[]
    p=payload(instruction,images,actions,0)
    for key in ['pose','navmesh','goal','reference_path','semantic','source_episode_id','future_rgb','target_action']:
        try:policy_forward(policy,dict(p,**{key:'forbidden'}),zero_memory())
        except ValueError:rejected.append(key)
        else:raise AssertionError('Whitelist failed')
    assert p['images']==images[:1] and p['executed']==[]
    assert payload(instruction,images,actions,2)['executed']==actions[:2]
    m1,a1,_=policy_forward(policy,p,zero_memory());m1.retain_grad()
    m2,a2,_=policy_forward(policy,payload(instruction,images,actions,1),m1)
    ce1=F.cross_entropy(a1,torch.tensor([ACTIONS.index(actions[0])],device=DEVICE))
    ce2=F.cross_entropy(a2,torch.tensor([ACTIONS.index(actions[1])],device=DEVICE));ce2.backward()
    lora=next(p for n,p in policy.named_parameters() if 'lora_B' in n)
    probes={k:grad(p) for k,p in [('previous_memory',m1),('writer',policy.writer.weight),('write_query',policy.write_query),('lora_B',lora),('action_head',policy.action_head.weight)]}
    passed=all(x['present'] and x['finite'] and x['norm']>1e-12 for x in probes.values())
    vision_frozen=all(not p.requires_grad and p.grad is None for p in policy.base.model.visual.parameters())
    save('TRAINING_PREFLIGHT.json',dict(real_action_ce=float(ce1.detach()),next_step_real_action_ce=float(ce2.detach()),probes=probes,adjacent_memory_gradient_pass=passed,vision_frozen=vision_frozen,whitelist_rejected=rejected,causal_window_pass=True,optimizer_updates=0))
    assert passed and vision_frozen
    policy.zero_grad(set_to_none=True)

@torch.no_grad()
def fixed_logits(policy,row):
    policy.eval();instruction,images,actions=load_record(row);m=zero_memory();out=[]
    for t in range(2):
        m,logits,_=policy_forward(policy,payload(instruction,images,actions,t),m);out.append(logits.cpu())
    return torch.cat(out)

def checkpoint(policy,name,row,opt=None,updates=0):
    before=fixed_logits(policy,row)
    path=OUT/'checkpoints'/f'{name}.pt';assert not path.exists()
    state=trainable_state(policy)
    torch.save(dict(trainable=state,optimizer=opt.state_dict() if opt else None,updates=updates,spec_sha256=hashlib.sha256((OUT/'EXPERIMENT_SPEC.json').read_bytes()).hexdigest()),path)
    with torch.no_grad():policy.action_head.bias.add_(1.0)
    changed=fixed_logits(policy,row)
    assert not torch.equal(before,changed)
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    load_trainable(policy,loaded['trainable'])
    after=fixed_logits(policy,row);delta=float((before-after).abs().max())
    append('RELOAD_CHECKS.jsonl',dict(checkpoint=str(path.relative_to(OUT)),updates=updates,logits_before=before.tolist(),logits_after=after.tolist(),max_abs_delta=delta,pass_gate=delta<=1e-5,perturbation_detected=True,scope='trainable checkpoint reloaded into fixed read-only base model'))
    assert delta<=1e-5
    return state

@torch.no_grad()
def evaluate(policy,rows,stage):
    policy.eval();all_loss=0.;n=0;correct=0;cm=np.zeros((4,4),dtype=np.int64);routes=defaultdict(lambda:[0.,0,0])
    for i,row in enumerate(rows):
        instruction,images,actions=load_record(row);m=zero_memory();ls=0.;ok=0
        for t,a in enumerate(actions):
            m,logits,details=policy_forward(policy,payload(instruction,images,actions,t),m)
            target=ACTIONS.index(a);pred=int(logits.argmax(-1));ce=float(F.cross_entropy(logits,torch.tensor([target],device=DEVICE)))
            assert np.isfinite(ce)
            cm[target,pred]+=1;ls+=ce;ok+=int(target==pred)
            append('MODEL_STEP_LEDGER.jsonl',dict(stage=stage,policy_file=row['policy_file'],job_id=row['job_id'],t=t,target=target,prediction=pred,ce=ce,**details))
        all_loss+=ls;n+=len(actions);correct+=ok
        r=routes[row['job_id']];r[0]+=ls;r[1]+=ok;r[2]+=len(actions)
        append('OFFLINE_RECORDS.jsonl',dict(stage=stage,policy_file=row['policy_file'],job_id=row['job_id'],decisions=len(actions),ce_sum=ls,correct=ok))
        if i%6==0:event('offline_progress',stage=stage,records=i+1,total=len(rows))
    result=dict(decisions=n,decision_micro_CE=all_loss/n,decision_micro_accuracy=correct/n,
        route_macro_CE=float(np.mean([r[0]/r[2] for r in routes.values()])),route_macro_accuracy=float(np.mean([r[1]/r[2] for r in routes.values()])),
        confusion_target_rows_prediction_columns=cm.tolist(),STOP_TP=int(cm[3,3]),STOP_FP=int(cm[:3,3].sum()),STOP_FN=int(cm[3,:3].sum()),STOP_TN=int(cm[:3,:3].sum()))
    save(f'OFFLINE_{stage.upper()}.json',result);event('offline_complete',stage=stage,**result)
    return result

@torch.no_grad()
def closed_loop(policy,rows,stage):
    policy.eval();parent,child=socket.socketpair();parent.settimeout(180)
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    env['PATH']=str(LINE/'.envs/q35n_habitat_v017_g0r/bin')+':/usr/bin:/bin'
    log=(OUT/f'habitat_{stage}.log').open('x')
    proc=subprocess.Popen([str(LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',str(OUT/'sim_server.py'),str(child.fileno())],pass_fds=(child.fileno(),),env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    child.close();stream=parent.makefile('rw')
    append('CHILD_PROCESSES.jsonl',dict(pid=proc.pid,stage=stage,role='Habitat executor',started_unix=time.time()))
    def rpc(x):
        stream.write(json.dumps(x)+'\n');stream.flush();line=stream.readline()
        if not line:raise RuntimeError('Simulator service EOF')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    try:
        for i,row in enumerate(rows):
            instruction=json.loads((LINE/'data_pipeline/ordinary_pilot_v1'/row['policy_file']).read_text())['instruction']
            resp=rpc(dict(op='reset',index=i,stage=stage));m=zero_memory();images=[];actions=[]
            for t in range(513):
                assert set(resp)=={'rgb'},'Privileged executor fields crossed policy boundary'
                rgb=np.frombuffer(base64.b64decode(resp['rgb']),dtype=np.uint8).reshape(224,224,3)
                images.append(Image.fromarray(rgb));images=images[-2:]
                m,logits,details=policy_forward(policy,dict(instruction=instruction,images=images,executed=actions[-8:]),m)
                a=ACTIONS[int(logits.argmax(-1))]
                append('MODEL_STEP_LEDGER.jsonl',dict(stage='closed_'+stage,job_id=row['job_id'],t=t,prediction=ACTIONS.index(a),logits=logits.cpu().tolist(),**details))
                resp=rpc(dict(op='action',action=a))
                if resp=={'done':True}:break
                actions.append(a)
            else:raise AssertionError('Executor did not end bounded episode')
            event('closed_episode_complete',stage=stage,index=i,decisions=t+1)
        stream.write(json.dumps({'op':'close'})+'\n');stream.flush();proc.wait(timeout=30)
        assert proc.returncode==0
        return True
    except Exception as ex:
        append('CLOSED_LOOP_FAILURES.jsonl',dict(stage=stage,error=repr(ex),traceback=traceback.format_exc()));event('closed_interface_failed',stage=stage,error=repr(ex))
        return False
    finally:
        stream.close();parent.close()
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        log.close()

def train(policy,rows,spec):
    policy.train();rng=random.Random(1109);cfg=spec['training']
    parameters=[p for p in policy.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
    opt.zero_grad(set_to_none=True);updates=0;chunks=0;decisions=0;loss_sum=0.;tokens=0;epoch=0
    while updates<cfg['max_optimizer_updates']:
        order=list(range(len(rows)));rng.shuffle(order)
        append('TRAIN_ORDER.jsonl',dict(epoch=epoch,order=[rows[i]['policy_file'] for i in order]))
        for index in order:
            row=rows[index];instruction,images,actions=load_record(row);m=zero_memory()
            for start in range(0,len(actions),cfg['tbptt']):
                losses=[]
                for t in range(start,min(start+cfg['tbptt'],len(actions))):
                    m,logits,details=policy_forward(policy,payload(instruction,images,actions,t),m)
                    target=ACTIONS.index(actions[t]);loss=F.cross_entropy(logits,torch.tensor([target],device=DEVICE))
                    assert torch.isfinite(loss)
                    losses.append(loss);tokens+=details['tokens'];decisions+=1;loss_sum+=float(loss.detach())
                    append('MODEL_STEP_LEDGER.jsonl',dict(stage='train',epoch=epoch,update_pending=updates+1,policy_file=row['policy_file'],job_id=row['job_id'],t=t,target=target,prediction=int(logits.argmax(-1)),ce=float(loss.detach()),**details))
                    assert tokens<=cfg['token_budget'],'TRAIN_TOKEN_BUDGET'
                torch.stack(losses).sum().backward();m=m.detach();chunks+=1
                del losses,logits,loss
                if chunks==cfg['gradient_accumulation_chunks']:
                    for p in parameters:
                        if p.grad is not None:p.grad.div_(decisions)
                    norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip_norm'],error_if_nonfinite=True)
                    assert torch.isfinite(norm) and float(norm)>0
                    opt.step();updates+=1
                    assert all(torch.isfinite(p).all() for p in parameters)
                    rec=dict(update=updates,unix=time.time(),mean_action_ce=loss_sum/decisions,decisions=decisions,chunks=chunks,grad_norm_before_clip=float(norm),lr=opt.param_groups[0]['lr'],cumulative_train_tokens=tokens,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                    append('UPDATE_LEDGER.jsonl',rec);event('optimizer_update',**rec)
                    opt.zero_grad(set_to_none=True);chunks=0;decisions=0;loss_sum=0.
                    if updates==cfg['max_optimizer_updates']:return opt,updates,tokens
            append('TRAIN_EPISODE_LEDGER.jsonl',dict(epoch=epoch,policy_file=row['policy_file'],completed=True,last_update=updates,decisions=len(actions)))
        epoch+=1

def main():
    check_lock();spec=json.loads((OUT/'EXPERIMENT_SPEC.json').read_text());split=json.loads((OUT/'SPLIT.json').read_text())
    event('model_load_start');policy=build();event('model_loaded')
    save('TRAINABLE_PARAMETERS.json',{n:{'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in policy.named_parameters() if p.requires_grad})
    preflight(policy,split['train'][0]);event('preflight_pass')
    initial=checkpoint(policy,'initial',split['train'][0]);event('initial_reload_pass')
    before=evaluate(policy,split['seen_house_route_dev'],'before')
    closed_before=closed_loop(policy,split['closed_loop'],'before')
    event('training_start',closed_before_pass=closed_before)
    opt,updates,tokens=train(policy,split['train'],spec)
    final=checkpoint(policy,'terminal',split['train'][0],opt,updates)
    changes={n:float((final[n].float()-initial[n].float()).norm()) for n in initial}
    save('PARAMETER_CHANGES.json',changes)
    assert changes['writer.weight']>0 and changes['action_head.weight']>0 and any(v>0 for n,v in changes.items() if 'lora_B' in n)
    after=evaluate(policy,split['seen_house_route_dev'],'after')
    closed_after=closed_loop(policy,split['closed_loop'],'after') if closed_before else None
    save('WORKER_RESULT.json',dict(training_interface_pass=True,offline_learning_signal=after['decision_micro_CE']<before['decision_micro_CE'],closed_loop_interface_pass=closed_before and closed_after,optimizer_updates=updates,training_tokens=tokens,all_forward_tokens=policy.forward_tokens,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),scientific_pass=False))
    event('worker_complete',updates=updates)

if __name__=='__main__':
    try:main()
    except BaseException as ex:
        append('FAILURES.jsonl',dict(error=repr(ex),traceback=traceback.format_exc(),unix=time.time()))
        raise
