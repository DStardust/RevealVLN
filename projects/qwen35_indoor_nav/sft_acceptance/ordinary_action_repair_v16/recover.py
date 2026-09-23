"""Real V13-policy and route-teacher observation collection, with causal frozen features."""
import json,os,socket,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u

def main(session):
    u.verify();p=u.read(session/'CONFIG.json');sys.path.insert(0,str(u.V16));enc=u.load('motion16_recovery_encoder',u.V16/'encoder.py')
    import torch
    u.write(session/'PROGRESS.json',dict(status='LOADING_MODEL',complete=0,unix=time.time()))
    model,policy,initial=enc.load_policy(session,p)
    checkpoint=torch.load(p['reference_checkpoint'],map_location='cpu',weights_only=True);u.validate_head(model.trainable_state(policy),checkpoint['trainable'])
    assert u.sha(Path(p['reference_checkpoint']))==p['reference_checkpoint_sha256']
    weight=checkpoint['trainable']['action_head.weight'].cuda();bias=checkpoint['trainable']['action_head.bias'].cuda()
    window=u.c.Window()
    class Store:
        def get(self,i,t):return window.item()
    forward=enc.Forward(model,policy,Store(),[dict(record_idx=0,t=0,target=0,weight=1.)])
    parent,child=socket.socketpair();parent.settimeout(300);log=(session/'simulator.log').open('x');env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    proc=subprocess.Popen([p['habitat_python'],'-I','-B',p['runtime_executor'],str(child.fileno()),str(session),'C'],pass_fds=(child.fileno(),),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    child.close();stream=parent.makefile('rw');complete=[];order=u.read(Path(p['order_path']))
    def call(req):
        stream.write(json.dumps(req)+'\n');stream.flush();line=stream.readline()
        if not line:raise EOFError('SIMULATOR_EOF')
        return json.loads(line)
    try:
        with torch.inference_mode():
            for rank in p['scheduled_ranks']:
                row=order[rank];folder=session/'pairs'/f'pair_{rank:03d}'/'C';folder.mkdir(parents=True)
                assert window.receive(call(dict(op='reset',index=row['index'])))
                features=[];base_logits=[];ids=[];step=0
                while True:
                    h,z,processed,timing=forward(0);final=torch.nn.functional.linear(h,weight,bias)[0];assert torch.isfinite(final).all()
                    action=u.c.ACTIONS[int(final.argmax())];raw=u.c.raw_input(window)
                    reply=call(dict(op='action',action=action));actual=reply.pop('executed_action')
                    assert set(reply)==({'done'} if reply['done'] else {'done','rgb'}),'PRIVATE_REPLY_FIELDS'
                    if actual is not None:
                        step+=1;u.append(folder/'POLICY_STEPS.jsonl',dict(step=step,raw=raw,processed=processed,logits=final.cpu().tolist(),base_logits=z[0].cpu().tolist(),native_action=action,proposed_action=action,executed_action=actual,collection_mode=row['mode'],**timing))
                        features.append(h[0].cpu().clone());base_logits.append(z[0].cpu().clone());ids.append(raw['input_key'])
                    if step%20==0:u.write(session/'PROGRESS.json',dict(status='COLLECTING',rank=rank,mode=row['mode'],step=step,complete=len(complete),planned=len(p['scheduled_ranks']),unix=time.time()))
                    if reply['done']:break
                    assert actual in u.c.ACTIONS[:3],'EXECUTED_HISTORY_ACTION'
                    assert window.receive(reply,actual)
                # Private truth is opened only after the entire trajectory has ended.
                if step:
                    result=u.audit_episode(folder,row['index'],p['gt_path'],collection=True)
                    labels=u.c.records(folder/'SUPERVISION_ONLY.jsonl');assert len(labels)==len(features)==result['steps']
                    for i,label in enumerate(labels):
                        assert label['step']==i+1 and label['distance_before']==result['distances'][i]
                        assert label['target']==int(result['distances'][i]<3),'STOP_LABEL_DEFINITION'
                else:
                    result=u.read(folder/f'episode_{row["index"]:02d}.json');labels=[]
                    assert result['termination']=='TEACHER_UNAVAILABLE'
                tmp=folder/'FEATURES.tmp.pt';torch.save(dict(features=torch.stack(features) if features else torch.empty(0,2048),logits=torch.stack(base_logits) if base_logits else torch.empty(0,4),input_keys=ids,source_checkpoint_sha256=p['checkpoint_sha256']),tmp);tmp.rename(folder/'FEATURES.pt')
                actual_logs=u.c.records(folder/'POLICY_STEPS.jsonl');source_logs=u.c.records(Path(row['source_folder'])/'POLICY_STEPS.jsonl');cutoff=row['cutoff']
                assert step>=cutoff,'INCOMPLETE_REGISTERED_PREFIX'
                for i in range(cutoff):
                    assert actual_logs[i]['raw']==source_logs[i]['raw'] and actual_logs[i]['executed_action']==source_logs[i]['executed_action'],'FULL_REPLAY_PREFIX_MISMATCH'
                if step>cutoff:assert actual_logs[cutoff]['raw']==source_logs[cutoff]['raw'],'TAKEOVER_INPUT_MISMATCH'
                u.write(folder/'REPLAY_AUDIT.json',dict(cutoff=cutoff,matched=True,source_policy_sha256=row['policy_sha256'],real_teacher_after_prefix=True,teacher_decisions=step-cutoff),True)
                state=u.c.model_identity(policy);assert state['sha256']==initial['sha256'],'BASE_STATE_CHANGED'
                refs={h for r in u.c.records(folder/'POLICY_STEPS.jsonl') for h in r['raw']['rgb_sha256']} if step else set()
                seal=dict(rank=rank,episode_id=row['episode_id'],house=row['house'],mode=row['mode'],steps=step,termination=result['termination'],base_unchanged=True,base_state_sha256=initial['sha256'],positive=sum(x['target'] for x in labels),files={str(f.relative_to(folder)):u.sha(f) for f in folder.iterdir() if f.is_file()},frames={str(session/'frames'/(h+'.png')):u.sha(session/'frames'/(h+'.png')) for h in refs})
                u.write(folder.parent/'COLLECT.json',seal,True);complete.append(rank)
                u.write(session/'PROGRESS.json',dict(status='COLLECTING',rank=rank,mode=row['mode'],step=step,complete=len(complete),planned=len(p['scheduled_ranks']),unix=time.time()))
            assert call(dict(op='close'))['closed']
        u.write(session/'RESULT.json',dict(status='COMPLETE',ranks=complete),True)
    except BaseException as error:
        u.write(session/'FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),completed=complete),True);raise
    finally:
        forward.close();stream.close();parent.close()
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15)
        log.close()
if __name__=='__main__':main(Path(sys.argv[1]))
