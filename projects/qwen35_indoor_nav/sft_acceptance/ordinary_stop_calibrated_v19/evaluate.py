"""Same-process native/candidate pairs, real independent environments and atomic seals."""
import json,os,socket,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u

def audit_prefix(a,b):
    import math
    if a['raw']!=b['raw'] or a['processed']!=b['processed']:raise ValueError('PREFIX_INPUT_MISMATCH')
    if not all(math.isfinite(x) for x in a['base_logits']+b['base_logits']):raise ValueError('NONFINITE_LOGITS')
    delta=max(abs(x-y) for x,y in zip(a['base_logits'],b['base_logits']))
    if a['base_action']!=b['base_action']:raise ValueError('PREFIX_BASE_ARGMAX_FLIP')
    return delta

def main(session):
    u.verify();p=u.read(session/'CONFIG.json');run=Path(p['run']);c=u.c
    sys.path.insert(0,str(u.V16))
    enc=u.load('calibrated19_encoder',u.V16/'encoder.py')
    import torch
    write=lambda name,value,exclusive=False:u.write(session/name,value,exclusive)
    write('PROGRESS.json',dict(status='LOADING_MODEL',unix=time.time(),complete=0))
    model,policy,initial=enc.load_policy(session,p)
    ckpt=torch.load(run/'CANDIDATE.pt',map_location='cpu',weights_only=True)
    original=model.trainable_state(policy);u.validate_head(original,ckpt['trainable'])
    assert u.sha(run/'CANDIDATE.pt')==p['candidate_sha256'],'CANDIDATE_CHANGED'
    weight=ckpt['trainable']['action_head.weight'].cuda();bias=ckpt['trainable']['action_head.bias'].cuda()
    assert u.sha(Path(p['reference_checkpoint']))==p['reference_checkpoint_sha256'],'REFERENCE_CHANGED'
    reference=torch.load(p['reference_checkpoint'],map_location='cpu',weights_only=True)
    u.validate_head(original,reference['trainable'])
    ref_weight=reference['trainable']['action_head.weight'].cuda();ref_bias=reference['trainable']['action_head.bias'].cuda()
    reference_stamp=u.state_stamp(dict(weight=ref_weight,bias=ref_bias))
    write('HEAD_IDENTITY.json',dict(A=p['reference_checkpoint_sha256'],B=p['candidate_sha256'],reference=reference_stamp,candidate=u.state_stamp(dict(weight=weight,bias=bias))),True)
    candidate_stamp=u.state_stamp(dict(weight=weight,bias=bias))
    windows={a:c.Window() for a in ('A','B')}
    class Store:
        def get(self,i,t):return windows[('A','B')[i]].item()
    fw=enc.Forward(model,policy,Store(),[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(2)])
    processes=[];streams={};sockets=[];logs=[];completed=[]
    def call(arm,req):
        stream=streams[arm];stream.write(json.dumps(req)+'\n');stream.flush();raw=stream.readline()
        if not raw:raise EOFError('SIMULATOR_EOF_'+arm)
        response=json.loads(raw)
        if req.get('op')=='action':assert response.pop('executed_action')==req['action'],'EXECUTOR_ACTION_CHANGED'
        return response
    try:
        # Three fixed FIT diagnostic inputs. Does not repeat the 5487-input replay.
        rows=c.records(u.OLD/'POLICY_INPUTS.jsonl')
        class Gold:
            def get(self,i,t):
                from PIL import Image
                r=rows[i];images=[]
                for h in r['rgb_sha256']:
                    with Image.open(u.ASSET/'data_pipeline/ordinary_route_teacher_v11/run_001/content'/(h+'.png')) as im:
                        image=im.convert('RGB');assert __import__('hashlib').sha256(image.tobytes()).hexdigest()==h;images.append(image)
                return dict(instruction=r['instruction'],executed=r['executed_actions'],images=images)
        # Use the same forward wrapper sequentially; never install duplicate hooks.
        fw.close();gold=enc.Forward(model,policy,Gold(),[dict(record_idx=i,t=0,target=0,weight=1.) for i in (0,1829,3658)])
        cache=torch.load(u.OLD/'FEATURES.pt',map_location='cpu',weights_only=True);parity=[]
        with torch.inference_mode():
            for j,i in enumerate((0,1829,3658)):
                h,z,processed,timing=gold(j);old=cache['logits'][i];cand=torch.nn.functional.linear(h,weight,bias)[0].cpu();current=z[0].cpu()
                cached_cand=torch.nn.functional.linear(cache['features'][i].float(),weight.cpu(),bias.cpu())
                parity.append(dict(record_id=rows[i]['record_id'],raw=rows[i],processed=processed,
                    cache_feature_sha256=c.tensor_identity(cache['features'][i]),live_feature_sha256=c.tensor_identity(h[0]),
                    feature_max_abs=float((h[0].cpu()-cache['features'][i]).abs().max()),
                    base_cache_logits=old.tolist(),base_live_logits=current.tolist(),base_argmax_flip=int(old.argmax()!=current.argmax()),
                    candidate_cache_logits=cached_cand.tolist(),candidate_live_logits=cand.tolist(),candidate_argmax_flip=int(cached_cand.argmax()!=cand.argmax()),
                    base_max_abs=float((old-current).abs().max()),timing=timing))
        gold.close();del cache
        write('CACHE_LIVE_PARITY.json',dict(scope='3 fixed FIT inputs, cross-session descriptive diagnostic, not full parity certification',rows=parity),True)
        fw=enc.Forward(model,policy,Store(),[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(2)])
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
        for arm in ('A','B'):
            parent,child=socket.socketpair();parent.settimeout(300);log=(session/f'simulator_{arm}.log').open('x')
            proc=subprocess.Popen([p['habitat_python'],'-I','-B',p['runtime_executor'],str(child.fileno()),str(session),arm],pass_fds=(child.fileno(),),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            child.close();processes.append(proc);sockets.append(parent);logs.append(log);streams[arm]=parent.makefile('rw')
        order=u.read(Path(p['order_path']));began=time.time()
        with torch.inference_mode():
            for rank in p['scheduled_ranks']:
                row=order[rank];folder=session/'pairs'/f'pair_{rank:03d}';folder.mkdir()
                for arm in ('A','B'):(folder/arm).mkdir()
                alive={a:windows[a].receive(call(a,dict(op='reset',index=row['index']))) for a in row['inference_order']}
                assert u.read(folder/'A/INITIAL_STATE.json')==u.read(folder/'B/INITIAL_STATE.json'),'INITIAL_STATE_MISMATCH'
                diverged=False;first=None;prefix_delta=0.;prefix_steps=0;step=0;seen={'A':set(),'B':set()}
                while any(alive.values()):
                    step+=1;decisions={}
                    for arm in row['inference_order']:
                        if not alive[arm]:continue
                        h,z,processed,timing=fw(('A','B').index(arm));base=z[0]
                        final=torch.nn.functional.linear(h,ref_weight,ref_bias)[0] if arm=='A' else torch.nn.functional.linear(h,weight,bias)[0]
                        assert torch.isfinite(final).all(),'NONFINITE_METHOD_LOGITS'
                        action=c.ACTIONS[int(final.argmax())];raw=c.raw_input(windows[arm]);key=raw['input_key']
                        record=dict(step=step,raw=raw,processed=processed,base_logits=base.cpu().tolist(),logits=final.cpu().tolist(),
                            base_action=c.ACTIONS[int(base.argmax())],native_action=action,executed_action=action,
                            candidate_differs_from_base=action!=c.ACTIONS[int(base.argmax())],override=False,
                            repeated_input=key in seen[arm],feature_sha256=c.tensor_identity(h),**timing)
                        seen[arm].add(key);decisions[arm]=record
                        u.append(folder/arm/'POLICY_STEPS.jsonl',record)
                    if not diverged:
                        try:
                            assert set(decisions)=={'A','B'},'UNEXPLAINED_PREFIX_TERMINATION'
                            delta=audit_prefix(decisions['A'],decisions['B']);prefix_delta=max(prefix_delta,delta);prefix_steps+=1
                        except BaseException as e:
                            u.write(folder/'FIRST_DIVERGENCE.json',dict(reason=repr(e),decisions=decisions),True);raise
                        if decisions['A']['executed_action']!=decisions['B']['executed_action']:diverged=True;first=step
                    for arm,r in decisions.items():alive[arm]=windows[arm].receive(call(arm,dict(op='action',action=r['executed_action'])),r['executed_action'])
                    if step%10==0:u.write(session/'PROGRESS.json',dict(status='EVALUATING',rank=rank,step=step,complete=len(completed),planned=len(p['scheduled_ranks']),unix=time.time()))
                result={a:u.audit_episode(folder/a,row['index'],p['gt_path']) for a in ('A','B')}
                if not diverged:
                    for k in ('positions','distances','success','spl','ndtw','steps','stopped'):assert result['A'][k]==result['B'][k],'NO_DIVERGENCE_TERMINAL_MISMATCH'
                now=c.model_identity(policy);unchanged=now['sha256']==initial['sha256']
                assert unchanged and candidate_stamp==u.state_stamp(dict(weight=weight,bias=bias)),'MODEL_STATE_CHANGED'
                assert reference_stamp==u.state_stamp(dict(weight=ref_weight,bias=ref_bias)),'REFERENCE_STATE_CHANGED'
                pair=dict(rank=rank,episode_id=row['episode_id'],house=row['house'],A=result['A'],B=result['B'],
                    first_action_divergence=first,input_prefix_matched=True,base_action_prefix_matched=True,
                    logits_bitwise_equal=prefix_delta==0,max_base_logit_delta=prefix_delta,prefix_steps=prefix_steps,
                    base_state_sha256=initial['sha256'],candidate_sha256=p['candidate_sha256'],reference_sha256=p['reference_checkpoint_sha256'],state_unchanged=True,
                    trace_hashes={str(f.relative_to(folder)):u.sha(f) for f in folder.glob('*/*') if f.is_file()})
                u.write(folder/'PAIR.json',pair,True);completed.append(rank)
                write('PROGRESS.json',dict(status='EVALUATING',rank=rank,step=step,complete=len(completed),planned=len(p['scheduled_ranks']),unix=time.time(),wall_seconds=time.time()-began))
            for arm in ('A','B'):assert call(arm,dict(op='close'))['closed']
        write('RESULT.json',dict(status='COMPLETE',ranks=completed,model_loaded=True,base_frozen=True,checkpoint_sha256=p['candidate_sha256']),True)
    except BaseException as error:
        write('FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),completed=completed),True);raise
    finally:
        fw.close()
        for stream in streams.values():stream.close()
        for sock in sockets:sock.close()
        for proc in processes:
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15)
        for log in logs:log.close()

if __name__=='__main__':main(Path(sys.argv[1]))
