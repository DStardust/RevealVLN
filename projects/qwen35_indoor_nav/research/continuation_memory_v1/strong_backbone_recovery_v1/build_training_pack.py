"""Admit actual action forks; no fabricated counterfactual action labels."""
import argparse
import importlib.util
import itertools
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch


def window(trace,step):
    return ([x['rgb_hash'] for x in trace['observations'][max(0,step-1):step+1]],trace['actions'][max(0,step-8):step])


def first_fork(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)


def main(data,features,output):
    output.mkdir(parents=True,exist_ok=False)
    source=u.read(u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json')
    spec=importlib.util.spec_from_file_location('strong_pack_compiler',u.PROJECT/'data_pipeline/mechanism_factory_v2/compiler.py')
    compiler_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler_module)
    pools={'FIT':[],'CHECK':[]};attempts=[]
    for family in source['families']:
        fid=family['family_id'];audit_path=data/fid/'AUDIT.json';audit=u.read(audit_path)
        result_path=features/fid/'RESULT.json';result=u.read(result_path)
        assert result['all_parameters_and_persistent_buffers_unchanged']
        traces={(x['history'],x['continuation']):(x,u.read(x['path'])) for x in audit['traces']}
        cache={}
        for row in result['rows']:
            assert u.sha(row['path'])==row['sha256']
            cache[row['history'],row['continuation'],row['task']]=(row,torch.load(row['path'],weights_only=True))
        compiler=compiler_module.Compiler(**family['compiler']);tasks={**family['compiler']['tasks'],'task_T':family['task_terminal_only']}
        suffixes=family['candidate']['continuations']
        for histories in [('H_A','H_B'),('H_A_N','H_B_N')]:
            for c0,c1 in itertools.combinations(suffixes,2):
                offset=first_fork(suffixes[c0],suffixes[c1]);cutoffs=[len(family['candidate']['histories'][h]) for h in histories]
                attempt=dict(family=fid,split=family['partition'],histories=histories,continuations=[c0,c1],offset=offset)
                attempts.append(attempt)
                if offset is None or any((cut+offset)%4 for cut in cutoffs):
                    attempt['excluded']='NO_FORK_AT_REGISTERED_MODEL_QUERY';continue
                steps=[cut+offset for cut in cutoffs]
                cells=[traces[h,c] for h in histories for c in (c0,c1)]
                if any(x[0]['collisions'] or not compiler.complete(x[1]) for x in cells):
                    attempt['excluded']='INCOMPLETE_OR_COLLIDING_ACTUAL_EXECUTION';continue
                if any(any(x[0]['old_event_seen'][role] and not x[0]['policy_witness_visible'][role] for role in ('anchor_A','anchor_B','terminal')) for x in cells):
                    attempt['excluded']='TASK_EVENT_NOT_OBSERVABLE_TO_POLICY';continue
                windows=[window(traces[h,c][1],s) for h,s in zip(histories,steps) for c in (c0,c1)]
                if any(x!=windows[0] for x in windows):
                    attempt['excluded']='DIFFERENT_CURRENT_SHORT_WINDOW';continue
                certificates={str(audit_path):u.sha(audit_path),str(result_path):u.sha(result_path)}
                for item,_ in cells:certificates[item['path']]=item['sha256']
                seqs=[];actors=[];max_delta=0.;states=[];known=[];targets=[];returns=[];pairs=[];base=[]
                for task_id,task in tasks.items():
                    seqrow=[];actorrow=[];baserow=[];staterow=[];knownrow=[];targetrow=[];returnrow=[]
                    for history,step in zip(histories,steps):
                        arow,a=cache[history,c0,task_id];brow,b=cache[history,c1,task_id]
                        q=a['query_steps'].index(step);n=q+1
                        assert a['query_steps'][:n]==b['query_steps'][:n]
                        assert a['input_hashes'][:n]==b['input_hashes'][:n], 'SHARED_EXECUTED_PREFIX_INPUT_MISMATCH'
                        delta=float((a['features'][:n]-b['features'][:n]).abs().max());max_delta=max(max_delta,delta)
                        assert torch.equal(a['base_action_logits'][:n].argmax(-1),b['base_action_logits'][:n].argmax(-1)), 'SHARED_EXECUTED_PREFIX_ARGMAX_FLIP'
                        certificates[arow['path']]=arow['sha256'];certificates[brow['path']]=brow['sha256']
                        assert torch.equal(a['memory_features'][:step+1],b['memory_features'][:step+1]),'CAUSAL_VISUAL_MEMORY_PREFIX_CHANGED'
                        seqrow.append(a['memory_features'][:step+1])
                        action_features=torch.zeros(step+1,a['features'].shape[-1]);action_base=torch.zeros(step+1,4)
                        action_features[a['query_steps'][:n]]=a['features'][:n];action_base[a['query_steps'][:n]]=a['base_action_logits'][:n]
                        actorrow.append(action_features);baserow.append(action_base)
                        trace=traces[history,c0][1];events=compiler.atoms(trace['observations']);st=[];seen=False
                        for event in events[:step+1]:
                            terminal=bool(event['terminal'])
                            if task_id=='task_T':value=[1,1,int(terminal),int(terminal)]
                            else:
                                anchor=bool(event[task['anchor']]);value=[int(seen),int(seen or anchor),int(terminal),int(seen and terminal)];seen|=anchor
                            st.append(value)
                        sampled=torch.tensor(st,dtype=torch.float32)
                        # Mask states until the corresponding positive event has
                        # appeared in two actual encoder query frames. The old
                        # checker and its labels are never rewritten.
                        masks=torch.ones_like(sampled,dtype=torch.bool);witness_seen=False;last=None
                        threshold=u.read(data/'PROTOCOL.json')['policy_witness_min_pixels']
                        for j,t in enumerate(range(step+1)):
                            obs=trace['observations'][t]
                            def visible(role):
                                return last is not None and any(last['policy_pixels'].get(str(i),0)>=threshold and obs['policy_pixels'].get(str(i),0)>=threshold for i in compiler.eligible[role])
                            if task_id!='task_T':
                                witness_seen|=visible(task['anchor'])
                                for k in (0,1,3):
                                    if sampled[j,k] and not witness_seen:masks[j,k]=False
                            if sampled[j,2] and not visible('terminal'):masks[j,2:]=False
                            last=obs
                        staterow.append(sampled);knownrow.append(masks)
                        labels=[traces[history,c][0]['labels'][task_id] for c in (c0,c1)]
                        assert all(v in ('pass','fail') for v in labels)
                        returnrow.append([float(v=='pass') for v in labels])
                        passed=[c for c,label in zip((c0,c1),labels) if label=='pass']
                        teacher=min(passed,key=lambda c:(len(traces[history,c][1]['actions'])-step,c)) if passed else None
                        targetrow.append(None if teacher is None else 'SFLR'.index(traces[history,teacher][1]['actions'][step]))
                    seqs.append(seqrow);actors.append(actorrow);base.append(baserow);states.append(staterow);known.append(knownrow);targets.append(targetrow);returns.append(returnrow)
                    pairs.append(['SFLR'.index(suffixes[c][offset]) for c in (c0,c1)])
                if not any(v is not None for row in targets for v in row):
                    attempt['excluded']='NO_ACTUALLY_SUCCESSFUL_TEACHER';continue
                length=max(x.shape[0] for row in seqs for x in row);width=seqs[0][0].shape[-1]
                shape=(len(tasks),2,length);x=torch.zeros(*shape,width);actor_x=torch.zeros(*shape,width);z=torch.zeros(*shape,4);st=torch.zeros(*shape,4);sk=torch.zeros(*shape,4,dtype=torch.bool)
                am=torch.zeros(shape,dtype=torch.bool);at=torch.zeros(shape,dtype=torch.long);pm=torch.zeros(shape,dtype=torch.bool);lengths=torch.zeros(len(tasks),2,dtype=torch.long)
                for t in range(len(tasks)):
                    for h in range(2):
                        n=seqs[t][h].shape[0];lengths[t,h]=n;x[t,h,:n]=seqs[t][h];actor_x[t,h,:n]=actors[t][h];z[t,h,:n]=base[t][h];st[t,h,:n]=states[t][h];sk[t,h,:n]=known[t][h]
                        if targets[t][h] is not None:
                            am[t,h,n-1]=True;at[t,h,n-1]=targets[t][h]
                            pm[t,h,n-1]=z[t,h,n-1].argmax().item()==targets[t][h]
                group=dict(family=fid,house=family['house'],histories=histories,continuations=[c0,c1],steps=steps,tasks=list(tasks),
                    features=x,actor_features=actor_x,lengths=lengths,base_action_logits=z,state_targets=st,state_known=sk,action_known=am,action_targets=at,
                    preservation_mask=pm,action_pairs=torch.tensor(pairs),returns=torch.tensor(returns),return_known=torch.ones(len(tasks),2,2,dtype=torch.bool),
                    actual_execution_certified=True,policy_witness_certified=True,same_current_input_certified=True,fork_is_runtime_query=True,
                    certificates=certificates,shared_prefix_feature_max_delta=max_delta)
                pools[family['partition'].upper()].append(group);attempt['admitted_debug']=True
    assert {g['house'] for g in pools['FIT']}.isdisjoint({g['house'] for g in pools['CHECK']})
    for split,groups in pools.items():
        torch.save(dict(split=split,scope='STREAMVLN_EXECUTED_HISTORY_FEATURES',future_in_policy_features=False,groups=groups,
            debug_only=True,old_admission_unchanged=True,not_blind_test=True),output/(split+'.pt'))
    u.write(output/'ADMISSION.json',dict(status='DEBUG_PACK_COMPLETE' if pools['FIT'] else 'NO_ADMISSIBLE_FIT_GROUPS',
        groups={k:len(v) for k,v in pools.items()},attempts=attempts,old_admission_unchanged=True,
        scope='Previously exposed two-house moving histories. New camera verified; no formal generalization admission.',
        witness_rule='Task roles only; unrelated filler category does not label any loss. CPU/read-only decision before training.',
        policy_input_equivalence='Current two raw frames and last eight executed actions; native long history may legitimately differ.',
        teacher='Actually successful shortest registered suffix at a true runtime-query action fork; no BC on failed histories',
        feature_shas={k:u.sha(output/(k+'.pt')) for k in pools}))
    if not pools['FIT']:raise ValueError('NO_ADMISSIBLE_FIT_GROUPS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.data,a.features,a.output)
