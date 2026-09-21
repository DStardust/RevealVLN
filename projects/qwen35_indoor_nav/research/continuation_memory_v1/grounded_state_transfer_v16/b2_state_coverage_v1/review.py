"""Recompute certificates from traces and original arrays; export causal model inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from v16_common import LINE, read, write, sha, digest
from evaluator_v16 import legacy, state_sequence, query_contexts, compose
sys.path.insert(0,str(HERE))
from certify import certify, HISTORIES, QUERIES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();families=read(run/'DATASET.json')['families']
    config=read(run/'BINDING.json')['config'];lookup={};features=[];records=[];reviewed=[]
    arrays={};owners={};physical_owners={};family_signatures=set();counts=Counter();byhouse=Counter()
    action_patterns=defaultdict(Counter);pattern_examples=[];ages=[]
    def array(root, pixel_sha, kind):
        key=(str(root),pixel_sha,kind)
        if key not in arrays:
            p=root/(pixel_sha+'.'+kind+'.npy');a=np.load(p,allow_pickle=False)
            if hashlib.sha256(a.tobytes()).hexdigest()!=pixel_sha:raise ValueError('ARRAY_PIXEL_HASH_MISMATCH')
            if kind=='rgb':
                if a.shape!=(224,224,3) or a.dtype!=np.uint8:raise ValueError('RGB_CONTRACT')
                pixels=None
            else:
                if a.shape!=(224,224) or a.dtype!=np.uint32:raise ValueError('SEMANTIC_CONTRACT')
                ids,nums=np.unique(a,return_counts=True);pixels={str(int(i)):int(n) for i,n in zip(ids,nums)}
            arrays[key]=dict(path=str(p.relative_to(LINE)),file_sha256=sha(p),pixels=pixels)
        return arrays[key]
    def feature(trace,t,instruction,split):
        item=dict(instruction=instruction,rgb_refs=['sha256:'+o['rgb_hash'] for o in trace['observations'][max(0,t-1):t+1]],
                  executed=[dict(F='move_forward',L='turn_left',R='turn_right')[a] for a in trace['actions'][max(0,t-8):t]])
        key=digest(item)
        if key in owners and owners[key]!=split:raise ValueError('CROSS_SPLIT_CURRENT_INPUT')
        owners[key]=split
        if key not in lookup:lookup[key]=len(features);features.append(dict(key=key,**item))
        return lookup[key]
    for family in families:
        compiler=legacy.Compiler(**family['compiler']);traces={};root=LINE/family['content_root'];teachers={}
        for key,ref in family['traces'].items():
            p=LINE/ref['path']
            if sha(p)!=ref['sha256']:raise ValueError('TRACE_HASH_CHANGED')
            trace=read(p)
            for obs in trace['observations']:
                array(root,obs['rgb_hash'],'rgb');pixels=array(root,obs['semantic_hash'],'semantic')['pixels']
                if pixels!=obs['pixels']:raise ValueError('SEMANTIC_COUNTS_DO_NOT_MATCH_ARRAY')
            traces[key]=trace
        certificate=certify(compiler,family['histories'],family['suffixes'],traces)
        if certificate!=family['certificate']:raise ValueError('CERTIFICATE_RECOMPUTATION_MISMATCH')
        split=family['split'];house=family['house'];byhouse[(split,house)]+=1
        cutoff=len(family['histories']['seen']);sequence_indices={}
        signature=digest(dict(house=house,position=family['initial_position'],roles=family['roles']))
        if signature in family_signatures:raise ValueError('DUPLICATE_PHYSICAL_FAMILY')
        family_signatures.add(signature)
        for h in HISTORIES:
            prefix=traces[h+'__direct_stop'];physical=digest(dict(actions=family['histories'][h],rgb=[o['rgb_hash'] for o in prefix['observations']]))
            if physical in physical_owners and physical_owners[physical]!=split:raise ValueError('CROSS_SPLIT_PHYSICAL_HISTORY')
            physical_owners[physical]=split
            pattern=''.join(family['histories'][h][:12]);state=int(h.startswith('seen'))
            action_patterns[pattern][state]+=1
            pattern_examples.append(dict(family=family['family_id'],split=split,house=house,pattern=pattern,state=state))
            if certificate['anchor_age'][h] is not None:ages.append(certificate['anchor_age'][h])
            for task in ('task_A','task_T'):
                instruction=family['compiler']['tasks']['task_A']['instruction'] if task=='task_A' else family['terminal_instruction']
                teacher=min((q for q in QUERIES if certificate['labels'][h+'__'+q+'__'+task]['label']=='PASS'), key=lambda q:(len(family['suffixes'][q]),q))
                teachers[(h,task)]=teacher
                for q in QUERIES:
                    trace=traces[h+'__'+q];states=state_sequence(compiler,trace['observations'],task)
                    queries=query_contexts(compiler,trace,task);label=certificate['labels'][h+'__'+q+'__'+task]['label'];y=int(label=='PASS')
                    if any(int(compose(states[t],queries[t]))!=y for t in range(len(trace['actions']))):raise ValueError('QUERY_STATE_DISAGREEMENT')
                    key=(h,task,q);sequence_indices[key]=len(records)
                    record=dict(family_id=family['family_id'],house=house,split=split,history=h,task=task,continuation=q,
                        features=[feature(trace,t,instruction,split) for t in range(len(trace['actions']))],cutoff=cutoff,
                        targets=['FLRS'.index(a) for a in trace['actions']],
                        action_masks=[int(t>=cutoff and q==teacher) for t in range(len(trace['actions']))],
                        state_targets=states[:len(trace['actions'])],state_masks=[1]*len(trace['actions']),
                        query_contexts=queries,y=[y]*len(trace['actions']),query_masks=[1]*len(trace['actions']),
                        source_trace=family['traces'][h+'__'+q],teacher_selection='shortest registered real PASS continuation; no model scores')
                    records.append(record)
        cases=[]
        for a,b,sham in (('seen','missing','seen_sham'),('missing','seen','missing_sham')):
            ids=[sequence_indices[(h,'task_A',teachers[h,'task_A'])] for h in (a,b,sham)]
            cutoff_stop=min(len(records[i]['features']) for i in ids)
            for t in range(cutoff,cutoff_stop):
                if len({records[i]['features'][t] for i in ids})!=1:break
                if records[ids[0]]['targets'][t]==records[ids[1]]['targets'][t]:continue
                cases.append(dict(task='task_A',correct=[ids[0],t],wrong=[ids[1],t],sham=[ids[2],t],
                    target=records[ids[0]]['targets'][t],interpretation='Actual same-input teacher fork after terminal-absent takeover; not uniqueness of every valid action'))
                controls=[sequence_indices[(h,'task_T',teachers[h,'task_T'])] for h in (a,b,sham)]
                if all(t<len(records[i]['features']) for i in controls) and len({records[i]['features'][t] for i in controls})==1:
                    cases.append(dict(task='task_T',correct=[controls[0],t],wrong=[controls[1],t],sham=[controls[2],t],target=records[controls[0]]['targets'][t]))
                break
        reviewed.append(dict(family_id=family['family_id'],parent_family_id=family['parent_family_id'],house=house,split=split,certificate=certificate,mechanism_cases=cases))
        counts.update(families=1,physical_executions=12,physical_histories=4,cross_labels=24,mechanism_cases=sum(c['task']=='task_A' for c in cases),task_T_control_cases=sum(c['task']=='task_T' for c in cases))
    fit={h for (s,h) in byhouse if s=='FIT'};dev={h for (s,h) in byhouse if s=='DEV'}
    if fit & dev:raise ValueError('HOUSE_SPLIT_LEAK')
    write(out/'POLICY_INPUTS.json',features,True);write(out/'SUPERVISION.json',records,True)
    write(out/'FAMILIES.json',reviewed,True)
    write(out/'CONTENT_MANIFEST.json',[dict(pixel_sha256=k[1],kind=k[2],path=v['path'],file_sha256=v['file_sha256']) for k,v in arrays.items()],True)
    leave_family_correct=0;leave_family_total=0
    for sample in pattern_examples:
        training=[r for r in pattern_examples if r['family']!=sample['family'] and r['pattern']==sample['pattern']]
        if not training:continue
        distribution=Counter(r['state'] for r in training);prediction=max((0,1),key=lambda v:(distribution[v],-v))
        leave_family_correct+=prediction==sample['state'];leave_family_total+=1
    shortcut=dict(scope='descriptive action-pattern shortcut audit; labels at complete history cutoff',
        patterns=len(action_patterns),patterns_with_both_states=sum(len(v)==2 for v in action_patterns.values()),
        fitted_pattern_majority_correct=sum(max(v.values()) for v in action_patterns.values()),
        total_histories=len(pattern_examples),leave_family_out_seen_pattern_correct=leave_family_correct,
        leave_family_out_seen_pattern_total=leave_family_total,
        interpretation='High accuracy suggests a shortcut. Exact input pairs alone do not rule out history-action-pattern cues.',
        patterns_detail={k:dict(v) for k,v in action_patterns.items()})
    write(out/'SHORTCUT_AUDIT.json',shortcut,True)
    write(out/'RESULT.json',dict(status='VERIFIED_TERMINAL_ABSENT_COUNTERPARTS' if families else 'NO_CERTIFIED_FAMILIES',
        counts=counts,by_house=[dict(split=s,house=h,families=n) for (s,h),n in sorted(byhouse.items())],
        unique_parent_families=len({f['parent_family_id'] for f in families}),terminal_present_at_takeover=False,raw_arrays_verified=len(arrays),unique_policy_inputs=len(features),supervision_sequences=len(records),
        configured_target=config['planned_families'],target_complete=len(families)==config['planned_families'],
        training_admission='CONTROLLED_MECHANISM_ONLY_WITH_REPORTED_SHORTCUT_LIMITS' if families else False,all_training_arms_share_pool=True,
        formal_test_accessed=False,natural_navigation_recovery_claim=False,
        reused_development_houses=True,independent_new_test_houses=0,anchor_age_min=min(ages) if ages else None,
        anchor_age_max=max(ages) if ages else None,seconds=time.monotonic()-began,
        scope='Actual terminal-absent extensions of old parent histories, same five development houses; stationary reorientation, not translational recovery, independent new families or method gain.'),True)
    print((out/'RESULT.json').read_text(),flush=True)

if __name__=='__main__':main()
