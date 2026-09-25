"""Shared actual ordinary rollouts and lawful successful continuation actions."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch


def main(root):
    u.verify_sources(root);p=u.read(root/'PROTOCOL.json')
    ordinary={'FIT':[],'DEV':[]};counts={k:Counter() for k in ordinary};seen=set();certificates={}
    for session in (root/'capture/evaluation').glob('*'):
        assert (session/'STATE_SEAL.json').exists(),'UNSEALED_CAPTURE'
        seal=u.read(session/'STATE_SEAL.json');assert seal['base_before']==seal['base_after']==p['base_state_sha256']
        for path in session.glob('episodes/*/COMPLETE.json'):
            c=u.read(path);assert c['id'] not in seen;seen.add(c['id'])
            cache=c['cache'];assert u.sha(cache['path'])==cache['sha256']
            certificates[cache['path']]=cache['sha256']
            row=torch.load(cache['path'],map_location='cpu',weights_only=True)
            assert row['action_boundary']=='AFTER_NATIVE_ASSISTANT_HEADER_V2'
            ordinary[row['partition']].append(row);counts[row['partition']].update(row['targets'].tolist())
    assert len(seen)==100 and all(ordinary.values()),'INCOMPLETE_ORDINARY_POOL'
    source=u.read(u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json')
    index=u.read(Path(p['memory_run'])/'features/FEATURE_INDEX.json');dense=[];dense_counts=Counter();attempts=[]
    for family in source['families']:
        if family['partition'].upper()!='FIT':continue
        fid=family['family_id'];audit=u.read(Path(p['physical_data'])/fid/'AUDIT.json')
        features=u.read(Path(index[fid])/'RESULT.json')
        lookup={(x['history'],x['continuation'],x['task']):x for x in features['rows']}
        for history in family['candidate']['histories']:
            for task in list(family['compiler']['tasks'])+['task_T']:
                legal=[x for x in audit['traces'] if x['history']==history and x['labels'][task]=='pass' and x['collisions']==0
                    and all(not x['old_event_seen'][role] or x['policy_witness_visible'][role] for role in ('anchor_A','anchor_B','terminal'))]
                if not legal:
                    attempts.append(dict(family=fid,history=history,task=task,status='NO_LEGAL_VISIBLE_PASS'));continue
                teacher=min(legal,key=lambda x:(x['decisions'],x['continuation']))
                row=lookup[history,teacher['continuation'],task]
                assert u.sha(row['path'])==row['sha256'];assert u.sha(teacher['path'])==teacher['sha256']
                certificates[row['path']]=row['sha256'];certificates[teacher['path']]=teacher['sha256']
                cached=torch.load(row['path'],map_location='cpu',weights_only=True);trace=u.read(teacher['path'])
                assert cached['action_boundary']=='AFTER_NATIVE_ASSISTANT_HEADER_V2'
                chosen=[i for i,q in enumerate(cached['query_steps']) if q>=teacher['cutoff']]
                if not chosen:continue
                steps=torch.tensor([cached['query_steps'][i] for i in chosen]);target=torch.tensor(['SFLR'.index(trace['actions'][int(q)]) for q in steps])
                dense.append(dict(memory_features=cached['memory_features'],actor_features=cached['features'][chosen],
                    base_logits=cached['base_action_logits'][chosen],query_steps=steps,targets=target,
                    family=fid,house=family['house'],history=history,task=task,continuation=teacher['continuation'],partition='FIT'))
                dense_counts.update(target.tolist());attempts.append(dict(family=fid,history=history,task=task,status='ADMITTED',
                    continuation=teacher['continuation'],supervised_queries=len(chosen)))
    assert dense and dense_counts[1]+dense_counts[2]+dense_counts[3]>dense_counts[0],'NO_MOTION_COVERAGE_REPAIR'
    assert all(counts['FIT'][i]>0 for i in range(4)),'ORDINARY_ACTION_CLASS_MISSING'
    output=root/'data';output.mkdir(exist_ok=False)
    weights=torch.tensor([1/max(1,dense_counts[i]) for i in range(4)]);weights/=weights.mean()
    torch.save(dict(ordinary_fit=ordinary['FIT'],ordinary_dev=ordinary['DEV'],dense=dense,dense_class_weights=weights,
        action_boundary='AFTER_NATIVE_ASSISTANT_HEADER_V2',future_policy_input=False),output/'POOLS.pt')
    u.write(output/'ADMISSION.json',dict(ordinary={s:dict(trajectories=len(ordinary[s]),queries=sum(counts[s].values()),actions=dict(counts[s])) for s in ordinary},
        dense_continuations=len(dense),dense_queries=sum(dense_counts.values()),dense_action_counts=dict(dense_counts),
        dense_class_weights=weights.tolist(),attempts=attempts,certificates=certificates,pools_sha256=u.sha(output/'POOLS.pt'),
        ordinary_teacher='Frozen native behavior, including unsuccessful trajectories; not oracle-correct action labels',
        dense_teacher='Shortest actually executed collision-free visible PASS suffix per FIT family/history/task',
        old_sparse_groups_unchanged=True,no_unseen_training=True))
    print(u.read(output/'ADMISSION.json')['ordinary'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)

