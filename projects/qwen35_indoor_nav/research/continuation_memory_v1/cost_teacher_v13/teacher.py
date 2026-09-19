"""Select efficient real PASS teachers; never fabricate labels or next-action conflicts."""
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def admit(family):
    candidates=defaultdict(list)
    for i,cell in enumerate(family['cells']):
        if cell['mask'] and cell['y']==1 and cell['tail']:
            candidates[cell['prefix']].append(i)
    selected={p:[i for i in ids if len(family['cells'][i]['tail'])==
                  min(len(family['cells'][j]['tail']) for j in ids)] for p,ids in candidates.items()}
    masks=[[0]*len(cell['tail']) for cell in family['cells']]
    contexts=defaultdict(list)
    for prefix,ids in selected.items():
        for i in ids:
            sequence=list(family['prefixes'][prefix]['features'])
            cell=family['cells'][i]
            assert cell['tail'][0]['feature']==sequence[-1]
            for t,row in enumerate(cell['tail']):
                if t:sequence.append(row['feature'])
                if row['mask']:contexts[tuple(sequence)].append((i,t,row['target']))
    ambiguous=duplicates=0
    for owners in contexts.values():
        if len({x[2] for x in owners})>1:
            ambiguous+=1
            continue
        cell,t,_=owners[0]
        masks[cell][t]=1
        duplicates+=len(owners)-1
    assert sum(map(sum,masks))>0
    return dict(selected_cells={str(p):ids for p,ids in selected.items()},masks=masks,
                admitted_actions=sum(map(sum,masks)),
                original_action_owners=sum(r['mask'] for cell in family['cells'] for r in cell['tail']),
                tied_conflicting_contexts_abstained=ambiguous,duplicate_context_action_owners_removed=duplicates)


def branch_cases(family, admission):
    cases=[];excluded=[]
    for task in sorted({p['task_id'] for p in family['prefixes']}):
        lookup={p['history_id']:i for i,p in enumerate(family['prefixes']) if p['task_id']==task}
        left,right,sham=[lookup[h] for h in ('H_A','H_B','H_A_I')]
        selected=[admission['selected_cells'][str(i)] for i in (left,right,sham)]
        if any(len(ids)!=1 for ids in selected):
            excluded.append(dict(task=task,reason='TIED_SHORTEST_TESTED_PASS_SUFFIX'));continue
        il,ir,ish=[ids[0] for ids in selected]
        prefixes=family['prefixes']
        zl,zr,zs=[prefixes[i]['state_targets'][-1] for i in (left,right,sham)]
        assert zl==zs, 'SHAM_TASK_STATE_DIFFERS'
        if zl[1]==zr[1]:
            excluded.append(dict(task=task,reason='NO_OLD_ANCHOR_STATE_DIFFERENCE'));continue
        tails=[family['cells'][i]['tail'] for i in (il,ir,ish)]
        assert prefixes[left]['features'][-1]==prefixes[right]['features'][-1]==prefixes[sham]['features'][-1]
        found=False
        for t,(a,b) in enumerate(zip(tails[0],tails[1])):
            if a['feature']!=b['feature']:
                excluded.append(dict(task=task,reason='SHORT_WINDOW_DIFFERS_BEFORE_REQUIRED_ACTION',step=t));found=True;break
            if a['target']==b['target']:continue
            if not admission['masks'][il][t] or not admission['masks'][ir][t]:
                excluded.append(dict(task=task,reason='BRANCH_ACTION_ABSTAINED_OR_DUPLICATE',step=t));found=True;break
            sham_valid=(len(tails[2])>t and all(tails[2][k]['feature']==tails[0][k]['feature']
                        and tails[2][k]['target']==tails[0][k]['target'] for k in range(t+1)))
            positive=left if zl[1] else right
            critical=next(k for k,z in enumerate(prefixes[positive]['state_targets']) if z[1])
            gap=len(prefixes[positive]['features'])-1+t-critical
            assert gap>8, 'EVENT_NOT_OUTSIDE_ORIGINAL_WINDOW'
            cases.append(dict(task_id=task,left_prefix=left,right_prefix=right,sham_prefix=sham,
                left_cell=il,right_cell=ir,sham_cell=ish,step=t,
                left_action=a['target'],right_action=b['target'],common_feature=a['feature'],
                sham_valid=sham_valid,critical_step=critical,old_event_gap=gap,
                both_teachers_within_full500=all(len(prefixes[p]['features'])-1+len(tails[k])<=500
                                               for k,p in enumerate((left,right)))))
            found=True;break
        if not found:excluded.append(dict(task=task,reason='NO_REQUIRED_ACTION_DIFFERENCE_IN_TESTED_SUFFIXES'))
    return dict(cases=cases,excluded=excluded)


def prepare():
    source=HERE.parent/'multifamily_v7/DATA.json'
    data=json.loads(source.read_text())
    rows=[]
    for family in data['families']:
        admission=admit(family)
        rows.append(dict(family_id=family['family_id'],split=family['split'],house=family['house'],
                         **admission,**branch_cases(family,admission)))
    result=dict(status='REAL_PASS_TEACHER_ADMISSION_BEFORE_TRAINING',source_data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        family_rows=rows,original_labels_modified=False,
        rule='Minimum actual suffix action count among existing PASS cells for each real history/task; STOP counts. Exact full-causal-context ties with conflicting actions abstain, same-target duplicates count once.',
        not_global_shortest_path=True,source_trajectory_pool_shared_by_all_arms=True,
        future_query_and_cost_absent_from_runtime=True,original_training_admission=False,
        branch_measure='First actual expert-action divergence while both complete recent-input prefixes still match. No forced immediate conflict; excluded cases/reasons retained.',
        scope='Exploratory teacher objective and policy-memory diagnostic, not a new benchmark success definition or method novelty claim.')
    with (HERE/'ACTOR_ADMISSION.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print({split:dict(families=sum(r['split']==split for r in rows),
        branch_pairs=sum(len(r['cases']) for r in rows if r['split']==split),
        admitted_actions=sum(r['admitted_actions'] for r in rows if r['split']==split),
        original_action_owners=sum(r['original_action_owners'] for r in rows if r['split']==split)) for split in ('fit','check')})


if __name__=='__main__':
    prepare()
