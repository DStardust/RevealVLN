"""Read-only causal action ambiguity and tested-continuation cost audit."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent


def audit_family(family, features):
    candidates=defaultdict(list)
    targets=defaultdict(list)
    for i,cell in enumerate(family['cells']):
        if not cell['mask'] or not cell['y'] or not cell['tail']:
            continue
        prefix=family['prefixes'][cell['prefix']]
        assert cell['tail'][0]['feature']==prefix['features'][-1]
        candidates[cell['prefix']].append(dict(cell=i,query=cell['query'],
            suffix_actions=len(cell['tail']),full_actions=len(prefix['features'])-1+len(cell['tail']),
            first_action=cell['tail'][0]['target']))
        digest=hashlib.sha256()
        for feature in prefix['features']:
            digest.update(features[feature]['key'].encode())
        for t,row in enumerate(cell['tail']):
            if t:
                digest.update(features[row['feature']]['key'].encode())
            if row['mask']:
                # Same complete causal observation/action history, not merely same short window.
                targets[digest.hexdigest()].append(dict(cell=i,step=row['step'],action=row['target']))
    ambiguous={key:rows for key,rows in targets.items() if len({x['action'] for x in rows})>1}
    costs=[]
    for prefix_index, choices in candidates.items():
        prefix=family['prefixes'][prefix_index]
        minimum=min(c['suffix_actions'] for c in choices)
        shortest=[x for x in choices if x['suffix_actions']==minimum]
        costs.append(dict(prefix=prefix_index,history_id=prefix['history_id'],task_id=prefix['task_id'],
            candidates=choices,shortest_cells=[x['cell'] for x in shortest],
            shortest_suffix_actions=minimum,
            mean_excess_suffix_actions=sum(x['suffix_actions']-minimum for x in choices)/len(choices),
            all_shortest_tied_first_actions=len({x['first_action'] for x in shortest})>1))
    common=[]
    for task in sorted({p['task_id'] for p in family['prefixes']}):
        indices=[i for i,p in enumerate(family['prefixes']) if p['task_id']==task]
        sets=[{x['query'] for x in candidates[i]} for i in indices]
        shared=set.intersection(*sets)
        within_budget=[q for q in shared if all(any(x['query']==q and x['full_actions']<=500
            for x in candidates[i]) for i in indices)]
        common.append(dict(task_id=task,history_count=len(indices),
                           common_successful_tested_queries=sorted(shared),
                           common_queries_also_within_full500_budget=sorted(within_budget)))
    return dict(family_id=family['family_id'],house=family['house'],partition=family['partition'],
        unique_full_causal_contexts=len(targets),ambiguous_full_causal_contexts=len(ambiguous),
        ambiguous_action_records=sum(len(x) for x in ambiguous.values()),
        action_records=sum(len(x) for x in targets.values()),
        first_ambiguous_contexts=dict(list(ambiguous.items())[:3]),prefix_costs=costs,
        history_agnostic_tested_continuations=common)


def main():
    began=time.monotonic()
    data=json.loads((HERE/'DATA.json').read_text())
    rows=[audit_family(f,data['features']) for f in data['families']]
    summary={}
    for partition in sorted({r['partition'] for r in rows}):
        selected=[r for r in rows if r['partition']==partition]
        prefixes=[p for r in selected for p in r['prefix_costs']]
        tasks=[p for r in selected for p in r['history_agnostic_tested_continuations']]
        summary[partition]=dict(families=len(selected),
            unique_full_causal_contexts=sum(r['unique_full_causal_contexts'] for r in selected),
            ambiguous_full_causal_contexts=sum(r['ambiguous_full_causal_contexts'] for r in selected),
            ambiguous_action_records=sum(r['ambiguous_action_records'] for r in selected),
            action_records=sum(r['action_records'] for r in selected),
            prefixes=len(prefixes),prefixes_with_more_costly_successful_demonstrations=sum(p['mean_excess_suffix_actions']>0 for p in prefixes),
            prefix_macro_excess_suffix_actions=sum(p['mean_excess_suffix_actions'] for p in prefixes)/len(prefixes),
            task_families=len(tasks),tasks_with_common_successful_query=sum(bool(t['common_successful_tested_queries']) for t in tasks),
            tasks_with_common_query_within500=sum(bool(t['common_queries_also_within_full500_budget']) for t in tasks))
    result=dict(status='READ_ONLY_TEACHER_ACTION_COST_AUDIT',data_sha256=hashlib.sha256((HERE/'DATA.json').read_bytes()).hexdigest(),
        seconds=time.monotonic()-began,summary=summary,families=rows,
        interpretation='Different next actions under identical full causal inputs can each belong to a real successful suffix. This is valid multimodal supervision, not fabricated/wrong labels. Equal imitation of all passing suffixes need not reward avoiding redundant revisits. Shared success is only over the finite tested continuation set, not global state equivalence or universal solvability.',
        planned_fix_if_supported='A separate matched teacher-cost experiment may admit the shortest actually executed PASS suffix for each history/task, share the same source pool across all arms, preserve original labels, and abstain on tied conflicting experts. This is a teacher objective repair, not novelty or measured benefit.',
        modified_labels=0,new_optimizer_updates=0,new_navigation_episodes=0,original_training_admission=False)
    with (HERE/'ACTION_TARGET_AUDIT.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
