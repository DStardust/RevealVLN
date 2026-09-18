"""Read-only diagnosis: history-identifying pairs, query coverage and action ages."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def main():
    data=json.loads((HERE.parent/'multifamily_v7/DATA.json').read_text())
    result=json.loads((HERE/'run_001/RESULT.json').read_text())
    families={f['family_id']:f for f in data['families']}
    fit_tokens={t for f in families.values() if f['split']=='fit' for q in f['queries'] for t in q}
    check_tokens={t for f in families.values() if f['split']=='check' for q in f['queries'] for t in q}
    ages={}
    for split in ('fit','check'):
        steps=[x['step'] for f in families.values() if f['split']==split for cell in f['cells'] for x in cell['tail'] if x['mask']]
        ages[split]=dict(action_owners=len(steps),zero_based_min=min(steps),
                         zero_based_median=statistics.median(steps),zero_based_max=max(steps))
    runs={}
    for key,run in result['runs'].items():
        if run['arm']=='B1':continue  # Its query reader has no supervision.
        groups={}
        for measured in run['families']:
            family=families[measured['family_id']]
            group=groups.setdefault(family['split'],dict(opposing_pairs=0,both_correct=0,
                query_seen_correct=0,query_seen_cells=0,query_novel_correct=0,query_novel_cells=0))
            cells=family['cells'];prefixes=family['prefixes']
            predictions=[p>.5 for p in measured['probabilities']]
            for i,cell in enumerate(cells):
                if not cell['mask']:continue
                coverage='query_seen' if set(family['queries'][cell['query']])<=fit_tokens else 'query_novel'
                group[coverage+'_cells']+=1
                group[coverage+'_correct']+=int(predictions[i]==bool(cell['y']))
                left=prefixes[cell['prefix']]
                if left['history_id']!='H_A':continue
                for j,other in enumerate(cells):
                    right=prefixes[other['prefix']]
                    if (other['mask'] and right['history_id']=='H_B' and left['task_id']==right['task_id']
                            and cell['query']==other['query'] and cell['y']!=other['y']):
                        assert left['features'][-1]==right['features'][-1]
                        group['opposing_pairs']+=1
                        group['both_correct']+=int(predictions[i]==bool(cell['y']) and predictions[j]==bool(other['y']))
        runs[key]=groups
    report=dict(status='READ_ONLY_DIAGNOSTIC',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((HERE.parent/'multifamily_v7/DATA.json').read_bytes()).hexdigest(),
        action_supervision_ages=ages,check_only_query_tokens=[data['query_vocabulary'][t] for t in sorted(check_tokens-fit_tokens)],
        results=runs,interpretation='Observed coverage and paired prediction evidence; neither causal attribution of failures nor permission for score-selected retraining',gpu_used=False)
    with (HERE/'MECHANISM_DIAGNOSIS.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
