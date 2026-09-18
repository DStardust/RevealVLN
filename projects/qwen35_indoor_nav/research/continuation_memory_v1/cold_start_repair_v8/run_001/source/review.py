"""All fixed seeds, paired families and honest separation of auxiliary/action gains."""
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def compare(left,right):
    a={row['family_id']:row for row in left['families'] if row['split']=='check'}
    b={row['family_id']:row for row in right['families'] if row['split']=='check'}
    assert a.keys()==b.keys()
    rows=[]
    for key in a:
        x,y=a[key],b[key]
        assert (x['house'],x['action_owners'],x['targets'])==(y['house'],y['action_owners'],y['targets'])
        rows.append(dict(family_id=key,house=x['house'],delta_action_ce=y['action_ce']-x['action_ce'],
            delta_action_accuracy=y['action_accuracy']-x['action_accuracy'],
            delta_query_accuracy=y['query_accuracy']-x['query_accuracy']))
    return dict(delta_action_ce=statistics.mean(x['delta_action_ce'] for x in rows),
        delta_action_accuracy=statistics.mean(x['delta_action_accuracy'] for x in rows),
        delta_query_accuracy=statistics.mean(x['delta_query_accuracy'] for x in rows),families=rows)


def main():
    read=lambda path:json.loads(path.read_text())
    config=read(HERE/'PROTOCOL.json');result=read(HERE/'run_001/RESULT.json');data=read(HERE.parent/'multifamily_v7/DATA.json')
    runs=result['runs']
    expected={f'{arm}_{seed}' for arm in config['arms'] for seed in config['seeds']}
    assert set(runs)==expected
    summaries={}
    for left,right in [('B1','B2'),('B1','Ours'),('B2','Ours')]:
        rows=[dict(seed=seed,**compare(runs[f'{left}_{seed}'],runs[f'{right}_{seed}'])) for seed in config['seeds']]
        summaries[left+'_to_'+right]=dict(seeds=rows,
            mean_delta_action_ce=statistics.mean(x['delta_action_ce'] for x in rows),
            mean_delta_action_accuracy=statistics.mean(x['delta_action_accuracy'] for x in rows),
            mean_delta_query_accuracy=statistics.mean(x['delta_query_accuracy'] for x in rows) if left!='B1' else None,
            seed_sd_delta_action_accuracy=statistics.stdev(x['delta_action_accuracy'] for x in rows),
            all_seeds_lower_action_ce=all(x['delta_action_ce']<0 for x in rows))
    lookup_correct=lookup_cells=0
    for family in data['families']:
        if family['split']!='check':continue
        groups={}
        for cell in family['cells']:
            if not cell['mask']:continue
            key=(family['prefixes'][cell['prefix']]['task_id'],cell['query'])
            groups.setdefault(key,[]).append(cell['y'])
        lookup_correct+=sum(max(sum(values),len(values)-sum(values)) for values in groups.values())
        lookup_cells+=sum(map(len,groups.values()))
    output=dict(status='MATCHED_METHOD_COMPARISON_REVIEWED',comparisons=summaries,
        no_history_task_query_lookup_upper_bound=dict(correct=lookup_correct,cells=lookup_cells),
        heldout_house_count=2,heldout_family_count=12,fit_house_count=6,fit_family_count=13,
        independent_blind_test=False,closed_loop_benefit_not_determined_by_this_report=True,
        B1_query_reader_untrained_and_excluded_from_auxiliary_comparison=True,
        generalization_limit='Two held-out memory-training houses from existing exposed backbone-FIT assets; correlated families; original training_admission=false',
        model_selection='All three prespecified seeds and fixed final600 updates; no best-seed or best-step selection')
    with (HERE/'REVIEW.json').open('x') as stream:json.dump(output,stream,indent=2)
    print(json.dumps({k:v for k,v in output.items() if k!='comparisons'},indent=2))
    for key,value in summaries.items():print(key,{k:v for k,v in value.items() if k!='seeds'})


if __name__=='__main__':main()
