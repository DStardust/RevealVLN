"""All fixed seeds and matched natural routes; no checkpoint selection."""
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def compare(left,right):
    a={r['record_id']:r for r in left['ordinary']['rows'] if r['partition']=='check'}
    b={r['record_id']:r for r in right['ordinary']['rows'] if r['partition']=='check'}
    assert a.keys()==b.keys()
    rows=[]
    for key,x in a.items():
        y=b[key]
        assert (x['house'],x['decisions'],x['base_ce'],x['base_correct'])==(y['house'],y['decisions'],y['base_ce'],y['base_correct'])
        rows.append(dict(record_id=key,house=x['house'],decisions=x['decisions'],
            delta_ce=y['ce']-x['ce'],delta_correct=y['correct']-x['correct']))
    houses={}
    for row in rows:houses.setdefault(row['house'],[]).append(row)
    return dict(macro_delta_ce=statistics.mean(r['delta_ce'] for r in rows),
        delta_accuracy=sum(r['delta_correct'] for r in rows)/sum(r['decisions'] for r in rows),
        by_house={h:dict(routes=len(items),macro_delta_ce=statistics.mean(r['delta_ce'] for r in items),
            delta_accuracy=sum(r['delta_correct'] for r in items)/sum(r['decisions'] for r in items)) for h,items in houses.items()},
        matched_routes=rows)


def main():
    result=json.loads((HERE/'run_001/RESULT.json').read_text())
    config=json.loads((HERE/'PROTOCOL.json').read_text())
    expected={f'{a}_{s}' for a in config['arms'] for s in config['seeds']}
    assert set(result['runs'])==expected
    runs=result['runs']; comparisons={}
    for left,right in [('B1','B2'),('B1','Ours'),('B2','Ours')]:
        rows=[dict(seed=seed,**compare(runs[f'{left}_{seed}'],runs[f'{right}_{seed}'])) for seed in config['seeds']]
        comparisons[left+'_to_'+right]=dict(seeds=rows,
            mean_delta_accuracy=statistics.mean(r['delta_accuracy'] for r in rows),
            mean_delta_ce=statistics.mean(r['macro_delta_ce'] for r in rows))
    summaries={}
    for arm in config['arms']:
        natural=[runs[f'{arm}_{seed}']['ordinary']['summaries']['check'] for seed in config['seeds']]
        special=[runs[f'{arm}_{seed}']['summaries']['check'] for seed in config['seeds']]
        summaries[arm]=dict(natural_check=natural,
            natural_mean_accuracy=statistics.mean(r['accuracy'] for r in natural),
            natural_mean_ce=statistics.mean(r['macro_ce'] for r in natural),
            natural_mean_base_accuracy=statistics.mean(r['base_accuracy'] for r in natural),
            natural_all_seeds_lower_ce=all(r['macro_ce']<r['base_macro_ce'] for r in natural),
            special_check=special)
    output=dict(status='MATCHED_TRANSFER_TRAINING_REVIEWED',arms=summaries,comparisons=comparisons,
        fixed_seeds=config['seeds'],checkpoint_selection=False,original_special_training_admission=False,
        ordinary_check_houses=5,ordinary_check_routes=60,ordinary_check_decisions=3769,
        no_closed_loop_effect_claim=True,independent_blind_test=False,
        interpretation='Teacher-action accuracy/CE and SEE2 reader metrics remain diagnostic. Natural closed-loop SR and history-use mechanisms require their own tests.')
    with (HERE/'REVIEW.json').open('x') as stream:json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({a:{k:v for k,v in row.items() if k not in ('natural_check','special_check')} for a,row in summaries.items()}))


if __name__=='__main__':main()
