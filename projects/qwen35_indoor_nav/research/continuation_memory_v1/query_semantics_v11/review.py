"""All fixed B2/Ours seeds; mechanism and navigation claims stay separate."""
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c


def paired(measured, family):
    predicted=[p>.5 for p in measured['probabilities']]
    rows=[]
    for i,left in enumerate(family['cells']):
        prefix=family['prefixes'][left['prefix']]
        if not left['mask'] or prefix['history_id']!='H_A':continue
        for j,right in enumerate(family['cells']):
            donor=family['prefixes'][right['prefix']]
            if (right['mask'] and donor['history_id']=='H_B' and prefix['task_id']==donor['task_id']
                    and left['query']==right['query'] and left['y']!=right['y']):
                assert prefix['features'][-1]==donor['features'][-1]
                signed_gap=(measured['probabilities'][i]-measured['probabilities'][j])*(1 if left['y'] else -1)
                rows.append(dict(task_id=prefix['task_id'],query=left['query'],
                    both_correct=predicted[i]==bool(left['y']) and predicted[j]==bool(right['y']),
                    signed_probability_gap=signed_gap))
    return rows


def main():
    config=c.read(HERE/'PROTOCOL.json')
    result=c.read(HERE/'run_001/RESULT.json')
    assert c.read(HERE/'run_001/LAUNCH_RESULT.json')['status']=='COMPLETE'
    assert c.read(HERE/'CHECKPOINT_AUDIT.json')['runtime_parameters_updated_as_declared']
    data=c.read(HERE.parent/'multifamily_v7/DATA.json')
    families={f['family_id']:f for f in data['families']}
    assert set(result['runs'])=={f'{arm}_{seed}' for arm in config['arms'] for seed in config['seeds']}
    rows={}
    for key,run in result['runs'].items():
        assert run['updates']==600
        summaries={}
        for split in ('fit','check'):
            selected=[m for m in run['families'] if m['split']==split]
            details=[]
            for measured in selected:
                pairs=paired(measured,families[measured['family_id']])
                details.append(dict(family_id=measured['family_id'],house=measured['house'],
                    both_correct=sum(p['both_correct'] for p in pairs),opposing_pairs=len(pairs),
                    positive_history_gap=sum(p['signed_probability_gap']>0 for p in pairs),
                    query_correct=measured['query_correct'],query_cells=measured['query_cells'],pairs=pairs))
            houses=sorted({m['house'] for m in details})
            summaries[split]=dict(families=details,by_house={h:dict(
                both_correct=sum(m['both_correct'] for m in details if m['house']==h),
                opposing_pairs=sum(m['opposing_pairs'] for m in details if m['house']==h)) for h in houses},
                both_correct=sum(m['both_correct'] for m in details),opposing_pairs=sum(m['opposing_pairs'] for m in details),
                query_correct=sum(m['query_correct'] for m in details),query_cells=sum(m['query_cells'] for m in details))
            summaries[split]['query_accuracy']=summaries[split]['query_correct']/summaries[split]['query_cells']
        rows[key]=dict(**summaries,ordinary=run['ordinary']['summaries'],
                       old_write_gradient=c.read(HERE/'run_001'/f'{key}_GRADIENT.json'))
    pair_deltas=[rows[f'Ours_{seed}']['check']['both_correct']-rows[f'B2_{seed}']['check']['both_correct'] for seed in config['seeds']]
    accuracy_deltas=[rows[f'Ours_{seed}']['check']['query_accuracy']-rows[f'B2_{seed}']['check']['query_accuracy'] for seed in config['seeds']]
    signal=all(delta>=0 for delta in pair_deltas) and statistics.mean(pair_deltas)>0 and statistics.mean(accuracy_deltas)>=0
    review=dict(status='MATCHED_CPU_PILOT_COMPLETE',runs=rows,source_sha256=c.sha(Path(__file__)),
        frozen_protocol_sha256=c.sha(HERE/'PROTOCOL.json'),optimizer_updates=result['optimizer_updates'],
        cpu_wall_seconds=c.read(HERE/'run_001/LAUNCH_RESULT.json')['cpu_wall_seconds'],gpu_hours=0,
        ours_minus_b2_check_both_correct=pair_deltas,ours_minus_b2_check_accuracy=accuracy_deltas,
        preregistered_exploratory_signal=signal,policy_gain='NOT_MEASURED',adopted=False,publication_ready=False,
        limitation='Previously exposed12 CHECK families in2houses; three seed repeats are not independent data. Original debug-only certificate limitations remain. Auxiliary/readability signal cannot establish closed-loop use, independent generalization, or publication readiness.')
    c.write(HERE/'REVIEW.json',review,True)
    print({k:v for k,v in review.items() if k!='runs'})


if __name__=='__main__':main()
