"""Matched-window training-loss diagnosis, separate from navigation admission."""
import csv
import json
import math
from pathlib import Path
import statistics
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]


def records(name):
    path=LINE/'sft_acceptance'/name/'formal/attempt_001/PROGRESS.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    result={r['cursor']['updates']:r for r in rows}
    assert len(result)==len(rows) and set(result)==set(range(4020,8001,20))
    return result


def main():
    a=records('ordinary_expanded_continue_v2');b=records('ordinary_expanded_low_lr_v3')
    rows=[]
    for step in sorted(a):
        high,low=a[step],b[step]
        assert high['global_plan_decisions']==low['global_plan_decisions']
        assert high['metrics']['action_target_counts']==low['metrics']['action_target_counts']
        assert math.isclose(low['metrics']['lr']/high['metrics']['lr'],.1,rel_tol=1e-12)
        x,y=high['metrics']['mean_ce'],low['metrics']['mean_ce']
        rows.append(dict(updates=step,global_decisions=high['global_plan_decisions'],high_lr_ce=x,low_lr_ce=y,delta_low_minus_high=y-x))
    result=dict(status='COMPLETE_MATCHED_TRAINING_DIAGNOSTIC',unix=time.time(),windows=len(rows),
        identical_decision_and_target_counts=True,lr_ratio_low_to_high=.1,
        mean_of_window_CE_high=statistics.mean(x['high_lr_ce'] for x in rows),
        mean_of_window_CE_low=statistics.mean(x['low_lr_ce'] for x in rows),
        median_paired_window_CE_delta=statistics.median(x['delta_low_minus_high'] for x in rows),
        lower_CE_windows=sum(x['delta_low_minus_high']<0 for x in rows),
        metric_scope='Unweighted summary of same 20-update windows; not exact epoch weighted CE, held-out CE or navigation SR',
        navigation_gain_verified=False)
    with (HERE/'MATCHED_TRAINING_LOSS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    with (HERE/'matched_training_windows.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
