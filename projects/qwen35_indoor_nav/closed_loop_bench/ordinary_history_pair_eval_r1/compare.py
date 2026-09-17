"""Complete paired INTERNAL_DEV100 comparison, never a full-val SR40 success claim."""
import collections
import csv
import json
from pathlib import Path
import random
import statistics
import time
HERE=Path(__file__).resolve().parent
CASES={'control_recent2':'ordinary_history2_dev_r1','treatment_prefix8':'ordinary_history8_dev_r1'}
BASE=HERE.parent/'ordinary_expanded_dev_after_single_v1'
def read(p):return json.loads(p.read_text())
def episodes(folder):
    with (folder/'run_001/episodes.csv').open() as stream:raw=list(csv.DictReader(stream))
    assert len(raw)==100
    rows={int(x['index']):dict(episode_id=x['episode_id'],house=x['house'],
          sr=float(x['success']),spl=float(x['spl']),ndtw=float(x['ndtw'])) for x in raw}
    assert sorted(rows)==list(range(100))
    return rows
def comparison(candidate,baseline):
    assert all(candidate[i]['episode_id']==baseline[i]['episode_id'] and candidate[i]['house']==baseline[i]['house'] for i in candidate)
    delta={k:statistics.mean(candidate[i][k]-baseline[i][k] for i in candidate) for k in ('sr','spl','ndtw')}
    houses=sorted({x['house'] for x in candidate.values()})
    assert len(houses)==5
    groups={h:[i for i,x in candidate.items() if x['house']==h] for h in houses}
    def subset(ids):return {k:statistics.mean(candidate[i][k]-baseline[i][k] for i in ids) for k in delta}
    rng=random.Random(1209);draws={k:[] for k in delta}
    for _ in range(2000):
        ids=[i for h in rng.choices(houses,k=len(houses)) for i in groups[h]]
        d=subset(ids)
        for key in draws:draws[key].append(d[key])
    intervals={}
    for key,values in draws.items():
        values.sort();intervals[key]=[values[49],values[1949]]
    return dict(delta=delta,wins=sum(candidate[i]['sr']>baseline[i]['sr'] for i in candidate),
        losses=sum(candidate[i]['sr']<baseline[i]['sr'] for i in candidate),
        by_house={h:subset(ids) for h,ids in groups.items()},
        leave_one_house_out={h:subset([i for i in candidate if candidate[i]['house']!=h]) for h in houses},
        house_bootstrap95=intervals,bootstrap_draws=2000,bootstrap_seed=1209,
        pass_development_gate=delta['sr']>0 and delta['spl']>=0 and delta['ndtw']>=-.01)
def main():
    loaded={};metrics={}
    for arm,name in CASES.items():
        folder=HERE.parent/name;result=read(folder/'run_001/RESULT.json');launch=read(folder/'run_001/LAUNCH_RESULT.json')
        assert result['status']=='COMPLETE' and result['completed']==result['planned']==100 and result['missing']==0
        assert result['history_arm']==arm and result['selected_history_audited_actions']==result['audited_actions']
        assert launch['status']=='COMPLETE' and launch['returncode']==0 and not launch['cleanup']['remaining']
        assert not launch['foreign_processes_signaled'] and not launch['gpus_after'][1]['contexts']
        loaded[arm]=episodes(folder);metrics[arm]=result
    base=episodes(BASE)
    paired=comparison(loaded['treatment_prefix8'],loaded['control_recent2'])
    versus_base={arm:comparison(rows,base) for arm,rows in loaded.items()}
    eligible=[arm for arm in loaded if versus_base[arm]['pass_development_gate'] and (arm=='control_recent2' or paired['pass_development_gate'])]
    eligible.sort(key=lambda arm:(-metrics[arm]['sr'],-metrics[arm]['spl'],arm!='control_recent2'))
    selected=eligible[0] if eligible else None
    result=dict(status='COMPLETE_INTERNAL_DEV_ONLY',unix=time.time(),scope='100 already-exposed internal train-split heldout-house episodes',
        ordinary_only=True,metrics=metrics,prefix8_vs_recent2=paired,versus_old_best4k=versus_base,
        selected_candidate_for_separate_full_protocol=selected,
        full_val1839_sr=None,sr40_goal_achieved=False,novel_contribution_claimed=False,
        next_action='freeze separate complete1839 compute protocol' if selected else 'retain failures and diagnose ordinary base recipe; no automatic new training')
    with (HERE/'COMPARISON_RESULT.json').open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
if __name__=='__main__':main()

