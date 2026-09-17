"""Read-only partial summary. Never presents partial SR as the full result."""
import json
from pathlib import Path
import time
import statistics

HERE=Path(__file__).resolve().parent;RUN=HERE/'run_001'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def records(path):
    try:
        with path.open() as f:
            for line in f:
                try:yield json.loads(line)
                except json.JSONDecodeError:continue
    except FileNotFoundError:return


def collect():
    p=read(HERE/'PROTOCOL.json');progress=read(RUN/'PROGRESS.json');launch=read(RUN/'LAUNCH_RESULT.json')
    final=read(RUN/'RESULT.json');rows=[]
    for lane in range(p.get('lanes',8)):
        rows.extend(x for x in records(RUN/'lanes'/f'lane_{lane:02d}'/'EPISODES.jsonl') if x.get('event')=='complete')
    n=len(rows);total=p.get('episode_count',1839);successes=sum(x['success'] for x in rows)
    stage='PREPARING_OR_LOADING'
    if progress:
        stage=progress['status']
        if stage=='EVALUATING' and time.time()-progress.get('unix',0)>180:stage='STALE_CHECK_LOGS'
    if launch:stage='AGGREGATING' if launch.get('status')=='COMPLETE' else launch.get('status','UNKNOWN')
    if final:stage=final['status']
    if (RUN/'AUDIT_FAILURE.json').exists():stage='AUDIT_FAILED_RESULTS_NOT_ADMITTED'
    by_house=[]
    for house in p.get('houses',[]):
        rr=[x for x in rows if x['house']==house]
        by_house.append(dict(house=house,completed=len(rr),total=p.get('house_counts',{}).get(house),
            successes=sum(x['success'] for x in rr),sr=sum(x['success'] for x in rr)/len(rr) if rr else None,
            spl=statistics.mean(x['spl'] for x in rr) if rr else None))
    elapsed=progress.get('wall_seconds');eta=(total-n)*elapsed/n if elapsed and n else None
    return dict(status=stage,unix=time.time(),checkpoint_updates=p.get('checkpoint_updates'),
        checkpoint_sha256=p.get('checkpoint_sha256'),completed=n,total=total,successful_completed=int(successes),
        partial_sr=successes/n if n else None,partial_spl=statistics.mean(x['spl'] for x in rows) if n else None,
        partial_ne=statistics.mean(x['navigation_error_m'] for x in rows) if n else None,
        partial_ndtw=statistics.mean(x['ndtw'] for x in rows) if n else None,
        full_sr=final.get('sr') if final.get('status')=='COMPLETE' else None,
        eta_seconds=eta,progress=progress,by_house=by_house,final=final,
        warning='中途均值受房屋顺序影响，不是完整基准成绩；最终须1839条+审计通过。')
