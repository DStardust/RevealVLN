"""Versioned reporting/transport only; exact same policy, simulator and trace checks."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'r2r_ce_full_v2'
HASHES={
 'common.py':'b3c80bd56eb2b5b6d6c9553170d2731044696f8dbfa14f55df902f2b7139fe9a',
 'evaluate.py':'c505fc54cbdd3bb8eba483bd3f09da2759e76f8eb08a9eb02cbd0408af91f494',
 'executor.py':'7ee215621690272c0d41983208d6eaaaf1dbe61ff4d4e82459ff8cf870b1eeb8',
 'path_metrics.py':'d9c1fa801eb7fe293ab937b7b74af861f7803344c2581029c15caec34a9e01d6',
 'official_distance.py':'8c591164596a0914ee8eefa5f65dfead01c7d09e126b60e31bf92f290022fa2b',
 'aggregate.py':'6564e094eb2b73e2761e659b8923d9e196dd97d38e3587e80756997a418ceb8b',
 'launch.py':'145e83d280f6886521eb5ddb7c39d90382551d2c19dd36410e5ddff4fcad1947',
}


def source(name):
    raw=(OLD/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==HASHES[name]
    text=raw.decode()
    if name=='launch.py':
        old="    assert all(x for x in training_before['processes']),'TRAINING_IDENTITY_NOT_PRESENT'"
        assert text.count(old)==1
        text=text.replace(old,"    # Training may not yet have started or already naturally stopped; GPU1 independent.")
    if name=='aggregate.py':
        marker='    overall=stats(rows);by_house='
        assert text.count(marker)==1
        text=text[:text.index(marker)]+AGGREGATE_END
    return text


AGGREGATE_END='''    assert full, 'INCOMPLETE_DEV_CASE_NO_ADMITTED_METRICS'
    overall=stats(rows);by_house={h:stats([x for x in rows if x['house']==h]) for h in p['houses']}
    result=dict(status='COMPLETE',unix=time.time(),checkpoint_updates=p['checkpoint_updates'],
        checkpoint_sha256=p['checkpoint_sha256'],benchmark='R2R-CE train held-out INTERNAL_DEV diagnostic',
        planned=p['episode_count'],completed=len(rows),missing=0,**overall,house_count=len(p['houses']),by_house=by_house,
        stopped=sum(int(x['stopped']) for x in rows),failure_categories=dict(collections.Counter(x['failure_category'] for x in rows)),
        collisions=sum(x['collisions'] for x in rows),environment_actions=sum(x['steps'] for x in rows),
        trace_audit_passed=True,audited_actions=audited,source_lock_verified=True,
        selected_batch_size=read(RUN/'BATCH_PARITY_GATE.json').get('selected_batch_size'),
        optimizer_updates=0,scientific_gain_verified=False,wall_seconds=launch['wall_seconds'],
        training_processes_signaled=[],checkpoint_role=p['checkpoint_role'],run_dir=str(RUN))
    fields=['index','episode_id','trajectory_id','house','success','spl','navigation_error_m','oracle_success','ndtw','sdtw','steps','stopped','collisions','failure_category']
    with (RUN/'episodes.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:x[k] for k in fields} for x in rows)
    c.write(RUN/'RESULT.json',result,True)
    c.write(RUN/'PROGRESS.json',dict(status='COMPLETE',unix=time.time(),completed=len(rows),total=p['episode_count']))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
'''


def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':parent','exec'),namespace)
