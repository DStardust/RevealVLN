"""Read-only 18766 revision: completed budget and honest matched navigation panels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
OLD = HERE.parent / 'monitor_charts_expanded_v1'
for name, digest in {
    'server.py':'04fc1f4a0e4726d794e3779a2f0a9e42ea5249a791ffc3ecd65f93aa3867520d',
    'server_r1.py':'79801423b6817e695428438cbca8ebd74fd5195cba9a54fc6155edd510f3753f',
}.items(): assert hashlib.sha256((OLD/name).read_bytes()).hexdigest() == digest
s = importlib.util.spec_from_file_location('targeted_monitor_parent', OLD/'server_r1.py')
parent = importlib.util.module_from_spec(s); s.loader.exec_module(parent)
m = parent.m
BENCH = LINE / 'closed_loop_bench'
REVIEW = LINE / 'reviews/Q35N_ORDINARY_TARGETED_REPAIR_V1'


def read(p):
    try: return json.loads(p.read_text())
    except (OSError, ValueError): return {}


def eval_state(name):
    case = BENCH/name; run = case/'run_001'
    result = read(run/'RESULT.json'); progress = read(run/'PROGRESS.json')
    failure = read(run/'LAUNCH_FAILURE.json') or read(run/'INFERENCE_FAILURE.json') or read(run/'AUDIT_FAILURE.json')
    launch = read(run/'LAUNCH_RESULT.json')
    if result.get('status') == 'COMPLETE': status = 'COMPLETE'
    elif failure or launch.get('status') in ('SERVICE_FAILED','RESOURCE_CENSORED'): status = 'FAILED_OR_CENSORED'
    elif progress: status = progress.get('status','UNKNOWN')
    elif (run/'PROCESS.json').exists(): status = 'LOADING'
    elif (case/'SOURCE_LOCK.json').exists(): status = 'PREPARED'
    else: status = 'NOT_STARTED'
    age = max(0,time.time()-progress['unix']) if 'unix' in progress else None
    return dict(status=status, completed=result.get('completed',progress.get('completed',0)), total=100,
        result=result or None, progress=progress, error=failure or None, age_seconds=age,
        heartbeat_stale=status=='EVALUATING' and age is not None and age>90)


def classify_training(result, lease):
    completed = (result.get('status')=='STOPPED' and result.get('stop')==['BUDGET:max_updates']
                 and result.get('cursor',{}).get('updates')==4000)
    return dict(completed_budget=completed, stop_reason=result.get('stop'),
        holder_restoration_verified=lease.get('holders_restored') is True,
        latest_checkpoint=result.get('latest_checkpoint'), new_updates=result.get('cursor',{}).get('updates'),
        actions=result.get('global_decisions'), automatic_training=False)


def collect(include_gpu=True):
    d = m.collect(include_gpu=include_gpu)
    result = read(LINE/'sft_acceptance/ordinary_expanded_v1/formal/attempt_001/RESULT.json')
    lease = read(LINE/'sft_acceptance/ordinary_expanded_v1/lease_v1/LEASE_RESULT.json')
    training = classify_training(result,lease)
    d.update(monitor_version='ordinary_targeted_v1', training_completion=training,
             next_stage='MATCHED_BATCH1_THEN_SEPARATE_CAUSAL_GUARD_DIAGNOSTIC',
             automatic_training=False)
    if training['completed_budget']:
        d['display_state'] = '已完成 4,000 步预算' if training['holder_restoration_verified'] else '训练完成，资源恢复待核实'
        d['checkpoint_name'] = Path(training['latest_checkpoint']).name
        d['decisions'] = training['actions']
    d['navigation'] = {
        k:eval_state(v) for k,v in dict(before='ordinary_expanded_dev_before_v1',
        after_batch8='ordinary_expanded_dev_after_v1', matched='ordinary_expanded_dev_after_single_v1',
        guard='ordinary_visual_stall_guard_v1').items()}
    for stage in ('GUARD','MATCHED','INITIAL'):
        diagnosis = read(REVIEW/f'ANALYSIS_{stage}.json')
        if diagnosis:
            d['failure_diagnosis'] = dict(stage=stage,pairs=diagnosis['pairs'],cases={k:{a:v for a,v in row.items() if a!='episodes'} for k,row in diagnosis['cases'].items()})
            break
    d['targeted_workflow'] = read(REVIEW/'WORKFLOW_STATUS.json')
    return d


def html():
    text = m.html().decode()
    marker = '<section style="padding:16px;margin:16px;border:1px solid #50637d;border-radius:12px">'
    assert text.count(marker)==1
    text = text[:text.index(marker)] + '</html>'
    assert text.count('</main>')==1
    text = text.replace('</main>',(HERE/'panel.html').read_text()+'</main>')
    lines = text.splitlines()
    refresh_lines = [i for i,line in enumerate(lines) if line.startswith('async function refresh(){')]
    assert len(refresh_lines)==1
    lines[refresh_lines[0]] = (HERE/'refresh.js').read_text()
    text = '\n'.join(lines)
    # Historical logs naturally age after a finite pilot; retain raw age/stale fields.
    text = text.replace("d.run_name,stale?'warn':'good'", "d.run_name,d.training_completion.completed_budget?'good':stale?'warn':'good'")
    return text.encode()


m.r.b.collect = collect


class Handler(m.Handler):
    def do_GET(self):
        if self.path=='/': self.respond(200,html(),'text/html; charset=utf-8')
        else: super().do_GET()
    do_HEAD = do_GET


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);p.add_argument('--check',action='store_true');a=p.parse_args()
    html()
    if a.check:
        d=collect(False)
        print(json.dumps({k:d[k] for k in ('monitor_version','display_state','training_completion','navigation')},ensure_ascii=False))
    else:
        with m.r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
            server.daemon_threads=True;server.serve_forever(poll_interval=.5)
