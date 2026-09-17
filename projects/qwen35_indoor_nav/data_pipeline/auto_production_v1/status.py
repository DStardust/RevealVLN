"""Read-only live queue status. GPU utilization is not a production counter."""
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
LINE=ROOT/'projects/qwen35_indoor_nav'


def read(path):
    return json.loads(path.read_text())


def live_identity(identity):
    pid=identity['pid']
    try:
        argv=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
        return any(str(HERE/'queue.py').encode() in value or value.endswith(b'auto_production_v1/queue.py') for value in argv)
    except FileNotFoundError:
        return False


def production_progress(job):
    paths=[Path(c['path']).parent/'PROGRESS.json' for c in job['completion']]
    config=Path(job['command'][3]).parent/'PREPARED_CONFIG.json'
    if config.exists():
        paths += [ROOT/value/'PROGRESS.json' for value in read(config).get('output_roots',{}).values()]
    result=[]
    for path in paths:
        if path.resolve()!=path or not path.is_relative_to(LINE) or not path.exists():
            continue
        try:
            value=read(path)
            result.append({'path':str(path),'age_seconds':round(time.time()-path.stat().st_mtime,1),
                'provisional':True,'counts':{k:v for k,v in value.items() if k in
                    ('completed','target','audited_routes','audited_instruction_conditioned_decisions',
                     'complete_traces','actual_actions','physical_families','collisions')}})
        except (json.JSONDecodeError,FileNotFoundError):
            result.append({'path':str(path),'state':'ATOMIC_UPDATE_IN_PROGRESS'})
    return result


def inspect():
    result=[]
    for plan_path in sorted(HERE.glob('*/PLAN.json')):
        plan=read(plan_path)
        for gpu in sorted({j['gpu'] for j in plan['jobs']}):
            lane=plan_path.parent/f'lane_gpu_{gpu}'
            item={'plan':plan_path.parent.name,'gpu':gpu,'planned_jobs':sum(j['gpu']==gpu for j in plan['jobs']),
                  'queue_alive':False,'state':'NOT_LAUNCHED','last_event':None}
            if (lane/'IDENTITY.json').exists():
                identity=read(lane/'IDENTITY.json');item['queue_alive']=live_identity(identity);item['pid']=identity['pid']
                item['state']='RUNNING' if item['queue_alive'] else 'NO_LIVE_QUEUE_CHECK_RESULT'
            if (lane/'EVENTS.jsonl').exists():
                raw=(lane/'EVENTS.jsonl').read_bytes();complete=raw.split(b'\n')[:-1]
                if complete:
                    item['last_event']=json.loads(complete[-1]);item['last_event_age_seconds']=round(time.time()-item['last_event'].get('unix',time.time()),1)
                    ident=item['last_event'].get('job')
                    current=next((j for j in plan['jobs'] if j['id']==ident),None)
                    if current is not None:item['production_progress']=production_progress(current)
            if (lane/'RESULT.json').exists():
                final=read(lane/'RESULT.json');item['state']='STOPPED_ERROR' if final['error'] else 'CLOSED'
                if not final['error'] and any(value in ('PENDING','RESERVED_NOT_LAUNCHED_DRAIN_OR_DEADLINE') for value in final['states'].values()):
                    item['state']='STOPPED_WITH_PENDING_JOBS_NOT_COMPLETE'
                item['result']=final
            result.append(item)
    return result


if __name__=='__main__':
    print(json.dumps({'checked_unix':time.time(),'queues':inspect(),
        'note':'Producer progress is in each registered run; alive queue alone does not establish accepted data.'},indent=2))
