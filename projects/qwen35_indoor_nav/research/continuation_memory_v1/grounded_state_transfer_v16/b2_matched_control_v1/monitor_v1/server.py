"""Read-only, localhost dashboard for the eight-GPU B2 matched control. Standard library only."""
import argparse
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent


def read(path):
    return json.loads(path.read_text())


def optional(path, default=None):
    return read(path) if path.exists() else default


def tail(path, limit=6000):
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode('utf-8', errors='replace')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Reader:
    def __init__(self, run, job):
        self.run, self.job = run, job
        self.verified = {}

    def group(self, session, path, registry):
        record = read(path)
        condition = record['condition']
        sealpath = session / f'STATE_SEAL_{condition:03d}.json'
        if not sealpath.exists():
            return None
        stamp = (path.stat().st_mtime_ns, sealpath.stat().st_mtime_ns)
        if path in self.verified and self.verified[path][0] == stamp:
            return self.verified[path][1]
        seal = read(sealpath)
        slots = [s for s in registry['slots'] if s['condition'] == condition]
        ranks = [s['rank'] for s in slots]
        if record['ranks'] != ranks or seal['completed_ranks'] != ranks:
            raise ValueError('Incomplete registered model group')
        if not seal['base_unchanged'] or not seal['heads_unchanged']:
            raise ValueError('Model state changed')
        audits = record['audits']
        if len(audits) != 6 or any(not a['input_prefix_matched'] or not a['action_prefix_matched'] or a['argmax_flip_count'] for a in audits):
            raise ValueError('Pair prefix audit failed')
        # Hash a newly sealed group once, never reread all large trajectories every refresh.
        # The formal final review independently rechecks every group.
        for name, expected in record['files'].items():
            file = (session / name).resolve()
            if not file.is_relative_to(session.resolve()) or sha(file) != expected:
                raise ValueError('Group file SHA mismatch: ' + name)
        rows = []
        for slot in slots:
            name = f"rollouts/{slot['rank']:04d}/TASK_RESULT.json"
            if name not in record['files']:
                raise ValueError('Missing task result from sealed manifest')
            result = read(session / name)
            if result['model'] != slot['model'] or result['rank'] != slot['rank'] or result['condition'] != registry['conditions'][condition]:
                raise ValueError('Result does not match registered slot')
            if result['safe_v16_label'] not in ('PASS', 'FAIL', 'UNKNOWN'):
                raise ValueError('Unrecognized label')
            rows.append(dict(slot, **registry['conditions'][condition],
                label=result['safe_v16_label'], decisions=result['total_decisions'],
                collisions=result['collisions'], stopped=result['stopped'],
                exhausted=result['budget_exhausted']))
        value = dict(condition=condition, rows=rows, audits=audits)
        self.verified[path] = stamp, value
        return value

    def snapshot(self):
        run=self.run;registry=read(run/'EVALUATION_REGISTRY.json');status=read(run/'STATUS.json');cfg=read(run/'PROTOCOL.json')
        groups,errors={},[]
        for session in sorted((run/'evaluate').glob('session_*')):
            for path in sorted(session.glob('GROUP_*.json')):
                try:
                    group=self.group(session,path,registry)
                    if group is None:continue
                    if group['condition'] in groups:raise ValueError('Duplicate sealed condition')
                    groups[group['condition']]=group
                except (OSError,ValueError,KeyError,TypeError) as exc:errors.append(f'{session.name}/{path.name}: {exc}')
        rows=[r for g in groups.values() for r in g['rows']];metrics=[]
        for endpoint in ('main','control'):
            for arm in cfg['arms']:
                selected=[r for r in rows if r['endpoint']==endpoint and r['arm']==arm]
                planned=sum(s['arm']==arm and registry['conditions'][s['condition']]['endpoint']==endpoint for s in registry['slots'])
                counts=Counter(r['label'] for r in selected);n=len(selected)
                metrics.append(dict(endpoint=endpoint,arm=arm,complete=n,planned=planned,passed=counts['PASS'],failed=counts['FAIL'],unknown=counts['UNKNOWN'],not_run=planned-n,
                    partial_rate=counts['PASS']/n if n else None,identification_lower=counts['PASS']/planned,
                    identification_upper=(counts['PASS']+counts['UNKNOWN']+planned-n)/planned,
                    collisions=sum(r['collisions'] for r in selected),exhausted=sum(r['exhausted'] for r in selected)))
        values={(r['condition'],r['seed'],r['arm']):r['label'] for r in rows};paired=[]
        for endpoint in ('main','control'):
            for baseline in ('B1','Terminal'):
                counts=Counter()
                for (condition,seed,arm),label in values.items():
                    if arm!='B2' or registry['conditions'][condition]['endpoint']!=endpoint:continue
                    other=values[(condition,seed,baseline)]
                    counts['unknown' if 'UNKNOWN' in (label,other) else 'tie' if label==other else 'win' if label=='PASS' else 'loss']+=1
                paired.append(dict(endpoint=endpoint,baseline=baseline,**{k:counts[k] for k in ('win','loss','tie','unknown')}))
        live_progress={}
        for path in run.glob('TRAIN_PROGRESS_*.json'):
            r=read(path);live_progress[r['tag']]=r
        training_rows=[]
        for index,tag in enumerate(registry['models']):
            result=optional(run/'train'/tag/'RESULT.json');progress=live_progress.get(tag,{})
            step=result['updates'] if result else progress.get('step',0)
            logs=sorted((run/'train'/tag).glob('attempt_*/STEPS.jsonl'))
            recent={}
            if logs:
                text=tail(logs[-1]);lines=text.splitlines()
                # Only complete JSONL records; writer may be in the middle of its last append.
                for line in reversed(lines if text.endswith('\n') else lines[:-1]):
                    try:recent=json.loads(line);break
                    except ValueError:continue
            training_rows.append(dict(tag=tag,gpu=cfg['devices'][index%len(cfg['devices'])]['gpu'],step=step,planned=cfg['steps_per_arm'],
                complete=bool(result),loss=recent.get('loss'),auxiliary=recent.get('auxiliary'),gradient_norm=recent.get('gradient_norm')))
        workers=[];progresses=[]
        for w in status.get('workers',[]):
            if w['device'] is None:continue
            gpu=w['gpu'];tp=optional(run/f'TRAIN_PROGRESS_{gpu}.json',{});ep=optional(run/f'EVALUATION_PROGRESS_{gpu}.json',{})
            progress=dict(tp,model=tp.get('tag')) if status.get('stage')=='train' else ep
            if ep:progresses.append(ep)
            complete_files=sum(len(list(s.glob('rollouts/*/ROLLOUT.json'))) for s in (run/'evaluate').glob(f'session_gpu{gpu}_*'))
            logfiles=sorted((run/'attempts').glob(f"{status.get('stage')}_gpu{gpu}_*/stdout.log"))
            workers.append(dict(gpu_index=gpu,state='RUNNING' if w['returncode'] is None else 'COMPLETE' if w['returncode']==0 else 'FAILED',
                gpu=w['device'],progress=progress,completed_slots=complete_files,log=tail(logfiles[-1],2200) if logfiles else '',
                rss_bytes=w['resource']['rss_bytes']))
        audits=[a for g in groups.values() for a in g['audits']]
        completed_files=sum(len(list(s.glob('rollouts/*/ROLLOUT.json'))) for s in (run/'evaluate').glob('session_*'))
        ledger=run/'RESOURCES.jsonl';hours=0
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                try:hours+=json.loads(line)['gpu_hours']
                except ValueError:pass
        hours=max(hours,status.get('gpu_hours',0));service=optional(self.job/'STATUS.json',{})
        if service.get('status') in ('FAILED','INTERRUPTED'):errors.append('后台服务 '+service['status']+' '+str(service.get('error','')))
        logs=sorted((run/'attempts').glob('*/stdout.log'),key=lambda p:p.stat().st_mtime)
        local=PILOT.parent/'b2_localization_v1/run_002'
        localization=dict(status=optional(local/'STATUS.json',{}),counts=optional(local/'COUNTS.json'),
            workers=[optional(local/f'PROGRESS_{seed}.json',dict(seed=seed,status='WAITING')) for seed in (1209,1210,1211)],
            event=[dict(seed=seed,kind=kind,**row) for seed in (1209,1210,1211)
                   for kind,item in optional(local/f'seed_{seed}/EVENT_PROBE.json',{}).items()
                   for row in item['tables'] if row['house'] is None],
            report=(local/'REPORT_ZH.md').read_text() if (local/'REPORT_ZH.md').exists() else None)
        return dict(localization=localization,now=time.time(),run=run.name,status=status,service_status=service.get('status','UNKNOWN'),
            heartbeat_age_seconds=time.time()-(run/'STATUS.json').stat().st_mtime,groups=len(groups),planned_groups=len(registry['conditions']),
            complete=len(rows),planned=len(registry['slots']),unsealed_attempt_results=max(0,completed_files-len(rows)),
            models_complete=sum(r['complete'] for r in training_rows),planned_models=len(registry['models']),training_rows=training_rows,
            training_steps=sum(r['step'] for r in training_rows),training_planned_steps=len(registry['models'])*cfg['steps_per_arm'],
            training=max(live_progress.values(),key=lambda r:r['seconds']) if live_progress else {},feature=optional(run/'FEATURE_PROGRESS.json',{}),
            progress=max(progresses,key=lambda r:r['unix']) if progresses else {},metrics=metrics,paired=paired,workers=workers,
            gpu_hours=hours,gpu_budget_hours=cfg['gpu_session_hours'],audit=dict(comparisons=len(audits),bitwise_equal=sum(a['logits_bitwise_equal'] for a in audits),
                argmax_flips=sum(a['argmax_flip_count'] for a in audits),max_logit_delta=max((a['max_logit_delta'] for a in audits),default=None)),
            recent=sorted(rows,key=lambda r:r['rank'])[-18:],errors=errors,log=tail(logs[-1]) if logs else '',
            log_name=str(logs[-1].relative_to(run)) if logs else '',final_result=optional(run/'RESULT.json'))


def handler(reader):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == '/':
                    body = (HERE / 'index.html').read_bytes()
                    mime = 'text/html; charset=utf-8'
                elif path == '/api/status':
                    body = json.dumps(reader.snapshot(),ensure_ascii=False,allow_nan=False).encode()
                    mime = 'application/json; charset=utf-8'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type',mime)
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.send_error(503,'Snapshot temporarily unavailable')
                print('SNAPSHOT_ERROR',repr(exc),flush=True)

        def log_message(self, format, *args):
            if len(args)>1 and str(args[1]) not in ('200','304'):
                super().log_message(format,*args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=18770)
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    reader = Reader(PILOT / 'runs/matched_001', PILOT / 'standalone_jobs/b2-matched-20260921-01')
    if args.once:
        print(json.dumps(reader.snapshot(),ensure_ascii=False,indent=2))
        return
    server = HTTPServer(('127.0.0.1',args.port),handler(reader))
    print(f'READ_ONLY_MONITOR http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever(poll_interval=1)


if __name__ == '__main__':
    main()
