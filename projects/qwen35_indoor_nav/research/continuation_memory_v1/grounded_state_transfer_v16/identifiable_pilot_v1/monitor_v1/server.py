"""Read-only, localhost dashboard for the frozen identifiable pilot. Standard library only."""
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
        run = self.run
        registry = read(run / 'EVALUATION_REGISTRY.json')
        status = read(run / 'STATUS.json')
        groups, errors = {}, []
        for session in sorted((run / 'evaluate').glob('session_*')):
            for path in sorted(session.glob('GROUP_*.json')):
                try:
                    group = self.group(session, path, registry)
                    if group is None:
                        continue
                    condition = group['condition']
                    if condition in groups:
                        raise ValueError('Duplicate sealed condition')
                    groups[condition] = group
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f'{session.name}/{path.name}: {exc}')
        rows = [row for g in groups.values() for row in g['rows']]
        metrics = []
        for endpoint in ('main', 'control'):
            for arm in ('B1', 'B2', 'Ours'):
                selected = [r for r in rows if r['endpoint'] == endpoint and r['arm'] == arm]
                planned = sum(s['arm'] == arm and registry['conditions'][s['condition']]['endpoint'] == endpoint for s in registry['slots'])
                counts = Counter(r['label'] for r in selected)
                n = len(selected)
                metrics.append(dict(endpoint=endpoint, arm=arm, complete=n, planned=planned,
                    passed=counts['PASS'], failed=counts['FAIL'], unknown=counts['UNKNOWN'],
                    not_run=planned-n, partial_rate=counts['PASS']/n if n else None,
                    identification_lower=counts['PASS']/planned,
                    identification_upper=(counts['PASS']+counts['UNKNOWN']+planned-n)/planned,
                    collisions=sum(r['collisions'] for r in selected),
                    exhausted=sum(r['exhausted'] for r in selected)))
        paired = []
        values = {(r['condition'], r['seed'], r['arm']): r['label'] for r in rows}
        for endpoint in ('main', 'control'):
            for baseline in ('B1', 'B2'):
                count = Counter()
                for (condition, seed, arm), ours in values.items():
                    if arm != 'Ours' or registry['conditions'][condition]['endpoint'] != endpoint:
                        continue
                    base = values[(condition, seed, baseline)]
                    outcome = ('unknown' if 'UNKNOWN' in (ours, base) else 'tie' if ours == base else 'win' if ours == 'PASS' else 'loss')
                    count[outcome] += 1
                paired.append(dict(endpoint=endpoint, baseline=baseline, **{k:count[k] for k in ('win','loss','tie','unknown')}))
        audits = [a for g in groups.values() for a in g['audits']]
        resources = run / 'RESOURCES.jsonl'
        sessions = [json.loads(s) for s in resources.read_text().splitlines() if s.strip()] if resources.exists() else []
        hours = sum(r['seconds']/3600 for r in sessions if r['gpu'])
        live = status.get('status') == 'RUNNING'
        if live and status.get('stage') in ('features','train','diagnose','evaluate_continuations'):
            hours += status.get('elapsed', 0)/3600
        service = optional(self.job / 'STATUS.json', {})
        parallel_dir = run.parent.parent / 'parallel_eval_v1'
        parallel = optional(parallel_dir / 'STATUS.json')
        workers = []
        if parallel and parallel['status'] != 'HANDOFF_WAIT':
            workers = parallel.get('workers', [])
            stage = {'RUNNING':'parallel_evaluate','REVIEWING':'review','RESTORING_PLACEHOLDERS':'restore'}
            status = dict(status='RUNNING' if parallel['status'] in stage else parallel['status'],
                stage=stage.get(parallel['status'],parallel['status']),
                rss_bytes=sum(w.get('rss_bytes',0) for w in workers),gpu=None)
            if parallel.get('reason'):status['reason']=parallel['reason']
            service = optional(run.parent.parent / 'standalone_jobs/ident-parallel-20260921-01/STATUS.json', {})
            if 'gpu_hours' in parallel:
                hours = parallel['gpu_hours']
            elif (parallel_dir/'RESOURCES.jsonl').exists():
                ledger = [json.loads(x) for x in (parallel_dir/'RESOURCES.jsonl').read_text().splitlines() if x.strip()][-1]
                hours = ledger['baseline_gpu_hours']+sum(w['wall_seconds']/3600 for w in ledger['workers'])
        logs = sorted((run / 'attempts').glob('*/stdout.log'), key=lambda p:p.stat().st_mtime)
        if parallel and parallel['status'] != 'HANDOFF_WAIT':
            logs = sorted(parallel_dir.glob('gpu_*/stdout.log'),key=lambda p:p.stat().st_mtime) or logs
        train = optional(run / 'train/RESULT.json', {})
        completed_files = sum(len(list(s.glob('rollouts/*/ROLLOUT.json'))) for s in (run / 'evaluate').glob('session_*'))
        current_progress = optional(run / 'EVALUATION_PROGRESS.json', {})
        if workers:
            progressing = [w['progress'] for w in workers if w.get('progress')]
            current_progress = max(progressing,key=lambda p:p['unix']) if progressing else {}
        statefile=parallel_dir/'STATUS.json' if parallel else run/'STATUS.json'
        return dict(now=time.time(), run=run.name, status=status, service_status=service.get('status','UNKNOWN'),
            heartbeat_age_seconds=time.time()-statefile.stat().st_mtime,
            groups=len(groups), planned_groups=len(registry['conditions']), complete=len(rows),
            planned=len(registry['slots']), unsealed_attempt_results=max(0,completed_files-len(rows)),
            models_complete=len(train.get('models',[])), planned_models=len(registry['models']),
            training=optional(run / 'TRAIN_PROGRESS.json', {}),
            progress=current_progress, metrics=metrics, paired=paired, workers=workers, parallel=parallel,
            gpu_hours=hours, gpu_budget_hours=read(run / 'PROTOCOL.json')['gpu_session_hours'],
            audit=dict(comparisons=len(audits), bitwise_equal=sum(a['logits_bitwise_equal'] for a in audits),
                argmax_flips=sum(a['argmax_flip_count'] for a in audits),
                max_logit_delta=max((a['max_logit_delta'] for a in audits),default=None)),
            recent=sorted(rows,key=lambda r:r['rank'])[-18:], errors=errors,
            log=tail(logs[-1]) if logs else '', log_name=str(logs[-1].relative_to(run.parent.parent)) if logs else '',
            final_result=optional(run / 'RESULT.json'))


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
    reader = Reader(PILOT / 'runs/pilot_001', PILOT / 'standalone_jobs/ident-pilot-20260921-01')
    if args.once:
        print(json.dumps(reader.snapshot(),ensure_ascii=False,indent=2))
        return
    server = HTTPServer(('127.0.0.1',args.port),handler(reader))
    print(f'READ_ONLY_MONITOR http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever(poll_interval=1)


if __name__ == '__main__':
    main()
