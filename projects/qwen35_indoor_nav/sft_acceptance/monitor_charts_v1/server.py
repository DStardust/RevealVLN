"""Bounded, read-only chart monitor. No GPU ownership or production writes."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
EXECUTION = LINE / 'sft_acceptance/ordinary_execution_v6'
SNAPSHOT = LINE / 'sft_acceptance/ordinary_baseline_v2/snapshot_v1/RESULT.json'
QUEUES = LINE / 'data_pipeline/auto_production_v1'
HOST, PORT = '127.0.0.1', 18766
MAX_JSON, MAX_TAIL, MAX_POINTS = 2 * 1024**2, 8 * 1024**2, 1000
STALE_SECONDS = 120
SECRETS = re.compile(r'password|passwd|secret|credential|authorization|api_key|access_token|refresh_token|auth_token|^env$|environment', re.I)


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def scrub(value, depth=0):
    if depth > 8:
        return None
    if isinstance(value, dict):
        return {k: scrub(v, depth+1) for k, v in list(value.items())[:256] if not SECRETS.search(k)}
    if isinstance(value, list):
        return [scrub(v, depth+1) for v in value[:256]]
    if isinstance(value, str):
        return value[:1000]
    return value if value is None or type(value) in (bool, int) else number(value)


def safe(path, boundary=LINE):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(boundary).resolve()):
        raise ValueError('OUTSIDE_SCOPE')
    return path


def read_json(path, boundary=LINE, now=None):
    result = {'data': None, 'error': None, 'age_seconds': None, 'stale': None}
    try:
        path = safe(path, boundary)
        stat = path.stat()
        result['age_seconds'] = round(max(0, (time.time() if now is None else now)-stat.st_mtime), 1)
        result['stale'] = result['age_seconds'] > STALE_SECONDS
        if stat.st_size > MAX_JSON:
            raise ValueError('OVERSIZED_JSON')
        with path.open('rb') as handle:
            raw = handle.read(MAX_JSON+1)
        if len(raw) > MAX_JSON:
            raise ValueError('OVERSIZED_JSON')
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError('JSON_OBJECT_REQUIRED')
        result['data'] = scrub(payload)
    except FileNotFoundError:
        result['error'] = 'MISSING'
    except (ValueError, OSError, UnicodeError, RecursionError) as exc:
        result['error'] = type(exc).__name__
    return result


def tail_records(path, boundary=LINE, max_bytes=MAX_TAIL, max_points=MAX_POINTS):
    """Read only a byte tail; ignore incomplete first/last lines explicitly."""
    result = {'records': [], 'error': None, 'malformed_lines': 0,
              'partial_last_line': False, 'tail_truncated': False, 'points_truncated': False}
    try:
        path = safe(path, boundary)
        with path.open('rb') as handle:
            handle.seek(0, 2)
            size = handle.tell()
            start = max(0, size-max_bytes)
            handle.seek(start)
            raw = handle.read(max_bytes)
        result['tail_truncated'] = start > 0
        if start:
            raw = raw.partition(b'\n')[2]
        result['partial_last_line'] = bool(raw and not raw.endswith(b'\n'))
        lines = raw.split(b'\n')[:-1]
        result['points_truncated'] = len(lines) > max_points
        for line in lines[-max_points:]:
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('NOT_OBJECT')
                result['records'].append(row)
            except (ValueError, UnicodeError, RecursionError):
                result['malformed_lines'] += 1
    except FileNotFoundError:
        result['error'] = 'MISSING'
    except (OSError, ValueError) as exc:
        result['error'] = type(exc).__name__
    return result


def chart_points(records, window=20):
    points, recent = [], []
    previous = None
    for row in records:
        cursor = row.get('cursor') or {}
        metric = row.get('metrics') or {}
        if not all(isinstance(x, dict) for x in (cursor, metric)):
            continue
        last = metric.get('last_chunk') or {}
        if not isinstance(last, dict):
            continue
        unix, decisions = number(row.get('unix')), number(cursor.get('decisions'))
        if unix is None or decisions is None:
            continue
        ce = number(last.get('ce'))
        reset = previous is not None and (unix <= previous[0] or decisions <= previous[1])
        if reset:
            recent = []
        if ce is not None:
            recent.append(ce)
            recent = recent[-window:]
        points.append({'unix': unix, 'decisions': decisions, 'updates': number(cursor.get('updates')),
            'epoch': number(cursor.get('epoch')), 'last_chunk_ce': ce,
            'rolling_ce': sum(recent)/len(recent) if ce is not None and recent else None,
            'throughput': number(row.get('throughput')), 'discontinuity': reset})
        previous = (unix, decisions)
    return points


def action_summary(metrics):
    matrix = metrics.get('confusion') if isinstance(metrics, dict) else None
    if not (isinstance(matrix, list) and len(matrix) == 4 and all(
            isinstance(row, list) and len(row) == 4 and all(number(v) is not None and v >= 0 for v in row)
            for row in matrix)):
        return {'confusion': None, 'targets': None, 'predictions': None, 'recall': None}
    targets = [sum(row) for row in matrix]
    predictions = [sum(matrix[i][j] for i in range(4)) for j in range(4)]
    return {'confusion': matrix, 'targets': targets, 'predictions': predictions,
            'recall': [matrix[i][i]/targets[i] if targets[i] else None for i in range(4)]}


def gpu_status():
    try:
        value = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=False)
        devices = []
        if value.returncode:
            return {'devices': [], 'error': 'NONZERO_EXIT'}
        for line in value.stdout.splitlines()[:32]:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 4 and all(x.isdigit() for x in parts):
                i, used, total, util = map(int, parts)
                devices.append({'index': i, 'used_mib': used, 'total_mib': total, 'utilization_percent': util})
        return {'devices': devices, 'error': None if devices else 'NO_PARSED_DEVICES'}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'devices': [], 'error': type(exc).__name__}


def matches_queue(argv, cwd, plan_path, gpu):
    """Resolve relative CLI paths against the actual process cwd."""
    try:
        cwd = Path(cwd).resolve()
        if cwd != ROOT.resolve():
            return False
        script = QUEUES/'queue.py'
        if not any((cwd/value).resolve() == script.resolve() for value in argv if value.endswith('.py')):
            return False
        plan_index, gpu_index = argv.index('--plan'), argv.index('--gpu')
        return ((cwd/argv[plan_index+1]).resolve() == Path(plan_path).resolve()
                and argv[gpu_index+1] == str(gpu))
    except (OSError, ValueError, IndexError):
        return False


def queue_alive(pid, plan_path, gpu):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        proc = Path('/proc', str(pid))
        argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        return matches_queue(argv, (proc/'cwd').resolve(), plan_path, gpu)
    except (OSError, UnicodeError):
        return False


def queue_table(base=QUEUES, boundary=LINE):
    """Shallow receipts only: bounded JSON/tails, no broad content traversal."""
    output = []
    try:
        plans = sorted(Path(base).glob('*/PLAN.json'))[:64]
        for plan_path in plans:
            plan = read_json(plan_path, boundary)['data'] or {}
            jobs = plan.get('jobs') or []
            if not isinstance(jobs, list):
                continue
            for gpu in sorted({j.get('gpu') for j in jobs if isinstance(j, dict) and type(j.get('gpu')) is int}):
                lane = plan_path.parent / f'lane_gpu_{gpu}'
                identity = read_json(lane/'IDENTITY.json', boundary)['data'] or {}
                receipt = read_json(lane/'RESULT.json', boundary)
                events = tail_records(lane/'EVENTS.jsonl', boundary, max_bytes=256*1024, max_points=1)
                last = events['records'][-1] if events['records'] else {}
                pid = identity.get('pid')
                alive = queue_alive(pid, plan_path, gpu)
                final = receipt['data']
                state = 'RUNNING' if alive else 'NO_LIVE_QUEUE' if identity else 'NOT_LAUNCHED'
                if final is not None:
                    state = 'STOPPED_ERROR' if final.get('error') else 'CLOSED'
                    if not final.get('error') and any(v in ('PENDING', 'RESERVED_NOT_LAUNCHED_DRAIN_OR_DEADLINE')
                            for v in (final.get('states') or {}).values()):
                        state = 'CLOSED_WITH_PENDING'
                output.append({'plan': plan_path.parent.name, 'gpu': gpu, 'state': state,
                    'alive': alive, 'pid': pid, 'current_job': last.get('job'),
                    'last_event': last.get('event'), 'event_unix': number(last.get('unix')),
                    'note': 'GPU1串行恢复225→500、244→501及6个未执行批次；旧失败保留，无重复并发' if plan_path.parent.name == 'manual_failure_recovery_gpu1_20260910_v1' else None,
                    'error': scrub(final.get('error')) if final else receipt['error'] if receipt['error'] != 'MISSING' else None,
                    'planned_jobs': sum(j.get('gpu') == gpu for j in jobs if isinstance(j, dict))})
    except OSError as exc:
        output.append({'state': 'READ_ERROR', 'error': type(exc).__name__})
    return output


def collect(execution=EXECUTION, snapshot=SNAPSHOT, queues=QUEUES, boundary=LINE, include_gpu=True):
    now = time.time()
    execution = Path(execution)
    status = read_json(execution/'STATUS.json', boundary, now)
    probe = read_json(execution/'PROBE.json', boundary, now)
    snapshot_data = read_json(snapshot, boundary, now)
    runs = []
    try:
        runs = [p for p in execution.glob('run*') if p.is_dir() and not p.is_symlink()][:64]
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    run = runs[0] if runs else execution/'run_0001'
    progress = read_json(run/'PROGRESS.json', boundary, now)
    result = read_json(run/'RESULT.json', boundary, now)
    tail = tail_records(run/'PROGRESS.jsonl', boundary)
    rows = tail.pop('records')
    points = chart_points(rows)
    value = progress['data'] or {}
    cursor, budget = value.get('cursor') or {}, value.get('budget') or {}
    metrics = value.get('metrics') or {}
    counts = (snapshot_data['data'] or {}).get('counts') or {}
    total = number(counts.get('instruction_conditioned_decisions'))
    done, speed = number(cursor.get('decisions')), number(value.get('throughput'))
    limit = number(budget.get('max_decisions'))
    compute = value.get('cumulative_compute') or {}
    charged = number(compute.get('decisions'))
    eta = (max(0, limit-charged)/speed) if None not in (limit, charged, speed) and speed > 0 else None
    elapsed = number((metrics.get('last_chunk') or {}).get('elapsed_seconds'))
    wall = number(budget.get('wall_seconds'))
    wall_remaining = max(0, wall-600-elapsed) if None not in (wall, elapsed) else None
    soft_limit = 320000  # Exact V6 execute.py soft-stop cursor threshold.
    soft_eta = max(0, soft_limit-done)/speed if None not in (done, speed) and speed > 0 else None
    candidates = [x for x in (wall_remaining, eta, soft_eta) if x is not None]
    updates = number(cursor.get('updates'))
    max_updates = number(budget.get('max_updates'))
    epoch_eta = (max(0, total-done % total)/speed) if None not in (total, done, speed) and total > 0 and speed > 0 else None
    checkpoint = value.get('latest_checkpoint')
    return {'server_unix': now, 'run_name': run.name, 'status': status, 'probe': probe,
        'progress': progress, 'result': result, 'history': tail, 'points': points,
        'rolling_window_records': 20, 'actions': action_summary(metrics), 'snapshot_counts': counts,
        'snapshot_error': snapshot_data['error'], 'decision_limit': limit, 'decisions': done,
        'charged_compute_decisions': charged, 'training_soft_limit': soft_limit,
        'wall_remaining_seconds': wall_remaining, 'wall_grace_seconds': 600,
        'updates_remaining': max(0, max_updates-updates) if None not in (max_updates, updates) else None,
        'estimated_segment_remaining_seconds': min(candidates) if candidates else None,
        'epoch_total_decisions': total, 'budget_eta_seconds': eta, 'epoch_eta_seconds': epoch_eta,
        'checkpoint_name': Path(checkpoint).name if isinstance(checkpoint, str) else None,
        'gpu': gpu_status() if include_gpu else {'devices': [], 'error': 'NOT_REQUESTED'},
        'queues': queue_table(queues, boundary),
        'metric_scope': 'ONLINE_TRAINING_NOT_DEV_OR_CLOSED_LOOP',
        'eta_note': 'Linear estimate from recorded throughput; segment may end sooner at time/token/update budget.'}


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        server_version, sys_version = 'ReadOnlyTrainingCharts', ''
        def log_message(self, *args):
            pass
        def respond(self, code, payload, kind='application/json; charset=utf-8'):
            self.send_response(code)
            for key, value in [('Content-Type', kind), ('Content-Length', str(len(payload))),
                    ('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff'),
                    ('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")]:
                self.send_header(key, value)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(payload)
        def do_GET(self):
            if self.path == '/':
                self.respond(200, (HERE/'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/healthz':
                self.respond(200, b'{"ok":true,"read_only":true}')
            elif self.path == '/api/status':
                try:
                    self.respond(200, json.dumps(collect(), allow_nan=False).encode())
                except Exception as exc:
                    self.respond(503, json.dumps({'error': type(exc).__name__}).encode())
            else:
                self.respond(404, b'{"error":"NOT_FOUND"}')
        do_HEAD = do_GET
        def do_POST(self):
            self.respond(405, b'{"error":"READ_ONLY"}')
        do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_POST
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(collect(), indent=2, allow_nan=False))
        return
    with ThreadingHTTPServer((HOST, PORT), make_handler()) as server:
        server.daemon_threads = True
        print(f'Read-only charts http://{HOST}:{PORT}', flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
