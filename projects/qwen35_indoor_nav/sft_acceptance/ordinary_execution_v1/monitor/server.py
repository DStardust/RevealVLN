"""Loopback-only read-only training monitor; no model imports or file writes."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
HOST, PORT = '127.0.0.1', 18766
MAX_JSON_BYTES = 2 * 1024 * 1024
STALE_SECONDS = 120
SECRET_KEYS = re.compile(r'(password|passwd|secret|credential|authorization|api_key|access_token|refresh_token|auth_token|environment|^env$)', re.I)
VISIBLE_KEYS = frozenset(('status', 'phase', 'unix', 'updated_unix', 'started_unix',
    'finished_unix', 'cursor', 'metrics', 'budget', 'throughput', 'steps', 'updates',
    'optimizer_steps', 'optimizer_updates', 'decisions', 'train_decisions',
    'loss', 'ce', 'CE', 'STOP', 'stop', 'stop_statistics', 'action_counts',
    'confusion', 'gpu', 'gpu_info', 'checkpoint', 'latest_checkpoint',
    'last_checkpoint', 'checkpoint_path', 'checkpoint_name', 'result',
    'pass_gate', 'completed', 'training_complete', 'scientific_pass',
    'error_type', 'reason_code', 'decision', 'probe', 'wall_seconds',
    'peak_allocated', 'peak_reserved', 'tokens', 'forward_tokens',
    'error', 'returncode', 'cleanup_complete', 'training_started', 'worker_pid',
    'elapsed_seconds', 'navigation_gain_verified', 'evaluation_performed',
    'measured_decisions_per_second', 'longest_chunk_mean_tokens', 'longest_grad_norm',
    'real_STOP_head_row_grad_norm', 'checkpoint_reload_forward_and_optimizer_update_exact',
    'old_forward_max_abs'))


def clean(value, depth=0):
    if depth > 8:
        return '[depth limit]'
    if isinstance(value, dict):
        return {str(k)[:100]: clean(v, depth + 1) for k, v in list(value.items())[:128]
                if not SECRET_KEYS.search(str(k))}
    if isinstance(value, list):
        return [clean(v, depth + 1) for v in value[:256]]
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value is None or isinstance(value, (bool, int, float)) else None


def read_status_file(path, root, now):
    record = {'file': path.name, 'exists': False, 'readable': False,
              'age_seconds': None, 'stale': None, 'data': None, 'error': None}
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('path outside root')
        stat = path.stat()
        record['exists'] = True
        record['age_seconds'] = round(max(0, now - stat.st_mtime), 1)
        record['stale'] = record['age_seconds'] > STALE_SECONDS
        if stat.st_size > MAX_JSON_BYTES:
            raise ValueError('oversized JSON')
        with path.open('rb') as stream:
            blob = stream.read(MAX_JSON_BYTES + 1)
        if len(blob) > MAX_JSON_BYTES:
            raise ValueError('oversized JSON')
        payload = json.loads(blob)
        if not isinstance(payload, dict):
            raise ValueError('status is not object')
        record['data'] = clean({k: v for k, v in payload.items() if k in VISIBLE_KEYS})
        record['readable'] = True
    except FileNotFoundError:
        record['error'] = 'MISSING'
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        record['error'] = type(exc).__name__
    return record


def gpu_status():
    command = ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
        if result.returncode:
            return {'available': False, 'error': 'NONZERO_EXIT', 'devices': []}
        devices = []
        for line in result.stdout.splitlines()[:32]:
            values = [x.strip() for x in line.split(',')]
            if len(values) != 4 or not all(x.isdigit() for x in values):
                continue
            index, used, total, utilization = map(int, values)
            devices.append(dict(index=index, used_mib=used, total_mib=total, utilization_percent=utilization))
        return {'available': bool(devices), 'error': None if devices else 'UNPARSEABLE', 'devices': devices}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'error': type(exc).__name__, 'devices': []}


def checkpoint_name(run, progress, result):
    for source in (result, progress):
        payload = source.get('data') or {}
        for key in ('latest_checkpoint', 'last_checkpoint', 'checkpoint', 'checkpoint_path', 'checkpoint_name'):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return Path(value).name
    folder = run / 'checkpoints'
    try:
        if not folder.resolve().is_relative_to(run.resolve()):
            return None
        candidates = []
        for i, path in enumerate(folder.iterdir()):
            if i >= 256:
                break
            if path.suffix in ('.pt', '.pth', '.safetensors') and path.is_file() and not path.is_symlink():
                candidates.append((path.stat().st_mtime, path.name))
        return max(candidates)[1] if candidates else None
    except OSError:
        return None


def collect_status(root=RUN_ROOT, *, now=None, include_gpu=True):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    report = {'server_unix': now, 'stale_after_seconds': STALE_SECONDS,
              'read_only': True, 'status': read_status_file(root / 'STATUS.json', root, now),
              'probe': read_status_file(root / 'PROBE.json', root, now), 'runs': [],
              'scan_error': None, 'runs_truncated': False}
    try:
        candidates = []
        for i, path in enumerate(root.iterdir()):
            if i >= 4096:
                report['runs_truncated'] = True
                break
            if re.fullmatch(r'run[A-Za-z0-9_.-]*', path.name) and path.is_dir() and not path.is_symlink():
                candidates.append((path.stat().st_mtime, path))
        candidates.sort(key=lambda item: item[0], reverse=True)
        report['runs_truncated'] |= len(candidates) > 64
        for _, run in candidates[:64]:
            progress = read_status_file(run / 'PROGRESS.json', root, now)
            result = read_status_file(run / 'RESULT.json', root, now)
            report['runs'].append({'name': run.name, 'progress': progress, 'result': result,
                'checkpoint_name': checkpoint_name(run, progress, result)})
    except OSError as exc:
        report['scan_error'] = type(exc).__name__
    report['gpu'] = gpu_status() if include_gpu else {'available': False, 'error': 'NOT_REQUESTED', 'devices': []}
    return report


HTML = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>普通导航训练监控</title><style>body{font:16px system-ui;margin:24px;background:#10171f;color:#e1e9ef}h1{font-size:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#192632;padding:16px;border-radius:8px}.stale{color:#ffbc5c}.ok{color:#86dbb1}small{color:#b7c3cd}section{margin:20px 0}</style>
<h1>普通导航训练 · 只读监控</h1><p id="connection">读取中…</p><small>每10秒刷新。陈旧表示文件超过120秒未更新；历史终态陈旧不代表失败。GPU利用率不代表训练有效进度。</small><main id="view"></main>
<script>const view=document.getElementById('view'),connection=document.getElementById('connection');
function block(title,obj,stale){const s=document.createElement('section'),h=document.createElement('h2'),p=document.createElement('pre');h.textContent=title;h.className=stale?'stale':'ok';p.textContent=JSON.stringify(obj,null,2);s.append(h,p);view.append(s)}
async function refresh(){try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();connection.textContent='连接正常 · '+new Date(data.server_unix*1000).toLocaleString();view.replaceChildren();block('总状态',data.status,data.status.stale);block('训练前探测',data.probe,data.probe.stale);block('GPU（只读）',data.gpu,false);if(!data.runs.length)block('尚无 run 目录',{scan_error:data.scan_error},false);for(const run of data.runs)block(run.name+(run.progress.stale?' · 进度陈旧':'')+' · checkpoint: '+(run.checkpoint_name||'尚无'),run,run.progress.stale);if(data.runs_truncated)block('注意',{message:'运行列表达到显示上限'},true)}catch(e){connection.textContent='读取失败；保留上次画面：'+e.message;connection.className='stale'}}refresh();setInterval(refresh,10000);</script></html>'''.encode()


def make_handler(root=RUN_ROOT):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'TrainingMonitor'
        sys_version = ''

        def log_message(self, *args):
            pass

        def respond(self, status, payload, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/':
                self.respond(200, HTML, 'text/html; charset=utf-8')
            elif self.path == '/healthz':
                self.respond(200, b'{"ok":true,"read_only":true}')
            elif self.path == '/api/status':
                self.respond(200, json.dumps(collect_status(root), allow_nan=False).encode())
            else:
                self.respond(404, b'{"error":"NOT_FOUND"}')

        do_HEAD = do_GET

        def do_POST(self):
            self.respond(405, b'{"error":"READ_ONLY"}')

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_POST
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='print snapshot without listening')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(collect_status(), indent=2, allow_nan=False))
        return
    server = ThreadingHTTPServer((HOST, PORT), make_handler())
    server.daemon_threads = True
    print(f'Read-only monitor http://{HOST}:{PORT}; SSH forwarding only', flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
