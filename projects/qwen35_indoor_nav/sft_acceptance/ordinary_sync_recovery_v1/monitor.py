"""Loopback-only read-only live monitor; access remotely through an SSH tunnel."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / 'formal'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def collect():
    status = read(OUT / 'STATUS.json')
    run = Path(status.get('run_dir') or OUT / 'attempt_001')
    assert run.resolve().is_relative_to(OUT.resolve())
    progress = read(run / 'PROGRESS.json')
    now = time.time()
    age = now - progress['unix'] if progress else None
    supervisor_age = now - status['unix'] if status else None
    state = status.get('status', 'NOT_STARTED')
    if state in ('TRAINING', 'STARTING') and supervisor_age is not None and supervisor_age > 20:
        state = 'SUPERVISOR_STALE'
    elif state == 'TRAINING' and age is not None and age > 120:
        state = 'STALLED'
    points = []
    path = run / 'PROGRESS.jsonl'
    if path.exists():
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 2 * 1024**2))
            blob = f.read()
        for line in blob.splitlines()[-1000:]:
            try:
                r = json.loads(line)
                points.append(dict(updates=r['cursor']['updates'], ce=r['metrics']['mean_ce'],
                    accuracy=r['metrics']['accuracy'], stop=r['metrics']['action_recall'][3],
                    speed=r['throughput']))
            except (json.JSONDecodeError, KeyError):
                pass
    return dict(state=state, server_unix=now, progress_age_seconds=age,
                supervisor_age_seconds=supervisor_age, supervisor=status, progress=progress,
                result=read(run / 'RESULT.json'), points=points,
                metric_scope='ONLINE_FIT_TRAINING_NOT_DEV_NOT_CLOSED_LOOP', read_only=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, content='application/json; charset=utf-8'):
        self.send_response(code)
        self.send_header('Content-Type', content)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            self.send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/status':
            self.send(200, json.dumps(collect(), allow_nan=False).encode())
        elif self.path == '/healthz':
            self.send(200, b'{"ok":true,"read_only":true}')
        else:
            self.send(404, b'{"error":"NOT_FOUND"}')

    def do_POST(self):
        self.send(405, b'{"error":"READ_ONLY"}')

    do_PUT = do_DELETE = do_PATCH = do_POST


if __name__ == '__main__':
    with ThreadingHTTPServer(('127.0.0.1', 18767), Handler) as server:
        print('Read-only recovery monitor: http://127.0.0.1:18767', flush=True)
        server.serve_forever()
