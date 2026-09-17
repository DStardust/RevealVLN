"""Cached GET-only loopback monitor. Frozen production sources are read-only."""
import argparse
import copy
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'runtime_v2'
LINE = HERE.parents[2]
ROOT = LINE.parents[1]
LOCK_SHA = 'c693ff433c3a02557c5d125400ce77d877b396a4d26544ffd7420fba1189509f'
VERSION = 'expansion-monitor-recovery-v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('EXPECTED_JSON_OBJECT')
        return value
    except FileNotFoundError:
        return {}

def load_collector():
    lock = OLD / 'INPUT_LOCK.json'
    assert sha(lock) == LOCK_SHA, 'FROZEN_INPUT_LOCK_CHANGED'
    source = OLD / 'monitor.py'
    assert sha(source) == read(lock)[str(source.relative_to(ROOT))], 'FROZEN_MONITOR_SOURCE_CHANGED'
    spec = importlib.util.spec_from_file_location('frozen_expansion_monitor_reader', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def collect():
        data = module.collect()
        training = read(LINE / 'sft_acceptance/ordinary_sync_recovery_v1/formal/STATUS.json')
        data['training_status'] = training.get('status', 'UNKNOWN')
        bench = read(LINE / 'closed_loop_bench/r2r_ce_full_v2/run_001/RESULT.json')
        if not bench:
            bench = read(LINE / 'closed_loop_bench/r2r_ce_full_v2/run_001/PROGRESS.json')
        data['benchmark'] = {k: bench.get(k) for k in ('status', 'completed', 'planned', 'total', 'checkpoint_updates')}
        for lane in data['lanes']:
            lane['heartbeat_unix'] = None
            shard = lane['latest_shard']
            if shard is not None:
                path = HERE.parent / f'envdrop_production_v2/shard_{shard:04d}/PROGRESS.json'
                lane['heartbeat_unix'] = path.stat().st_mtime
        # Reject non-finite values here, not halfway through an HTTP response.
        json.dumps(data, allow_nan=False)
        return data
    return collect

class SnapshotCache:
    def __init__(self, collector, clock=time.time):
        self.collector, self.clock = collector, clock
        self.lock = threading.Lock()
        self.data = None
        self.error = None
        self.success_unix = None

    def refresh(self):
        try:
            data = self.collector()
            if not isinstance(data, dict) or len(data.get('lanes', [])) != 3:
                raise ValueError('INVALID_COLLECTOR_SCHEMA')
            json.dumps(data, allow_nan=False)
        except Exception as exc:
            with self.lock:
                self.error = type(exc).__name__ + ': ' + str(exc)
            return False
        with self.lock:
            self.data = data
            self.success_unix = self.clock()
            self.error = None
        return True

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.data) if self.data is not None else {}
            now = self.clock()
            age = now - self.success_unix if self.success_unix is not None else None
            result['monitor'] = dict(version=VERSION, server_unix=now, snapshot_unix=self.success_unix,
                age_seconds=age, stale=bool(self.error) or age is None or age > 20,
                source_error=self.error, has_snapshot=self.data is not None, read_only=True)
        return result

def script_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')

def page(data):
    esc = lambda value: html.escape(str(value))
    rows = []
    for lane in data.get('lanes', []):
        status = {'FAILED': '已停止（失败）', 'PRODUCING': '生产中', 'CLOSED': '已结束', 'FIRST_STRICT_GATE': '首批审核'}.get(lane['stage'], lane['stage'])
        detail = lane.get('result', {}).get('error') or ''
        rows.append('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in
            [lane['gpu'], status, f"{lane['completed']} / {lane['target']}", lane['strict_routes'], lane['strict_decisions'], detail]) + '</tr>')
    count = f"已处理 {data.get('completed', '—')} / {data.get('total_routes', 9661)} 条；严格合格 {data.get('strict_routes', '—')} 条 / {data.get('strict_decisions', '—')} 动作"
    cache_state = data['monitor']
    banner = '服务器已提供快照；正在确认浏览器实时连接。' if cache_state['has_snapshot'] else '暂未取得有效快照；正在重试。'
    page_source = (HERE / 'index.html').read_text()
    values = {'@@COUNT@@': esc(count), '@@ROWS@@': ''.join(rows), '@@BANNER@@': esc(banner),
              '@@BOOTSTRAP@@': script_json(data), '@@CLIENT@@': (HERE / 'client.js').read_text()}
    for key, value in values.items():
        assert page_source.count(key) == 1, key
        page_source = page_source.replace(key, value)
    return page_source.encode()

class Server(ThreadingHTTPServer):
    request_queue_size = 64
    daemon_threads = True
    allow_reuse_address = True

def handler_for(cache):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            data = cache.snapshot()
            code = 200
            if path in ('/', '/index.html'):
                body, kind = page(data), 'text/html; charset=utf-8'
            elif path == '/api/status':
                body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
                kind = 'application/json; charset=utf-8'
                if not data['monitor']['has_snapshot']:
                    code = 503
            elif path == '/healthz':
                body = json.dumps(dict(ok=True, read_only=True, version=VERSION,
                    data_ready=data['monitor']['has_snapshot'], data_stale=data['monitor']['stale'])).encode()
                kind = 'application/json'
            elif path == '/favicon.ico':
                self.send_response(204); self.end_headers(); return
            else:
                self.send_error(404); return
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store, max-age=0')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Client disconnected; no source-state mutation and no traceback flood.

        def do_POST(self):
            self.send_error(405)
        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST
        def log_message(self, *args):
            pass
    return Handler

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--port', type=int, choices=[18769, 18770], default=18769)
    args = args.parse_args()
    cache = SnapshotCache(load_collector())
    stop = threading.Event()
    def poll():
        previous = None
        while not stop.is_set():
            success = cache.refresh()
            if success != previous:
                print(json.dumps(dict(event='source_ready' if success else 'source_read_failed', unix=time.time())), flush=True)
                previous = success
            stop.wait(5)
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        with Server(('127.0.0.1', args.port), handler_for(cache)) as server:
            print(json.dumps(dict(event='listening', port=args.port, version=VERSION)), flush=True)
            server.serve_forever(poll_interval=.5)
    finally:
        stop.set()

if __name__ == '__main__':
    main()
