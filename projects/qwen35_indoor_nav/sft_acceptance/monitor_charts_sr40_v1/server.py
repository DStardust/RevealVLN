"""SR40 roadmap monitor. Read-only snapshot and live resource API; no launch endpoint."""
import importlib.util
import json
from pathlib import Path
from http.server import ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
REVIEW = LINE / 'reviews/Q35N_SR40_BASELINE_RESET_V1'
PARENT = HERE.parent / 'monitor_charts_stop_row_v12r1/server.py'
spec = importlib.util.spec_from_file_location('sr40_previous_monitor', PARENT)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

def collect(include_gpu=True):
    d = previous.collect(include_gpu=include_gpu)
    d['monitor_version'] = 'ordinary_sr40_v1'
    d['sr40'] = previous.read(REVIEW / 'RESULT.json')
    # This version has no registered SR40 training or full-benchmark run.
    # Future run wiring requires a new monitor version, not guessed old paths.
    d['sr40_training'] = None
    d['sr40_full_result'] = None
    return d

class Handler(previous.Handler):
    def do_GET(self):
        try:
            if self.path == '/':
                self.respond(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/refresh.js':
                self.respond(200, (HERE / 'refresh.js').read_bytes(), 'application/javascript')
            elif self.path == '/api/status':
                self.respond(200, json.dumps(collect(), ensure_ascii=False, allow_nan=False).encode(), 'application/json')
            elif self.path == '/healthz':
                self.respond(200, b'{"status":"ok","version":"ordinary_sr40_v1"}', 'application/json')
            else:
                self.respond(404, b'Not found', 'text/plain')
        except Exception as exc:
            self.respond(503, json.dumps({'error': repr(exc)}).encode(), 'application/json')
    do_HEAD = do_GET

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18766)
    args = parser.parse_args()
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.daemon_threads = True
        server.serve_forever(poll_interval=.5)

