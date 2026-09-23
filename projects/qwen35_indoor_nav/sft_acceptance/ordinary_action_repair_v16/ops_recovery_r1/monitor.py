"""Read-only live worker heartbeat alongside separately timestamped metric snapshots."""
import json,sys,time
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import common as u
base=u.load('motion16_original_monitor',HERE.parent/'monitor.py')
def live(run):
    status=base.read(run/'STATUS.json',{});phase=status.get('phase','');stage=run/phase
    sessions=[]
    for p in stage.glob('sessions/*/PROGRESS.json'):
        row=u.read(p);sessions.append(dict(session=p.parent.name,age_seconds=round(time.time()-p.stat().st_mtime,1),**row))
    seals=list(stage.glob('sessions/*/pairs/*/PAIR.json'))
    ranks=[p.parent.name for p in seals]
    metric=stage/'LIVE_RESULT.json'
    return dict(phase=phase,last_worker_update_seconds=min((x['age_seconds'] for x in sessions),default=None),
        sealed_pairs=len(set(ranks)),duplicate_seal_names=len(ranks)-len(set(ranks)),
        metric_snapshot_age_seconds=round(time.time()-metric.stat().st_mtime,1) if metric.exists() else None,
        pipeline_heartbeat_age_seconds=round(time.time()-float(status.get('unix',0)),1),sessions=sessions)
def snapshot():
    data=base.snapshot();data['live']={'V15':live(u.HERE.parent/'ordinary_stop_coverage_v15/runs/coverage_001'),'V16':live(u.HERE/'runs/motion_001')}
    return data
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path=='/':body=(HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
            elif self.path in ('/api/status','/api/ordinary'):body=json.dumps(snapshot(),ensure_ascii=False).encode();mime='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:self.send_error(503);print(repr(e),flush=True)
    def log_message(self,*a):pass
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',18770),Handler).serve_forever()
