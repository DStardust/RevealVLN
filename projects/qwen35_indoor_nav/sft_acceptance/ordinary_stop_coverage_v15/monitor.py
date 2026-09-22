"""Read-only ordinary coverage and existing 100k memory progress at the original URL."""
import json,sys,time
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
review=u.load('coverage15_monitor_review',u.HERE/'review.py')
LONG=u.LINE/'research/continuation_memory_v1/evidence_state_policy_v1/expanded_longtrain_v1/runs/long_001'
def read(path,default=None):return u.read(path) if path.exists() else default
def snapshot():
    run=u.HERE/'runs'/(u.HERE/'LAST_RUN.txt').read_text().strip();values=review.completed_collect(run/'collect');collection=list(values.values());checkpoints=[]
    for step in range(5000,100001,5000):
        folder=LONG/'evaluations'/f'STEP_{step:06d}';r=read(folder/'RESULT.json');groups=list((folder/'evaluate').glob('session_*/STATE_SEAL_???.json'))
        row=dict(step=step,saved=(LONG/'checkpoints'/f'STEP_{step:06d}/COMMIT.json').exists(),complete=r['complete'] if r else len(groups),planned=256,main=None,control=None)
        if r:
            for m in r['metrics']:
                if m['house'] is None:row[m['endpoint']]=m
        checkpoints.append(row)
    return dict(unix=time.time(),status=read(run/'STATUS.json',{}),collection=dict(complete=len(collection),planned=640,decisions=sum(v['steps'] for v in collection),stop_positive_observations=sum(v['positive'] for v in collection),teacher_unavailable=sum(v['termination']=='TEACHER_UNAVAILABLE' for v in collection)),
        sessions=[dict(session=str(p.parent.relative_to(run)),**u.read(p)) for p in run.glob('*/sessions/*/PROGRESS.json')],
        training=read(run/'TRAIN_RESULT.json'),DEV=read(run/'DEV/LIVE_RESULT.json'),UNSEEN=read(run/'UNSEEN/LIVE_RESULT.json'),
        long_training=dict(status=read(LONG/'STATUS.json',{}),progress=read(LONG/'train/PROGRESS.json',{}),checkpoints=checkpoints))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path=='/':body=(u.HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
            elif self.path in ('/api/status','/api/ordinary'):body=json.dumps(snapshot(),ensure_ascii=False).encode();mime='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:self.send_error(503);print(repr(e),flush=True)
    def log_message(self,*a):pass
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',18770),Handler).serve_forever()
