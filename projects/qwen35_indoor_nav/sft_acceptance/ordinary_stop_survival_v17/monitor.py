"""Original history dashboard plus the new stop-risk repair, without changing old files."""
import importlib.util,json,sys,time
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'ordinary_action_repair_v16'
# Import the old monitor with its own common module before adding this version's view.
sys.path.insert(0,str(OLD))
s=importlib.util.spec_from_file_location('survival17_previous_monitor',OLD/'ops_recovery_r1/monitor.py');base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def snapshot():
    result=base.snapshot();run=HERE/'runs'/(HERE/'LAST_RUN.txt').read_text().strip()
    result['survival']=dict(status=read(run/'STATUS.json',{}),training=read(run/'TRAIN_PROGRESS.json',{}),train_result=read(run/'TRAIN_RESULT.json'),DEV=read(run/'DEV/LIVE_RESULT.json'),UNSEEN=read(run/'UNSEEN/LIVE_RESULT.json'),gate=read(run/'DEV_GATE.json'),unseen_not_run=read(run/'UNSEEN_NOT_RUN.json'),live=base.live(run))
    return result
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
