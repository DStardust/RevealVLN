import importlib.util,json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
P=HERE.parent/'monitor_charts_onpolicy_eval_v6r3/server.py'
s=importlib.util.spec_from_file_location('closed_correction_monitor',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
REVIEW=LINE/'reviews/Q35N_ORDINARY_EQUAL_MERGE_V7R1'
FIT=LINE/'closed_loop_bench/ordinary_equal_merge_fit_v7r1'
DEV=LINE/'closed_loop_bench/ordinary_equal_merge_dev_v7r1'
def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None
def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu);d['monitor_version']='ordinary_equal_merge_v7r1'
    d['equal_merge']=dict(workflow=read(REVIEW/'WORKFLOW_STATUS.json'),workflow_result=read(REVIEW/'WORKFLOW_RESULT.json'),
      result=read(REVIEW/'RESULT.json'),fit_gate=read(REVIEW/'FIT_GATE.json'),
      fit_progress=read(FIT/'run_001/PROGRESS.json'),fit_result=read(FIT/'run_001/RESULT.json'),
      dev_progress=read(DEV/'run_001/PROGRESS.json'),dev_result=read(DEV/'run_001/RESULT.json'),
      dev_admitted=(DEV/'SOURCE_LOCK.json').exists(),alpha=.5,parameter_updates=0,
      architecture_unchanged=True,one_forward_per_action=True)
    return d
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,status,data,kind):
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(data)));self.end_headers()
        if self.command!='HEAD':self.wfile.write(data)
    def do_GET(self):
        try:
            if self.path=='/':self.respond(200,(HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/refresh.js':self.respond(200,(HERE/'refresh.js').read_bytes(),'application/javascript')
            elif self.path=='/api/status':self.respond(200,json.dumps(collect(),ensure_ascii=False,allow_nan=False).encode(),'application/json')
            elif self.path=='/healthz':self.respond(200,b'{"status":"ok"}','application/json')
            else:self.respond(404,b'Not found','text/plain')
        except Exception as e:self.respond(503,json.dumps(dict(error=repr(e))).encode(),'application/json')
    do_HEAD=do_GET
    def readonly(self):self.respond(405,b'Read only','text/plain')
    do_POST=readonly;do_PUT=readonly;do_DELETE=readonly;do_PATCH=readonly
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    with ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.daemon_threads=True;server.serve_forever(poll_interval=.5)
