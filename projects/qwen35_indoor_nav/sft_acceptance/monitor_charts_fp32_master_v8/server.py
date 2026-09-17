"""Read-only original-port monitor for the fixed correction training and evaluation."""
import hashlib,importlib.util,json,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
PARENT=HERE.parent/'monitor_charts_onpolicy_v6r1/server.py'
s=importlib.util.spec_from_file_location('previous_data_monitor',PARENT)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
TRAIN=HERE.parent/'ordinary_onpolicy_fp32_master_v8'
REVIEW=LINE/'reviews/Q35N_ORDINARY_FP32_MASTER_V8'
CASE=LINE/'closed_loop_bench/ordinary_fp32_master_dev_v8'
def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None
def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu);d['monitor_version']='ordinary_fp32_master_v8'
    history=[]
    try:
        for line in (TRAIN/'formal/attempt_001/PROGRESS.jsonl').read_text().splitlines():
            try:x=json.loads(line)
            except ValueError:continue
            history.append(dict(updates=x['cursor']['updates'],ce=x['metrics']['mean_ce'],unix=x['unix']))
    except FileNotFoundError:pass
    result=read(REVIEW/'RESULT.json')
    d['onpolicy_training']=dict(precision_bridge=read(TRAIN/'PRECISION_BRIDGE_AUDIT.json'),progress=read(TRAIN/'formal/attempt_001/PROGRESS.json'),
      training_result=read(TRAIN/'formal/attempt_001/RESULT.json'),
      lease_result=read(TRAIN/'lease_v1/LEASE_RESULT.json'),workflow=read(REVIEW/'WORKFLOW_STATUS.json'),
      workflow_result=read(REVIEW/'WORKFLOW_RESULT.json'),first_acceptance=read(REVIEW/'FIRST_CHECKPOINT_ACCEPTANCE.json'),
      final_acceptance=read(REVIEW/'FINAL_CHECKPOINT_ACCEPTANCE.json'),build=read(TRAIN/'BUILD_AUDIT.json'),
      history=history,result=result,eval_progress=read(CASE/'run_001/PROGRESS.json'),
      eval_result=read(CASE/'run_001/RESULT.json'),stage_updates=1000,optimizer_offset=4000,
      positive_navigation_result=None if result is None else result['positive_development_signal'])
    d['onpolicy_recovery']['training_started']=(TRAIN/'lease_v1/LEASE_BEFORE.json').exists()
    d['onpolicy_recovery']['navigation_gain_verified']=bool(result and result['positive_development_signal'])
    return d
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,status,data,kind):
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers()
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
