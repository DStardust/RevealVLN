"""Read-only original-port monitor for the fixed correction training and evaluation."""
import hashlib,importlib.util,json,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
PARENT=HERE.parent/'monitor_charts_route_teacher_training_v11/server.py'
s=importlib.util.spec_from_file_location('previous_data_monitor',PARENT)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
TRAIN=HERE.parent/'ordinary_stop_row_v12'
def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None
def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu);d['monitor_version']='ordinary_stop_row_v12r1'
    d['stop_row']=dict(build=read(TRAIN/'BUILD_RESULT.json'),extraction=read(TRAIN/'EXTRACTION_PROGRESS.json'),
      extraction_result=read(TRAIN/'EXTRACTION_RESULT.json'),resource=read(TRAIN/'RESOURCE_R1.json') if (TRAIN/'PREFLIGHT_R1.json').exists() else read(TRAIN/'RESOURCE.json'),launch=read(TRAIN/'LAUNCH_R1_RESULT.json') if (TRAIN/'PREFLIGHT_R1.json').exists() else read(TRAIN/'LAUNCH_RESULT.json'),prior_transport_failure=read(TRAIN/'LAUNCH_RESULT.json'),
      parity=read(TRAIN/'FEATURE_PARITY.json'),probe=read(TRAIN/'PROBE_RESULT.json'),fit=read(TRAIN/'FINAL_RESULT.json'),
      fit_eval=read(LINE/'closed_loop_bench/ordinary_stop_row_fit_v12/run_001/PROGRESS.json'),
      fit_report=read(LINE/'reviews/Q35N_ORDINARY_STOP_ROW_V12/FIT_RESULT.json'),
      dev_eval=read(LINE/'closed_loop_bench/ordinary_stop_row_dev_v12/run_001/PROGRESS.json'),
      result=read(LINE/'reviews/Q35N_ORDINARY_STOP_ROW_V12/RESULT.json'))
    d['stop_row']['positive_navigation_result']=bool(d['stop_row']['result'] and d['stop_row']['result']['positive_development_signal'])
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
