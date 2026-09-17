"""Read-only V6 data supervision phase, with closed V5 failures retained."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
PARENT=HERE.parent/'monitor_charts_r2r_adapt_v5r1/server.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='6cf0685ea44f97a9f9afeb114218bb454b3660ee8849258f474aba186b84d818'
s=importlib.util.spec_from_file_location('onpolicy_monitor_parent',PARENT)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
DATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v1'
def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu);read=m.m.h.read
    d['monitor_version']='ordinary_onpolicy_v6'
    progress=read(DATA/'run_001/PROGRESS.json');result=read(DATA/'run_001/RESULT.json')
    failure=read(DATA/'run_001/COLLECTION_FAILURE.json') or read(DATA/'run_001/LAUNCH_FAILURE.json')
    d['onpolicy_recovery']=dict(progress=progress,result=result,failure=failure,
        launch=read(DATA/'run_001/LAUNCH_RESULT.json'),prepared=(DATA/'SOURCE_LOCK.json').exists(),
        training_started=False,navigation_gain_verified=False)
    return d
def html():return (HERE/'index.html').read_bytes()
m.m.r.b.collect=collect
class Handler(m.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        elif self.path=='/refresh.js':self.respond(200,(HERE/'refresh.js').read_bytes(),'application/javascript')
        else:super().do_GET()
    do_HEAD=do_GET
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    with m.m.r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.daemon_threads=True;server.serve_forever(poll_interval=.5)
