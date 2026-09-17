"""Read-only transport revision display on the unchanged original port."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
PARENT=HERE.parent/'monitor_charts_r2r_adapt_v5/server.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='9f982e7510eb3acc101e1783927aac61d64b86f5daf7d1d630339b0483f61e83'
s=importlib.util.spec_from_file_location('r2r_transport_monitor',PARENT)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.h.REVIEW=LINE/'reviews/Q35N_ORDINARY_R2R_ADAPT_V5_TRANSPORT_R1'
def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu)
    d['monitor_version']='ordinary_r2r_adapt_v5r1'
    d['navigation']['continued']=m.h.prior.eval_state('ordinary_r2r_adapt_dev_v5r1')
    d['eta_note']+=' 初次评测启动缺少依赖，0动作失败已保留；当前仅补齐依赖重接相同检查点/100条评测。'
    d['predecessor_startup_failure']=m.h.read(LINE/'reviews/Q35N_ORDINARY_R2R_ADAPT_V5/WORKFLOW_RESULT.json')
    return d
def html():return (HERE/'index.html').read_bytes()
m.r.b.collect=collect
class Handler(m.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        elif self.path=='/refresh.js':self.respond(200,(HERE/'refresh.js').read_bytes(),'application/javascript')
        else:super().do_GET()
    do_HEAD=do_GET
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    with m.r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.daemon_threads=True;server.serve_forever(poll_interval=.5)
