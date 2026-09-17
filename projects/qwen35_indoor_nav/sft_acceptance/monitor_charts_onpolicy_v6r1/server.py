"""Original-port data status; typed-distance transport revision only."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
PARENT=HERE.parent/'monitor_charts_onpolicy_v6/server.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='0543553e21855855156ed658be7185d5cb3fdfb9e3386092df806276f0ce2589'
s=importlib.util.spec_from_file_location('onpolicy_v2_monitor_parent',PARENT)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.DATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v2'
def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu);d['monitor_version']='ordinary_onpolicy_v6r1'
    d['onpolicy_recovery']['transport_revision']='V1距离对象类型错误在0动作退出，已保留；V2仅修复Episode缓存适配。'
    return d
def html():return (HERE/'index.html').read_bytes()
m.m.m.r.b.collect=collect
class Handler(m.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        elif self.path=='/refresh.js':self.respond(200,(HERE/'refresh.js').read_bytes(),'application/javascript')
        else:super().do_GET()
    do_HEAD=do_GET
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    with m.m.m.r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.daemon_threads=True;server.serve_forever(poll_interval=.5)
