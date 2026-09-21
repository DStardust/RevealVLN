"""Same progress URL; show real readout updates, gate and sealed six-model groups."""
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
original=load('repair_monitor_reader',PARENT/'monotonic_holdout_v1/monitor.py')

class Reader(original.Reader):
    def snapshot(self):
        value=super().snapshot();run=self.run
        value['training']=[dict(original.old.optional(run/f'TRAIN_PROGRESS_{s}.json',dict(stage='PENDING',step=0,target=1200)),seed=s) for s in read(run/'PROTOCOL.json')['seeds']]
        value['new_training_updates']=sum(r['step'] for r in value['training'])
        value['local_gate']=original.old.optional(run/'LOCAL_GATE.json')
        return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=18770);p.add_argument('--once',action='store_true');args=p.parse_args();reader=Reader(args.run)
    if args.once:print(json.dumps(reader.snapshot(),ensure_ascii=False));return
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':body=(HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(reader.snapshot(),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
                elif self.path in ('/report','/diagnosis'):
                    path=args.run/'REPORT_ZH.md' if self.path=='/report' else PARENT/'monotonic_diagnosis_v1/runs/diagnosis_001/REPORT_ZH.md'
                    report=path.read_text() if path.exists() else '尚在运行，最终报告未生成。'
                    body=('<meta charset="utf-8"><style>pre{white-space:pre-wrap}</style><a href="/">返回监控</a><pre>'+escape(report)+'</pre>').encode();mime='text/html; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
            except (OSError,ValueError,KeyError) as exc:self.send_error(503);print(repr(exc),flush=True)
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
