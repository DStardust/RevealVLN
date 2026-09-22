"""Live combined progress of both production batches, never hidden denominator changes."""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cfg=read(a.run/'PROTOCOL.json')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':data=(HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
                elif self.path=='/api/status':
                    value=read(a.run/'STATUS.json')
                    if (a.run/'COUNTS.json').exists():value['counts']=read(a.run/'COUNTS.json')
                    old=read(Path(cfg['prior_run'])/'STATUS.json')
                    value['prior_status']=old['status'];value['prior_workers']=old.get('workers',[])
                    value['run']=str(a.run)
                    data=json.dumps(value,ensure_ascii=False).encode();mime='application/json; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
            except (OSError,ValueError) as exc:self.send_error(503,str(exc))
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',18770),Handler).serve_forever()
if __name__=='__main__':main()

