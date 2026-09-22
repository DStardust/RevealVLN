"""Read-only event probes, six matched policy jobs and sealed DEV groups."""
from html import escape
from http.server import BaseHTTPRequestHandler,HTTPServer
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
original=load('stop_eval_monitor_reader',PARENT/'monotonic_holdout_v1/monitor.py')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();reader=original.Reader(args.run)
    def snapshot():
        value=reader.snapshot();run=args.run
        value['probe']=[dict(model=p.parent.name,**read(p)) for p in (run/'event_probe').glob('*/PROGRESS.json')]
        value['event_preflight']=read(run/'EVENT_PREFLIGHT.json') if (run/'EVENT_PREFLIGHT.json').exists() else None
        cfg=read(run/'PROTOCOL.json');value['training']=[]
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                folder=run/'train'/f'{arm}_{seed}';progress=read(folder/'PROGRESS.json') if (folder/'PROGRESS.json').exists() else {}
                value['training'].append(dict(model=f'{arm}_{seed}',step=progress.get('step',0),target=cfg['steps'],complete=(folder/'RESULT.json').exists()))
        value['new_training_updates']=sum(x['step'] for x in value['training'])
        value['probe_updates']=sum(x['step'] for x in value['probe'])
        return value
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':body=(HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(snapshot(),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
                elif self.path=='/report':
                    path=args.run/'REPORT_ZH.md';report=path.read_text() if path.exists() else (HERE/'README_ZH.md').read_text()
                    body=('<meta charset="utf-8"><pre style="white-space:pre-wrap">'+escape(report)+'</pre>').encode();mime='text/html; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
            except (OSError,ValueError,KeyError) as exc:self.send_error(503);print(repr(exc),flush=True)
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',18770),Handler).serve_forever()

if __name__=='__main__':main()
