"""Read-only progress for frozen references, event repair and complete paired groups."""
from html import escape
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
reader_module=load('preservation_monitor_reader',PARENT/'monotonic_holdout_v1/monitor.py')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    reader=reader_module.Reader(args.run)
    def snapshot():
        value=reader.snapshot();cfg=read(args.run/'PROTOCOL.json');value['training']=[]
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                folder=args.run/'train'/f'{arm}_{seed}';p=folder/'PROGRESS.json'
                step=read(p)['step'] if p.exists() else 0
                value['training'].append(dict(model=f'{arm}_{seed}',step=step,target=cfg['model_updates'][arm],
                    frozen=arm=='ORIGINAL',complete=(folder/'RESULT.json').exists()))
        value['new_training_updates']=sum(r['step'] for r in value['training'])
        value['train_verified']=(args.run/'TRAINING_REVIEW.json').exists()
        return value
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':body=(HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(snapshot(),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
                elif self.path=='/report':
                    path=args.run/'REPORT_ZH.md';path=path if path.exists() else HERE/'README_ZH.md'
                    body=('<meta charset="utf-8"><pre style="white-space:pre-wrap">'+escape(path.read_text())+'</pre>').encode();mime='text/html; charset=utf-8'
                elif self.path=='/diagnosis':
                    path=PARENT/'event_grounding_localization_v1/runs/diagnosis_002/REPORT_ZH.md'
                    body=('<meta charset="utf-8"><pre style="white-space:pre-wrap">'+escape(path.read_text())+'</pre>').encode();mime='text/html; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
            except (OSError,KeyError,ValueError) as exc:self.send_error(503);print(repr(exc),flush=True)
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',18770),Handler).serve_forever()


if __name__=='__main__':main()
