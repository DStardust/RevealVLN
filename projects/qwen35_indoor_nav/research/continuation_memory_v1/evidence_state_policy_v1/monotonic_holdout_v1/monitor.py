"""Local read-only progress: physical collection, sealed groups and full denominators."""
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
old=load('holdout_monitor_readers',PARENT/'gpu_runtime_r1/monitor_v1/server.py')

class Reader:
    def __init__(self,run):self.run=run;self.verified={}
    def snapshot(self):
        run=self.run;cfg=read(run/'PROTOCOL.json');state=old.optional(run/'STATUS.json',{})
        collected=[dict(house=h,**old.optional(run/'collect'/h/'STATUS.json',dict(stage='PENDING',parents=0,variants=0))) for h in cfg['houses']]
        registry=old.optional(run/'EVALUATION_REGISTRY.json');groups={};rows=[];errors=[]
        if registry:
            for session in sorted((run/'evaluate').glob('session_*')):
                for path in session.glob('GROUP_*.json'):
                    seal=session/('STATE_SEAL_'+path.stem.split('_')[1]+'.json')
                    if not seal.exists():continue
                    if path not in self.verified:
                        value=read(path);s=read(seal)
                        if not s['base_unchanged'] or not s['heads_unchanged']:raise ValueError('STATE_CHANGED')
                        ranks=[r['rank'] for r in registry['slots'] if r['condition']==value['condition']]
                        if value['ranks']!=ranks or s['completed_ranks']!=ranks:raise ValueError('INCOMPLETE_GROUP')
                        if len(value['audits'])!=3 or any(not a['input_prefix_matched'] or not a['action_prefix_matched'] or a['argmax_flip_count'] for a in value['audits']):raise ValueError('PREFIX_AUDIT')
                        for name,digest_ in value['files'].items():
                            if sha(session/name)!=digest_:raise ValueError('GROUP_HASH')
                        records=[]
                        for slot in registry['slots']:
                            if slot['condition']!=value['condition']:continue
                            r=read(session/'rollouts'/f"{slot['rank']:04d}"/'TASK_RESULT.json')
                            records.append(dict(slot,**registry['conditions'][slot['condition']],label=r['safe_v16_label']))
                        self.verified[path]=(value['condition'],records)
                    condition,records=self.verified[path]
                    if condition in groups:raise ValueError('DUPLICATE_COMPLETE_GROUP')
                    groups[condition]=records;rows.extend(records)
        metrics=[]
        for endpoint in ('main','control'):
            for arm in cfg['arms']:
                r=[r for r in rows if r['endpoint']==endpoint and r['arm']==arm];counts=Counter(x['label'] for x in r)
                metrics.append(dict(endpoint=endpoint,arm=arm,complete=len(r),planned=cfg[endpoint+'_denominator_per_arm'],passed=counts['PASS'],failed=counts['FAIL'],unknown=counts['UNKNOWN']))
        attempts=sorted((run/'attempts').glob('*/stdout.log'),key=lambda p:p.stat().st_mtime)
        ledger=c.records(run/'RESOURCES.jsonl') if (run/'RESOURCES.jsonl').exists() else []
        return dict(status=state,collection=collected,groups=len(groups),planned_groups=cfg['planned_conditions'],complete=len(rows),planned=cfg['planned_rollouts'],metrics=metrics,
            gpu_hours=max(sum(r['gpu_hours'] for r in ledger),state.get('gpu_hours',0)),new_training_updates=0,
            log=old.tail(attempts[-1],5000) if attempts else '',result=old.optional(run/'RESULT.json'),errors=errors)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=18770);p.add_argument('--once',action='store_true');args=p.parse_args();reader=Reader(args.run)
    if args.once:print(json.dumps(reader.snapshot(),ensure_ascii=False));return
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':body=(HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(reader.snapshot(),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
            except (OSError,ValueError,KeyError) as exc:self.send_error(503);print(repr(exc),flush=True)
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
