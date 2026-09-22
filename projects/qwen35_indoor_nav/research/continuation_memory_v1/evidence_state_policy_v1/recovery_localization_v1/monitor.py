"""Same local website: measured positives, final negative result and current diagnosis."""
from html import escape
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import HERE,BASE,read

def main():
    out=HERE/'runs/diagnosis_001'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/':
                body=(HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
            elif self.path=='/api/status':
                p=read(out/'CAUSAL_PROBE.json');r=read(out/'ORIGINAL_RECOVERY.json')
                value=dict(status=read(out/'STATUS.json'),evidence=read(out/'EVIDENCE_LEDGER.json'),
                    probe_summary=[x for x in p['summary'] if x['seed'] is None and x['history']=='all'],
                    recovery_summary=[x for x in r['summary'] if x['seed'] is None],
                    next_step=read(out/'NEXT_STEP.json'),new_training=False,gpu_hours=0)
                body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
            elif self.path in ('/report','/last-experiment'):
                p=out/'REPORT_ZH.md' if self.path=='/report' else BASE/'teacher_alignment_v1/runs/aligned_001/REPORT_ZH.md'
                body=('<meta charset="utf-8"><pre style="white-space:pre-wrap">'+escape(p.read_text())+'</pre>').encode();mime='text/html; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',18770),Handler).serve_forever()

if __name__=='__main__':main()
