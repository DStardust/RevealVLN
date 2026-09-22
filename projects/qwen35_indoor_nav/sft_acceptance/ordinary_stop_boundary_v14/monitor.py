"""Add ordinary navigation to the existing 18770 page; long-train reader unchanged."""
import json,runpy,sys,time
from pathlib import Path
from http.server import HTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
review=u.load('stop14_monitor_review',u.HERE/'review.py')
old=u.LINE/'research/continuation_memory_v1/evidence_state_policy_v1/expanded_longtrain_v1'
sys.path.insert(0,str(old))
module=runpy.run_path(str(old/'monitor.py'),run_name='ordinary_monitor_reader')
def server(address,handler):
    class Combined(handler):
        def do_GET(self):
            if self.path not in ('/','/api/ordinary'):return super().do_GET()
            try:
                if self.path=='/':body=(u.HERE/'index.html').read_bytes();mime='text/html; charset=utf-8'
                else:
                    run=u.HERE/'runs'/(u.HERE/'LAST_RUN.txt').read_text().strip()
                    result=review.summarize(run);result['status_record']=u.read(run/'STATUS.json') if (run/'STATUS.json').exists() else {}
                    result['training']=u.read(run/'TRAIN_RESULT.json') if (run/'TRAIN_RESULT.json').exists() else None
                    result['sessions']=[u.read(p) for p in run.glob('sessions/*/PROGRESS.json')]
                    body=json.dumps(result,ensure_ascii=False).encode();mime='application/json'
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except Exception as e:self.send_error(503);print(repr(e),flush=True)
    return HTTPServer(address,Combined)
module['main'].__globals__['HTTPServer']=server
if __name__=='__main__':module['main']()
