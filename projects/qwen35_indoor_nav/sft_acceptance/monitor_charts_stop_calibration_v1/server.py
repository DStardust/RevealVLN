"""Existing 18766 page plus FIT-only stop-calibration progress."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'monitor_charts_low_lr_v3/server.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='cef27fd46bae417cb35959f7e2267516b819fcbc241036d951d47e99bab1f973'
s=importlib.util.spec_from_file_location('calibration_monitor_parent',OLD)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
REVIEW=LINE/'reviews/Q35N_ORDINARY_STOP_CALIBRATION_V1'


def read_optional(path):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return None


def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu)
    d['monitor_version']='ordinary_stop_calibration_v1'
    d['stop_calibration']=dict(
        fit=m.m.prior.eval_state('ordinary_stop_calibration_fit_v2'),
        prior_fit_failure=read_optional(LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v1/run_001/LAUNCH_RESULT.json'),
        calibration=read_optional(REVIEW/'CALIBRATION.json'),
        dev=read_optional(LINE/'closed_loop_bench/ordinary_stop_calibrated_dev_v1/run_001/PROGRESS.json'),
        dev_result=read_optional(LINE/'closed_loop_bench/ordinary_stop_calibrated_dev_v1/run_001/RESULT.json'),
        decision=read_optional(REVIEW/'FINAL_DECISION.json'),
        note='Only FIT houses select the STOP offset; best uncalibrated checkpoint remains 4000. No training or special-data mixing.')
    return d


def html():
    text=m.html().decode()
    panel=(HERE/'panel.html').read_text()
    text,n=re.subn(r'(<main[^>]*>)',lambda match:match.group(1)+panel,text,count=1)
    assert n==1
    return text.encode()


m.m.r.b.collect=collect
class Handler(m.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        else:super().do_GET()
    do_HEAD=do_GET


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    html()
    with m.m.r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.daemon_threads=True;server.serve_forever(poll_interval=.5)
