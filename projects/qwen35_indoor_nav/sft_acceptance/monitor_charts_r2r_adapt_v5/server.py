"""Read-only original-port monitor for the single R2R source adaptation."""
import hashlib,importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'monitor_charts_continue_v2/server.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='a6899d8c31f8dd7651057d53933d6252e4fe88cc5dfc38572ca149b5c0baafec'
s=importlib.util.spec_from_file_location('r2r_monitor_parent',OLD)
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
r=h.r;TRAIN=HERE.parent/'ordinary_r2r_adapt_v5'
h.TRAIN=TRAIN;h.REVIEW=LINE/'reviews/Q35N_ORDINARY_R2R_ADAPT_V5'
r.RECOVERY=TRAIN;r.FORMAL=TRAIN/'formal'
def collect(include_gpu=True):
    d=h.collect(include_gpu=include_gpu)
    d.update(monitor_version='ordinary_r2r_adapt_v5',current_segment_planned_decisions=396001,
             source_checkpoint_updates=4000,controller_globally_enabled=False,
             eta_note='固定4000→8000；新增396001个R2R决策。70分钟上限，完成后单次100条闭环；不自动重试。',
             main_gate='相对较好4000步：SR提高、SPL不降、nDTW下降不超过1个百分点。仅已暴露DEV工程信号。')
    d['navigation']['continued']=h.prior.eval_state('ordinary_r2r_adapt_dev_v5')
    for k,name in [('high_lr','ordinary_expanded_continue_dev_v2'),('low_lr','ordinary_expanded_low_lr_dev_v3'),
                   ('stop_calibrated','ordinary_stop_calibrated_dev_v1'),('full_epoch','ordinary_full_epoch_dev_v4')]:
        d['navigation'][k]=h.prior.eval_state(name)
    d['estimated_segment_remaining_seconds']=None
    pts=[p for p in d['points'] if p['segment']==d['run_name']]
    if d['display_state']=='TRAINING' and len(pts)>1:
        a,b=pts[max(0,len(pts)-21)],pts[-1];dt=b['unix']-a['unix'];du=b['updates']-a['updates']
        if dt>0 and du>0:d['estimated_segment_remaining_seconds']=max(0,8000-b['updates'])*dt/du
    d['sampling_plan_summary']=h.read(TRAIN/'RESUME_AUDIT.json')
    return d
def html():return (HERE/'index.html').read_bytes()
r.b.collect=collect
class Handler(h.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        elif self.path=='/refresh.js':self.respond(200,(HERE/'refresh.js').read_bytes(),'application/javascript')
        else:super().do_GET()
    do_HEAD=do_GET
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    with r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.daemon_threads=True;server.serve_forever(poll_interval=.5)
