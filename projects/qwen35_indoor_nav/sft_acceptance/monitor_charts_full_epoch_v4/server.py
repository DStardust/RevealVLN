"""Original 18766 interface: full-epoch coverage, all failed branches retained."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'monitor_charts_stop_calibration_v1/server.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='12d519b074cac115624f6dea2702754336f8c722cfdb7e3046ff60008340e5b2'
s=importlib.util.spec_from_file_location('full_epoch_monitor_parent',OLD)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
h=m.m.m;r=h.r
TRAIN=HERE.parent/'ordinary_expanded_full_epoch_v4'
REVIEW=LINE/'reviews/Q35N_ORDINARY_FULL_EPOCH_V4'
h.TRAIN=TRAIN;h.REVIEW=REVIEW;r.RECOVERY=TRAIN;r.FORMAL=TRAIN/'formal'


def points(segments):
    stages=[]
    for label,name in [('expanded_stage1','ordinary_expanded_v1'),('original_stage2','ordinary_expanded_continue_v2')]:
        records=r.b.tail_records(HERE.parent/name/'formal/attempt_001/PROGRESS.jsonl')['records']
        stages.append((label,records))
    return h.old_points(stages+segments)
r.points_for_segments=points


def collect(include_gpu=True):
    d=m.collect(include_gpu=include_gpu)
    final=h.read(TRAIN/'formal/attempt_001/RESULT.json');lease=h.read(TRAIN/'lease_v1/LEASE_RESULT.json')
    p=d['progress']['data'] or {};cursor=p.get('cursor',final.get('cursor',{}))
    updates=cursor.get('updates',8000);decisions=p.get('global_plan_decisions',final.get('global_decisions',682651))
    done=final.get('status')=='EPOCHS_COMPLETED' and cursor.get('epoch')==1 and cursor.get('position')==0 and updates==31059
    d.update(monitor_version='ordinary_full_epoch_v4',source_checkpoint_updates=8000,
        current_segment_new_updates=max(0,updates-8000),current_segment_new_decisions=max(0,decisions-682651),
        current_segment_planned_decisions=1967696,epoch_total_decisions=2650347,decisions=decisions,
        training_completion=dict(completed_budget=done,stop_reason=final.get('stop'),
            holder_restoration_verified=lease.get('holders_restored') is True,latest_checkpoint=final.get('latest_checkpoint'),
            new_updates=max(0,updates-8000),cumulative_expanded_updates=updates,actions=max(0,decisions-682651),automatic_training=False),
        full_epoch_target_updates=31059,full_epoch_coverage=decisions/2650347,
        automatic_next_epoch=False,failed_low_lr_extra_compute_decisions=341939,
        eta_note='固定一轮31,059更新；最多4小时，不自动第二轮，不启用停止校准或视觉停滞保护。')
    d['navigation']['full_epoch']=h.prior.eval_state('ordinary_full_epoch_dev_v4')
    d['navigation']['stop_calibrated']=h.prior.eval_state('ordinary_stop_calibrated_dev_v1')
    d['stop_calibration']['note']='Historical closed calibration, not adopted; current full-epoch training has no STOP offset.'
    if done:d['display_state']='完整一轮已完成，资源已恢复' if lease.get('holders_restored') else '完整一轮已完成，等待资源恢复'
    elif final:d['display_state']='本轮有界停止，未完成整轮' if final.get('status')=='STOPPED' else d['display_state']
    if d['display_state']=='TRAINING':
        pts=[x for x in d['points'] if x['segment']==d['run_name']]
        if len(pts)>1:
            a,b=pts[max(0,len(pts)-21)],pts[-1];dt=b['unix']-a['unix'];du=b['updates']-a['updates']
            if dt>0 and du>0:
                eta=max(0,31059-updates)*dt/du
                d['estimated_segment_remaining_seconds']=eta
                d['estimated_completion_within_budget']=eta<=d['wall_remaining_seconds']
    return d


def html():
    text=h.html().decode()
    start=text.index('<section class="panel full" style="margin-top:20px">');end=text.index('</main>',start)
    text=text[:start]+(HERE/'panel.html').read_text()+text[end:]
    text=text.replace('首段已完成4,000更新，当前继续至累计8,000','保留4000步较好基座，原配方从8000继续至完整一轮31,059')
    text=text.replace(' / 4,000',' / 23,059')
    return text.encode()


r.b.collect=collect
class Handler(h.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        else:super().do_GET()
    do_HEAD=do_GET


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    html()
    with r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.daemon_threads=True;server.serve_forever(poll_interval=.5)
