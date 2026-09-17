"""Keep 18766; expose a true continuation and matched development gate separately."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'monitor_charts_targeted_v1'
for name,digest in {'server.py':'f0ed45a05cd02a0b852c9a8f0b16aececc0ba449c088736688ed9ff0e943e82a',
                    'panel.html':'a810169b7eea3727839adaa0af246ec969119079e22d50ad0552f47b8412c800',
                    'refresh.js':'c43019a08331a8e8fb0ae412dc3db4b90acd28cffba4b8194ea76c0a62b89f41'}.items():
    assert hashlib.sha256((OLD/name).read_bytes()).hexdigest()==digest
s=importlib.util.spec_from_file_location('continue_monitor_parent',OLD/'server.py')
prior=importlib.util.module_from_spec(s);s.loader.exec_module(prior)
r=prior.m.r
TRAIN=HERE.parent/'ordinary_expanded_continue_v2'
REVIEW=LINE/'reviews/Q35N_ORDINARY_CONTINUE_V2'
r.RECOVERY=TRAIN;r.FORMAL=TRAIN/'formal'
read=prior.read
old_points=r.points_for_segments


def points(segments):
    history=r.b.tail_records(HERE.parent/'ordinary_expanded_v1/formal/attempt_001/PROGRESS.jsonl')['records']
    return old_points([('expanded_stage1',history)]+segments)
r.points_for_segments=points


def collect(include_gpu=True):
    d=r.collect(include_gpu=include_gpu)
    result=read(TRAIN/'formal/attempt_001/RESULT.json');lease=read(TRAIN/'lease_v1/LEASE_RESULT.json')
    p=d['progress']['data'] or {};cursor=p.get('cursor',result.get('cursor',{}))
    completed=result.get('status')=='STOPPED' and result.get('stop')==['BUDGET:max_updates'] and result.get('cursor',{}).get('updates')==8000
    updates=cursor.get('updates',4000);decisions=p.get('global_plan_decisions',result.get('global_decisions',340712))
    d.update(monitor_version='ordinary_continue_v2',epoch_total_decisions=2650347,
        current_segment_new_updates=max(0,updates-4000),current_segment_new_decisions=max(0,decisions-340712),
        current_segment_planned_decisions=341939,source_checkpoint_updates=4000,
        training_completion=dict(completed_budget=completed,stop_reason=result.get('stop'),
            holder_restoration_verified=lease.get('holders_restored') is True,latest_checkpoint=result.get('latest_checkpoint'),
            new_updates=max(0,updates-4000),cumulative_expanded_updates=updates,actions=max(0,decisions-340712),automatic_training=False),
        decisions=decisions,automatic_training=False,controller_globally_enabled=False,
        eta_note='本段最多新增4000更新；累计到8000，完成后仅自动复测，不自动下一轮训练。')
    if completed:d['display_state']='累计8,000步完成，资源已恢复' if lease.get('holders_restored') else '累计8,000步完成，等待资源恢复'
    if result.get('latest_checkpoint'):d['checkpoint_name']=Path(result['latest_checkpoint']).name
    if d['display_state']=='TRAINING':
        pts=[x for x in d['points'] if x['segment']==d['run_name']]
        if len(pts)>1:
            a,b=pts[max(0,len(pts)-11)],pts[-1];dt=b['unix']-a['unix'];du=b['updates']-a['updates']
            if dt>0 and du>0:d['estimated_segment_remaining_seconds']=min(d['wall_remaining_seconds'],max(0,8000-updates)*dt/du)
    d['navigation']={k:prior.eval_state(v) for k,v in dict(before='ordinary_expanded_dev_before_v1',
        matched='ordinary_expanded_dev_after_single_v1',guard='ordinary_visual_stall_guard_v1',
        continued='ordinary_expanded_continue_dev_v2').items()}
    d['continuation_workflow']=read(REVIEW/'WORKFLOW_STATUS.json')
    d['continuation_result']=read(REVIEW/'RESULT.json') or None
    d['last_matched_result']=read(LINE/'reviews/Q35N_ORDINARY_TARGETED_REPAIR_V1/ANALYSIS_GUARD.json').get('pairs',{}).get('corrected_model_comparison')
    return d


def html():
    text=prior.html().decode()
    start=text.index('<section class="panel full" style="margin-top:20px">')
    end=text.index('</main>',start)
    text=text[:start]+(HERE/'panel.html').read_text()+text[end:]
    text=text.replace('最多 4,000 次新更新。265 万动作池','首段已完成4,000更新，当前继续至累计8,000。265 万动作池')
    text=text.replace('本阶段三卡实际计划计数；不是独立路线','扩产两段累计计划计数；不是独立路线')
    text=text.replace("card('计划已完成决策'","card('扩产累计决策'")
    text=text.replace("card('优化器更新',fmt(c.updates),'cursor.updates')","card('扩产累计更新',fmt(c.updates),'本段新增 '+fmt(d.current_segment_new_updates)+' / 4,000')")
    return text.encode()


r.b.collect=collect
class Handler(prior.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        else:super().do_GET()
    do_HEAD=do_GET


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);p.add_argument('--check',action='store_true');a=p.parse_args()
    html()
    if a.check:
        d=collect(False);print(json.dumps({k:d[k] for k in ('monitor_version','display_state','training_completion')},ensure_ascii=False))
    else:
        with r.b.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
            server.daemon_threads=True;server.serve_forever(poll_interval=.5)
