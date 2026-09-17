"""Local read-only dashboard and one-shot completion handoff; no GPU access."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

OUT=Path(__file__).resolve().parent
REC=OUT/'recovery_r1'
LIVE=OUT/'live'
ROOT=OUT.parents[3]
PORT=18765

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def records(path):
    try:raw=path.read_text()
    except FileNotFoundError:return []
    result=[]
    for line in raw.splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass  # Ignore only an in-flight last record.
    return result

def alive(pid):
    if pid is None:return False
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def snapshot():
    updates=records(REC/'UPDATE_LEDGER.jsonl');old=records(OUT/'UPDATE_LEDGER.jsonl')
    events=records(REC/'EVENTS.jsonl');samples=records(REC/'RESOURCE_SAMPLES.jsonl')
    sample=samples[-1] if samples else {};event=events[-1] if events else {}
    before=read(OUT/'OFFLINE_BEFORE.json');after=read(REC/'OFFLINE_AFTER.json')
    ep=records(OUT/'EPISODE_LEDGER.jsonl')+records(REC/'EPISODE_LEDGER.jsonl')
    episodes={stage:[e for e in ep if e.get('stage')==stage and e.get('event')=='complete'] for stage in ['before','after']}
    proc=read(REC/'WORKER_PROCESS.json') or {};running=alive(proc.get('pid'))
    execution=read(REC/'EXECUTION_RESULT.json');final=read(OUT/'result.json');restored=read(REC/'LEASE_RESTORED.json')
    if final:stage='已完成' if final['decision']=='COMPLETED_BOUNDED_ACCEPTANCE' else '已停止：工程失败'
    elif execution:stage='汇总报告中' if execution['returncode']==0 else '已停止：正在封存失败证据'
    elif after:stage='更新后闭环评估'
    elif len(updates)>=399:stage='终点保存 / 更新后离线评估'
    elif running:stage='训练中'
    else:stage='进程未运行，请查看事件'
    recent=updates[-20:];sec_per_update=None
    if len(recent)>1:sec_per_update=(recent[-1]['unix']-recent[0]['unix'])/(len(recent)-1)
    remaining_train_seconds=(399-len(updates))*sec_per_update if sec_per_update and len(updates)<399 else None
    previous_time=(read(OUT/'EXECUTION_RESULT.json') or {}).get('gpu_stage_wall_seconds',0)
    elapsed=previous_time+(execution['gpu_stage_wall_seconds'] if execution else sample.get('elapsed_seconds',0))
    offline_records=records(REC/'OFFLINE_RECORDS.jsonl')
    last=updates[-1] if updates else {}
    return dict(stage=stage,server_time=time.time(),running=running,last_event=event,recovery_updates=len(updates),recovery_target=399,cumulative_updates=len(old)+len(updates),cumulative_target=400,
        last_ce=last.get('mean_action_ce'),last_grad_norm=last.get('grad_norm_before_clip'),learning_rate=last.get('lr'),seconds_per_update=sec_per_update,remaining_train_seconds=remaining_train_seconds,
        gpu_memory_gib=sample.get('memory_mib',0)/1024,ram_gib=sample.get('worker_tree_rss_bytes',0)/1024**3,output_mib=sample.get('new_output_bytes',0)/1024**2,gpu_elapsed_seconds=elapsed,gpu_budget_seconds=21600,
        loss_curve=[dict(update=r['update'],ce=r['mean_action_ce']) for r in updates],before=before,after=after,after_offline_records=len(offline_records),offline_records_target=72,
        episodes={s:dict(completed=len(rows),total=10,stopped_within_3m=sum(e['stopped_within_3m'] for e in rows),collisions=sum(e['collisions'] for e in rows),action_limits=sum(e['action_limit'] for e in rows)) for s,rows in episodes.items()},
        restored=restored.get('restored') if restored else None,auto_finalization=read(LIVE/'AUTO_FINALIZATION.json'),final_result=final,
        notes=['原运行第1次更新后发生日志错误，已保留；当前从初始checkpoint恢复399次，累计400次。','固定同房屋route-dev，不代表未见房屋泛化。动作准确率不是导航SR；scientific_pass=false。'])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split('?')[0]
        if path=='/':body=(OUT/'progress.html').read_bytes();kind='text/html; charset=utf-8'
        elif path=='/api/status':body=json.dumps(snapshot(),ensure_ascii=False,allow_nan=False).encode();kind='application/json; charset=utf-8'
        elif path=='/report' and (OUT/'REPORT_ZH.md').exists():body=(OUT/'REPORT_ZH.md').read_bytes();kind='text/plain; charset=utf-8'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass

def finalize_when_done():
    while not (REC/'EXECUTION_RESULT.json').exists():time.sleep(5)
    # Launcher writes exit status before restoring its own GPU4 lease.
    deadline=time.time()+180
    while not (REC/'LEASE_RESTORED.json').exists() and time.time()<deadline:time.sleep(2)
    if (OUT/'result.json').exists():return
    with (LIVE/'AUTO_FINALIZATION_STARTED.json').open('x') as f:json.dump(dict(started_unix=time.time(),attempts=1),f)
    try:
        with (LIVE/'finalize.log').open('x') as log:
            p=subprocess.run([sys.executable,'-I','-B',str(OUT/'finalize.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        status=dict(returncode=p.returncode,finished_unix=time.time(),report_available=(OUT/'REPORT_ZH.md').exists(),automatic_retries=0)
        if p.returncode==0:
            r=subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=OUT,capture_output=True,text=True,timeout=900)
            status.update(manifest_check_pass=r.returncode==0,manifest_check_output=r.stdout+r.stderr)
    except Exception as ex:status=dict(error=repr(ex),traceback=traceback.format_exc(),automatic_retries=0)
    with (LIVE/'AUTO_FINALIZATION.json').open('x') as f:json.dump(status,f,indent=2)

if __name__=='__main__':
    LIVE.mkdir(exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    with (LIVE/'SERVER_PROCESS.json').open('x') as f:json.dump(dict(pid=os.getpid(),port=PORT,bind='127.0.0.1',started_unix=time.time()),f)
    threading.Thread(target=finalize_when_done,daemon=True).start()
    print(f'Progress dashboard: http://127.0.0.1:{PORT}',flush=True)
    server.serve_forever()
