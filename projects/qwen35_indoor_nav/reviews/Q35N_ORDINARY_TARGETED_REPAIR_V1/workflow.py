"""Finite serial evaluation handoff. No training, retries, or foreign signals."""
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
BASE=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1'
GUARD=LINE/'closed_loop_bench/ordinary_visual_stall_guard_v1'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def status(state,**extra):
    obj=dict(status=state,unix=time.time(),automatic_training=False,automatic_retry=False,**extra)
    tmp=HERE/'WORKFLOW_STATUS.tmp';tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(HERE/'WORKFLOW_STATUS.json')


def freeze():
    assert not (HERE/'WORKFLOW_SEAL.json').exists(),'ALREADY_FROZEN'
    t=subprocess.run([str(PY),'-I','-S','-B',str(HERE/'test_guard_audit.py')],cwd=ROOT,capture_output=True,text=True,timeout=60)
    save(HERE/'GUARD_AUDIT_CPU_RESULT.json',dict(passed=t.returncode==0,output=t.stderr,synthetic_contract_test_only=True))
    assert t.returncode==0,t.stderr
    files={str(p):sha(p) for p in list(HERE.glob('*.py'))+[HERE/'WORKFLOW_SPEC_ZH.md',GUARD/'SOURCE_LOCK.json',BASE/'SOURCE_LOCK.json']}
    for case in (BASE,GUARD):
        assert read(case/'CPU_TEST_RESULT.json')['passed']
        for p,d in read(case/'SOURCE_LOCK.json')['files'].items():assert sha(p)==d,p
    save(HERE/'WORKFLOW_SEAL.json',dict(files=files,baseline_input_sha256=sha(BASE/'EPISODES_PRIVILEGED.json'),
         guard_input_sha256=sha(GUARD/'EPISODES_PRIVILEGED.json'),new_training=False))
    print('FROZEN: CPU guard and independent audit checks passed')


def report(stage):
    with (HERE/f'analyze_{stage}.log').open('x') as log:
        result=subprocess.run([str(PY),'-I','-S','-B',str(HERE/'analyze.py'),stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert result.returncode==0,'FAILURE_ANALYSIS_FAILED'
    a=read(HERE/f'ANALYSIS_{stage.upper()}.json')
    text='# 普通导航针对性检查：'+('同批量复测' if stage=='matched' else '执行保护对照')+'\n\n'
    text+='同一批 100 条 INTERNAL_DEV / 5 屋，非官方全量或隐藏测试。原 batch=8 结果保留为数值敏感性参考。\n\n'
    text+='| 版本 | 实际批量 | SR | SPL | nDTW | 原地重复≥8步路线 |\n|---|---:|---:|---:|---:|---:|\n'
    labels=dict(before='扩产前纯模型',after_batch8='扩产后原 batch=8',after_matched_single='扩产后纯模型 batch=1',guard='扩产后模型＋视觉停滞保护')
    for k,row in a['cases'].items():
        r=row['result'];text+=f"| {labels[k]} | {r['selected_batch_size']} | {100*r['sr']:.2f}% | {100*r['spl']:.2f}% | {100*r['ndtw']:.2f}% | {row['episodes_with_stagnant_run_at_least_8']} |\n"
    text+='\n## 配对与解释\n\n'
    for key,label in [('corrected_model_comparison','扩产后纯模型对扩产前'),('guard_vs_pure_model','保护组对扩产后纯模型')]:
        if key not in a['pairs']:continue
        p=a['pairs'][key];delta=p['delta'];signal_ok=delta['success']>0 and delta['spl']>=-.02
        text+=f"{label}：新增成功 {int(p['wins'])} 条，失去成功 {int(p['losses'])} 条；SR 变化 {100*delta['success']:+.2f} 个百分点，SPL {100*delta['spl']:+.2f}，nDTW {100*delta['ndtw']:+.2f}。注册工程信号：{'满足' if signal_ok else '不满足'}。\n\n"
        text+='| 房屋 | ΔSR | ΔSPL | ΔnDTW |\n|---|---:|---:|---:|\n'
        for house,d in p['by_house_delta'].items():text+=f"| {house} | {100*d['success']:+.2f} | {100*d['spl']:+.2f} | {100*d['ndtw']:+.2f} |\n"
        text+='\n五屋配对 bootstrap 与留一屋详见 JSON，只描述本批开发波动；不得据总均值小正号宣称稳定提升。\n\n'
    if stage=='guard':
        g=a['cases']['guard']['result']['guard_audit']
        text+=f"保护独立因果审核通过；{g['intervened_episodes']} 条路线共 {g['total_interventions']} 次干预，STOP 改写 0 次，隐藏额外动作 0 次。保护得分必须标为模型＋控制器，不能替换纯模型分数。\n\n"
    text+='训练参数未更新，特殊数据未混入；旧失败、检查点、源码与输入锁均保留。本节点没有自动续训或改阈值重跑。下一步若要训练恢复行为，应先在 FIT 房屋独立采集离轨/恢复监督，保留专家零碰撞池原标准。\n'
    with (HERE/f'REPORT_{stage.upper()}_ZH.md').open('x') as f:f.write(text)
    return a


def main():
    lease=(HERE/'workflow.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'WORKFLOW_RESULT.json').exists(),'WORKFLOW_ALREADY_CLOSED'
    for p,digest in read(HERE/'WORKFLOW_SEAL.json')['files'].items():assert sha(p)==digest,p
    child=None
    def interrupted(signum,frame):raise RuntimeError('WORKFLOW_SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        began=time.monotonic()
        while not (BASE/'run_001/RESULT.json').exists():
            for name in ('LAUNCH_FAILURE.json','INFERENCE_FAILURE.json','AUDIT_FAILURE.json'):
                assert not (BASE/'run_001'/name).exists(),'BASELINE_FAILED:'+name
            if (BASE/'run_001/LAUNCH_RESULT.json').exists():assert read(BASE/'run_001/LAUNCH_RESULT.json')['status']=='COMPLETE'
            assert time.monotonic()-began<=4200,'BASELINE_WAIT_BUDGET'
            status('WAITING_MATCHED_BASELINE',waited_seconds=time.monotonic()-began)
            time.sleep(10)
        r=read(BASE/'run_001/RESULT.json')
        assert r['status']=='COMPLETE' and r['completed']==100 and r['trace_audit_passed'] and r['selected_batch_size']==1 and r['optimizer_updates']==0
        bp,gp=read(BASE/'PROTOCOL.json'),read(GUARD/'PROTOCOL.json')
        assert r['checkpoint_sha256']==bp['checkpoint_sha256']==gp['checkpoint_sha256']
        for key in ('checkpoint','seed','max_steps','success_distance','environment_seed','lanes','parity_pass_batch_size','parity_fallback_batch_size'):
            assert bp[key]==gp[key]
        assert sha(BASE/'EPISODES_PRIVILEGED.json')==sha(GUARD/'EPISODES_PRIVILEGED.json')
        status('ANALYZING_MATCHED_BASELINE');report('matched')
        assert not (GUARD/'run_001').exists(),'GUARD_ALREADY_ATTEMPTED_NO_RETRY'
        status('GUARD_RUNNING',baseline_audited=True)
        with (HERE/'guard_launcher.log').open('x') as log:
            child=subprocess.Popen([str(PY),'-I','-S','-B',str(GUARD/'launch.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            start=time.monotonic()
            while child.poll() is None:
                assert time.monotonic()-start<=5520,'GUARD_SUPERVISION_WALL_LIMIT'
                status('GUARD_RUNNING',own_launcher_pid=child.pid,elapsed_seconds=time.monotonic()-start)
                time.sleep(10)
            assert child.returncode==0,'GUARD_LAUNCH_OR_AUDIT_FAILED'
        status('ANALYZING_GUARD');a=report('guard')
        delta=a['pairs']['guard_vs_pure_model']['delta']
        final=dict(status='COMPLETE',unix=time.time(),paired_delta=delta,
            guard_development_signal=delta['success']>0 and delta['spl']>=-.02,
            scientific_gain_verified=False,automatic_training=False,automatic_retry=False,
            controller_globally_enabled=False,foreign_processes_signaled=[])
        save(HERE/'WORKFLOW_RESULT.json',final);status('COMPLETE',**{k:v for k,v in final.items() if k not in ('status','unix','automatic_training','automatic_retry')})
    except BaseException as exc:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()  # Exact Popen child, whose finally cleans only its own evaluator session.
            child.wait(timeout=60)
        final=dict(status='FAILED_OR_BLOCKED',unix=time.time(),error=repr(exc),traceback=traceback.format_exc(),automatic_training=False,automatic_retry=False,foreign_processes_signaled=[])
        save(HERE/'WORKFLOW_RESULT.json',final);status('FAILED_OR_BLOCKED',error=repr(exc));raise
    finally:lease.close()


if __name__=='__main__':
    if sys.argv[1:]==['freeze']:freeze()
    else:
        assert sys.argv[1:]==['run'];main()
