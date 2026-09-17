"""One bounded training acceptance -> final matched evaluation -> honest report."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_onpolicy_fp32_master_v8r1'
CASE=LINE/'closed_loop_bench/ordinary_fp32_master_dev_v8r1'
BASE=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
QPY=LINE/'.envs/q35n_qwen_g2_v1/bin/python3'
ANALYZER=LINE/'reviews/Q35N_ORDINARY_TARGETED_REPAIR_V1/analyze.py'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def status(state,**extra):
    tmp=HERE/'WORKFLOW_STATUS.tmp';tmp.write_text(json.dumps(dict(status=state,unix=time.time(),automatic_next_training=False,**extra),ensure_ascii=False,indent=2));tmp.replace(HERE/'WORKFLOW_STATUS.json')
def identity(pid):
    p=Path('/proc')/str(pid)
    return dict(pid=pid,start=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),
        cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1])


def freeze():
    assert not (HERE/'SEAL.json').exists(),'ALREADY_FROZEN'
    pid=int(subprocess.check_output(['tmux','display-message','-p','-t','q35n_fp32_master_v8r1','#{pane_pid}'],text=True).strip())
    owner=identity(pid)
    assert owner['argv']==[str(PY),'-I','-S','-B',str(TRAIN/'lease_run.py'),str(TRAIN/'RUNBOOK.json')]
    assert owner['cwd']==str(ROOT)
    assert sha(ANALYZER)=='827e9a1050715d9ca12469eb26379210fa2b23f959f1b73280f258145d46220e'
    paths=list(HERE.glob('*.py'))+[TRAIN/'PROTOCOL_FILESTORE.json',TRAIN/'RUNBOOK.json',TRAIN/'MAIN_AGENT_APPROVAL.json',CASE/'PENDING_SEAL.json',ANALYZER]
    save(HERE/'SEAL.json',dict(files={str(p):sha(p) for p in paths},owned_training_lease=owner,
        primary_metric='sr',positive_gate=dict(sr_delta_strictly_greater_than=0,spl_delta_at_least=0,ndtw_delta_at_least=-.01),
        no_retry=True,no_automatic_next_training=True))
    print('Workflow frozen: one continuation and one final matched evaluation')


def run_logged(argv,name,timeout,env=None):
    with (HERE/name).open('x') as log:
        result=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    assert result.returncode==0,'SUBTASK_FAILED:'+name


def close_report():
    assert sha(ANALYZER)=='827e9a1050715d9ca12469eb26379210fa2b23f959f1b73280f258145d46220e'
    s=importlib.util.spec_from_file_location('same_fixed_analyzer',ANALYZER)
    a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    before=a.case_report(BASE);after=a.case_report(CASE);paired=a.paired(before,after)
    assert paired['matched_inference_batch']
    delta=paired['delta'];positive=delta['success']>0 and delta['spl']>=0 and delta['ndtw']>=-.01
    result=dict(status='COMPLETE',unix=time.time(),before=before,after=after,paired=paired,
      positive_development_signal=positive,scientific_gain_verified=False,controller_enabled=False,
      automatic_next_training=False,checkpoint_selection='fixed newstage1000 optimizer5000 from best4000',
      source_and_input_locks_unchanged=True,build_audit=read(TRAIN/'BUILD_AUDIT.json'),precision_bridge=read(TRAIN/'PRECISION_BRIDGE_AUDIT.json'))
    save(HERE/'RESULT.json',result)
    report='# 普通导航：两组动作参数FP32主权重数值修复\n\n同100条、5屋INTERNAL_DEV；batch1、纯模型、500动作。不是独立论文确认。\n\n'
    report+='| 模型 | SR | SPL | nDTW | OSR |\n|---|---:|---:|---:|---:|\n'
    for label,case in [('当前最佳4000步',before),('最佳4000步＋FP32主权重1000步',after)]:
        r=case['result'];report+=f"| {label} | {100*r['sr']:.2f}% | {100*r['spl']:.2f}% | {100*r['ndtw']:.2f}% | {100*r['osr']:.2f}% |\n"
    report+=f"\n新增成功{int(paired['wins'])}条，失去{int(paired['losses'])}条。事前门槛：{'满足' if positive else '不满足'}。\n\n"
    report+='| 房屋 | ΔSR（百分点） | ΔSPL | ΔnDTW |\n|---|---:|---:|---:|\n'
    for h,d in paired['by_house_delta'].items():report+=f"| {h} | {100*d['success']:+.2f} | {100*d['spl']:+.2f} | {100*d['ndtw']:+.2f} |\n"
    report+='\n纠错数据来自16个FIT房屋的64条原模型实际轨迹，双重物理重放、独立标签和像素审计通过。5365条合法建议排除382条旧输入重合（含54条矛盾）后，新增4983种因果输入。本段99047次训练决策含9875次纠错，比例9.97%，重复抽取不是新增独立数据。\n'
    report+='\n普通导航工程试验：模型结构不变，不是UAD创新。几何教师不保证语言中间路线语义最优；无目标坐标或未来帧输入策略。相对V6的19%对照，只改变两组动作参数与AdamW矩的累积精度；同初始数值、同数据顺序、同1000更新、同学习率。主门槛仍对照最佳4k21%，不是只超过19%就算成功。所有失败、原4k权重和输入锁保留；无自动长训。五屋配对bootstrap和留一屋仅作描述性诊断。\n'
    report+='\n输入注入仍为原BF16，训练时采用FP32 master累积；原推理加载器在参数注入前的BF16转换与其有效前向一致。系统盘满，本版本临时文件改放项目NAS，不改变科学输入。\n'
    with (HERE/'REPORT_ZH.md').open('x') as f:f.write(report)
    return result


def main():
    lease=(HERE/'workflow.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'WORKFLOW_RESULT.json').exists(),'ALREADY_CLOSED'
    seal=read(HERE/'SEAL.json')
    for p,d in seal['files'].items():assert sha(p)==d,p
    def interrupted(sig,frame):raise RuntimeError('WORKFLOW_SIGNAL_'+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    child=None
    try:
        start=time.monotonic()
        while not (TRAIN/'lease_v1/LEASE_RESULT.json').exists():
            assert time.monotonic()-start<2400,'TRAINING_WAIT_BUDGET'
            first=TRAIN/'formal/attempt_001/checkpoint_000000200.pt.json'
            if first.exists() and not (HERE/'FIRST_CHECKPOINT_ACCEPTANCE.json').exists():
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                run_logged([str(QPY),'-I','-B',str(HERE/'accept_first.py')],'first_acceptance.log',300,env)
            status('WAITING_TRAINING',first_checkpoint_accepted=(HERE/'FIRST_CHECKPOINT_ACCEPTANCE.json').exists())
            time.sleep(10)
        resource=read(TRAIN/'lease_v1/LEASE_RESULT.json')
        assert resource['execute_returned'] and resource['holders_restored'] and resource['error'] is None,'TRAINING_OR_RESTORATION_FAILED'
        assert (HERE/'FIRST_CHECKPOINT_ACCEPTANCE.json').exists(),'FIRST_CHECKPOINT_NOT_ACCEPTED'
        final=read(TRAIN/'formal/attempt_001/RESULT.json')
        assert final['cursor']==read(TRAIN/'PROTOCOL_FILESTORE.json')['expected_final_cursor'] and final['status']=='EPOCHS_COMPLETED' and final['stop']==[],'PLANNED_FINAL_NOT_REACHED'
        status('VERIFYING_FINAL_CHECKPOINT')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        run_logged([str(QPY),'-I','-B',str(HERE/'accept_final.py')],'final_acceptance.log',300,env)
        status('BINDING_FINAL_CHECKPOINT')
        run_logged([str(PY),'-I','-S','-B',str(CASE/'prepare.py'),'bind'],'bind.log',300)
        status('EVALUATING_FIXED_CORRECTION_1000')
        with (HERE/'eval_launcher.log').open('x') as log:
            child=subprocess.Popen([str(PY),'-I','-S','-B',str(CASE/'launch.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            began=time.monotonic()
            while child.poll() is None:
                assert time.monotonic()-began<5520,'EVAL_SUPERVISION_WALL_LIMIT'
                status('EVALUATING_FIXED_CORRECTION_1000',own_launcher_pid=child.pid,elapsed_seconds=time.monotonic()-began)
                time.sleep(10)
            assert child.returncode==0,'EVALUATION_OR_TRACE_AUDIT_FAILED'
        status('ANALYZING_PAIRED_RESULTS');r=close_report()
        final=dict(status='COMPLETE',unix=time.time(),positive_development_signal=r['positive_development_signal'],
            paired_delta=r['paired']['delta'],scientific_gain_verified=False,automatic_next_training=False)
        save(HERE/'WORKFLOW_RESULT.json',final);status('COMPLETE',positive_development_signal=r['positive_development_signal'])
    except BaseException as exc:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=60)
        owner=seal['owned_training_lease'];signaled=False
        if (Path('/proc')/str(owner['pid'])).exists() and not (TRAIN/'lease_v1/LEASE_RESULT.json').exists():
            assert identity(owner['pid'])==owner,'UNKNOWN_LEASE_IDENTITY_NO_SIGNAL'
            os.kill(owner['pid'],signal.SIGTERM);signaled=True
        final=dict(status='FAILED_OR_BLOCKED',unix=time.time(),error=repr(exc),traceback=traceback.format_exc(),
            own_training_lease_signaled=signaled,foreign_processes_signaled=[],automatic_retry=False)
        save(HERE/'WORKFLOW_RESULT.json',final);status('FAILED_OR_BLOCKED',error=repr(exc));raise
    finally:lease.close()


if __name__=='__main__':
    assert sys.argv[1:] in (['freeze'],['run'])
    freeze() if sys.argv[1]=='freeze' else main()
