"""Bounded FIT compatibility -> one predeclared DEV only if FIT passes."""
import fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
MERGE=LINE/'sft_acceptance/ordinary_equal_merge_v7'
FIT=LINE/'closed_loop_bench/ordinary_equal_merge_fit_v7'
DEV=LINE/'closed_loop_bench/ordinary_equal_merge_dev_v7'
FITBASE=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
BASE=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1'
ANALYZER=LINE/'reviews/Q35N_ORDINARY_TARGETED_REPAIR_V1/analyze.py'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def status(name,**extra):
    p=HERE/'WORKFLOW_STATUS.tmp';p.write_text(json.dumps(dict(status=name,unix=time.time(),training_allowed=False,automatic_retry=False,**extra),ensure_ascii=False));p.replace(HERE/'WORKFLOW_STATUS.json')
def freeze():
    assert not (HERE/'SEAL.json').exists()
    assert sha(ANALYZER)=='827e9a1050715d9ca12469eb26379210fa2b23f959f1b73280f258145d46220e'
    files=dict(read(FIT/'SOURCE_LOCK.json')['files'])
    files.update(read(DEV/'PENDING_SEAL.json')['files'])
    for f in list(HERE.glob('*.py'))+[FIT/'SOURCE_LOCK.json',FIT/'MAIN_REVIEW.json',DEV/'PENDING_SEAL.json',ANALYZER]:
        files[str(f)]=sha(f)
    save(HERE/'SEAL.json',dict(files=files,alpha_candidates=[.5],training_allowed=False,one_fit=True,
      dev_only_if_fit_gate=True,fit_max_success_losses=2,main_gate=dict(sr_delta_gt=0,spl_delta_ge=0,ndtw_delta_ge=-.01)))
    print('FROZEN_ONE_FIT_THEN_GATED_DEV')
def fit_gate():
    def load(case):
        r=read(case/'run_001/RESULT.json')
        assert r['status']=='COMPLETE' and r['completed']==r['planned']==64 and r['trace_audit_passed'] and r['source_lock_verified']
        eps=[read(p) for p in (case/'run_001/lanes').glob('lane_*/episode_*.json')]
        assert len(eps)==64 and {e['index'] for e in eps}==set(range(64))
        return r,sorted(eps,key=lambda e:e['index'])
    a,ae=load(FITBASE);b,be=load(FIT)
    assert a['selected_batch_size']==b['selected_batch_size']==1
    pairs=list(zip(ae,be));assert all(x['index']==y['index'] and x['episode_id']==y['episode_id'] and x['house']==y['house'] for x,y in pairs)
    wins=sum(not x['success'] and y['success'] for x,y in pairs)
    losses=sum(x['success'] and not y['success'] for x,y in pairs)
    assert sum(e['success'] for e in ae)==16
    delta={k:b[k]-a[k] for k in ('sr','spl','ndtw')}
    passed=delta['sr']>0 and delta['spl']>=0 and delta['ndtw']>=-.01 and losses<=2
    gate=dict(pass_gate=passed,unix=time.time(),before=a,after=b,delta=delta,wins=int(wins),losses=int(losses),
      retained_old_successes=16-int(losses),independent_validation=False,training_side_diagnostic=True,
      positive_navigation_result=False,dev_authorized=passed,alpha=.5,other_alphas_allowed=False)
    save(HERE/'FIT_GATE.json',gate)
    return gate
def close_report():
    s=importlib.util.spec_from_file_location('same_dev_analyzer',ANALYZER);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    before=a.case_report(BASE);after=a.case_report(DEV);paired=a.paired(before,after)
    assert paired['matched_inference_batch']
    d=paired['delta'];positive=d['success']>0 and d['spl']>=0 and d['ndtw']>=-.01
    result=dict(status='COMPLETE',unix=time.time(),before=before,after=after,paired=paired,
      positive_development_signal=positive,scientific_gain_verified=False,alpha=.5,fit_gate=read(HERE/'FIT_GATE.json'),
      parameter_updates=0,no_inference_compute_increase=True,no_controller=True,automatic_retry=False,
      ordinary_engineering_not_UAD_novelty=True,checkpoint=str(MERGE/'merged_equal_inference_only.pt'))
    save(HERE/'RESULT.json',result)
    report='# 普通导航：单一等权参数合并结果\n\n先通过训练侧64条兼容性门槛，再测固定100条/5屋INTERNAL_DEV。已暴露开发集，不是独立论文确认。\n\n'
    report+='| 模型 | SR | SPL | nDTW | OSR |\n|---|---:|---:|---:|---:|\n'
    for label,c in [('最佳4k',before),('固定0.5参数合并',after)]:
        r=c['result'];report+=f"| {label} | {100*r['sr']:.2f}% | {100*r['spl']:.2f}% | {100*r['ndtw']:.2f}% | {100*r['osr']:.2f}% |\n"
    report+=f"\n新增成功{int(paired['wins'])}条，失去{int(paired['losses'])}条。事前工程门槛：{'满足' if positive else '不满足'}。\n\n"
    report+='| 房屋 | ΔSR（百分点） | ΔSPL | ΔnDTW |\n|---|---:|---:|---:|\n'
    for h,v in paired['by_house_delta'].items():report+=f"| {h} | {100*v['success']:+.2f} | {100*v['spl']:+.2f} | {100*v['ndtw']:+.2f} |\n"
    report+='\n仅同名可训练参数的等权平均，没有重新训练；不是logit集成，也不是LoRA有效矩阵平均。结构、输入、参数量、每步一次forward均不变。两个来源权重及V6的19%失败保留。\n'
    report+='\n这借鉴已有参数插值工程，不是UAD模型创新。FIT64条参与过纠错数据生成，只是训练侧诊断；本次只试固定0.5，无扫参。逐屋/留一屋/五屋bootstrap在RESULT.json，仅描述内部开发现象，不能当独立泛化或显著性证明。\n'
    with (HERE/'REPORT_ZH.md').open('x') as f:f.write(report)
    return result
def main():
    lock=(HERE/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'WORKFLOW_RESULT.json').exists()
    for p,d in read(HERE/'SEAL.json')['files'].items():assert sha(p)==d,p
    def interrupted(sig,frame):raise RuntimeError('WORKFLOW_SIGNAL_'+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    child=None
    def run_case(case,tag):
        nonlocal child
        with (HERE/(tag+'_launcher.log')).open('x') as log:
            child=subprocess.Popen([str(PY),'-I','-S','-B',str(case/'launch.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            began=time.monotonic()
            while child.poll() is None:
                assert time.monotonic()-began<5520,'STAGE_WALL_LIMIT'
                status('EVALUATING_'+tag.upper(),own_launcher_pid=child.pid,elapsed_seconds=time.monotonic()-began);time.sleep(10)
            assert child.returncode==0,'EVAL_OR_AUDIT_FAILED_'+tag
    try:
        run_case(FIT,'fit');status('CHECKING_FIT_GATE');gate=fit_gate()
        if not gate['pass_gate']:
            value=dict(status='COMPLETE_FIT_GATE_FAILED',unix=time.time(),positive_development_signal=False,
              fit_gate=gate,dev_started=False,other_alpha_allowed=False,training_updates=0)
            save(HERE/'RESULT.json',value);save(HERE/'WORKFLOW_RESULT.json',value);status('COMPLETE_FIT_GATE_FAILED')
            with (HERE/'REPORT_ZH.md').open('x') as f:
                f.write('# 固定等权合并：训练侧门槛未通过\n\n未启动DEV，也没有把FIT成绩计作正向结果。原最佳4k保留，关闭本候选，不自动换混合比例。\n\n'+json.dumps(gate,ensure_ascii=False,indent=2))
            return
        status('ACTIVATING_PREDECLARED_DEV')
        with (HERE/'dev_activation.log').open('x') as log:
            result=subprocess.run([str(PY),'-I','-S','-B',str(MERGE/'prepare.py'),'activate_dev'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode==0,'DEV_ADMISSION_FAILED'
        run_case(DEV,'dev');status('ANALYZING_PAIRED_RESULTS');r=close_report()
        value=dict(status='COMPLETE',unix=time.time(),positive_development_signal=r['positive_development_signal'],
          paired_delta=r['paired']['delta'],scientific_gain_verified=False,training_updates=0,automatic_retry=False)
        save(HERE/'WORKFLOW_RESULT.json',value);status('COMPLETE',positive_development_signal=r['positive_development_signal'])
    except BaseException as exc:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=60)
        value=dict(status='FAILED_OR_BLOCKED',unix=time.time(),error=repr(exc),traceback=traceback.format_exc(),
          training_processes_signaled=[],foreign_processes_signaled=[],automatic_retry=False)
        save(HERE/'WORKFLOW_RESULT.json',value);status('FAILED_OR_BLOCKED',error=repr(exc));raise
    finally:lock.close()
if __name__=='__main__':
    assert sys.argv[1:] in (['freeze'],['run'])
    freeze() if sys.argv[1]=='freeze' else main()
