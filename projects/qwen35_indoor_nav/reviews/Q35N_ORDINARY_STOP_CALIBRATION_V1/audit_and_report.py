"""Verify the sole frozen bias, live trajectories, and the registered DEV gate."""
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
CASE=LINE/'closed_loop_bench/ordinary_stop_calibrated_dev_v1'
BASE=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1'
FIT=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
ANALYZER=LINE/'reviews/Q35N_ORDINARY_TARGETED_REPAIR_V1/analyze.py'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def traces(case):
    episodes={};policy=collections.defaultdict(list)
    for lane in sorted((case/'run_001/lanes').glob('lane_*')):
        for path in lane.glob('episode_*.json'):
            e=read(path);assert e['index'] not in episodes;episodes[e['index']]=e
        for line in (lane/'POLICY_STEPS.jsonl').read_text().splitlines():
            row=json.loads(line);policy[row['index']].append(row)
    assert sorted(episodes)==list(range(100))
    return episodes,policy


def main():
    assert not (HERE/'FINAL_DECISION.json').exists(),'ALREADY_CLOSED'
    p=read(CASE/'PROTOCOL.json');cal=read(HERE/'CALIBRATION.json');pred=read(HERE/'DEV_PREDICTION.json')
    locks=read(CASE/'SOURCE_LOCK.json')['files']
    for path in (HERE/'audit_and_report.py',HERE/'counterfactual.py',HERE/'CALIBRATION.json'):
        assert sha(path)==locks[str(path)],path
    assert cal['status']=='FROZEN_FIT_PARAMETER' and p['stop_logit_bias']==cal['bias']==pred['bias']
    assert sha(HERE/'CALIBRATION.json')==p['calibration_sha256']==pred['calibration_sha256']
    assert pred['additional_biases_evaluated_on_dev']==0
    assert sha(HERE/'DEV_PREDICTION.json')==read(CASE/'MAIN_REVIEW.json')['prediction_sha256']
    for path,digest in pred['trace_hashes'].items():assert sha(Path(path))==digest,path
    result=read(CASE/'run_001/RESULT.json');launch=read(CASE/'run_001/LAUNCH_RESULT.json')
    assert result['status']=='COMPLETE' and result['completed']==result['planned']==100 and result['trace_audit_passed']
    assert result['checkpoint_sha256']==cal['checkpoint_sha256'] and result['selected_batch_size']==1
    assert launch['status']=='COMPLETE' and not launch['cleanup']['remaining'] and not launch['foreign_processes_signaled']
    c=load('stop_rule_audit',HERE/'counterfactual.py')
    base,bp=traces(BASE);actual,ap=traces(CASE);mismatches=[];audited=interventions=0
    for i in range(100):
        e=actual[i];b=base[i];expected=pred['episodes'][i]
        assert (e['episode_id'],e['house'])==(b['episode_id'],b['house'])
        for row in ap[i]:
            raw=row['raw_logits'];adjusted=list(raw);adjusted[3]+=cal['bias']
            assert row['stop_logit_bias']==cal['bias'] and row['logits']==adjusted
            assert row['action']==c.choose(raw,cal['bias'])
            if c.choose(raw,0)!=row['action']:
                assert row['action']=='STOP';interventions+=1
            audited+=1
        differences=[]
        for key in ('success','spl','ndtw','steps','navigation_error_m','path_length_m'):
            if not math.isclose(e[key],expected[key],rel_tol=1e-9,abs_tol=1e-9):differences.append(key)
        for j,row in enumerate(ap[i]):
            if j>=len(bp[i]):differences.append('extends_original_trajectory');break
            if row['action']!='STOP' and row['action']!=bp[i][j]['action']:differences.append('movement_prefix')
        if e['stopped'] and e['steps']<=b['steps']:
            stop_at=e['steps']-1;positions=b['positions'][:stop_at+1]+[b['positions'][stop_at]]
        else:positions=b['positions']
        if e['positions']!=positions:differences.append('positions')
        if differences:mismatches.append(dict(index=i,fields=sorted(set(differences))))
    assert audited==result['environment_actions']==result['audited_actions']
    assert sha(ANALYZER)=='827e9a1050715d9ca12469eb26379210fa2b23f959f1b73280f258145d46220e'
    a=load('frozen_paired_analyzer',ANALYZER);before=a.case_report(BASE);after=a.case_report(CASE);paired=a.paired(before,after)
    delta=paired['delta'];gate=delta['success']>0 and delta['spl']>=0 and delta['ndtw']>=-.01
    positive=gate and not mismatches
    fit=read(FIT/'run_001/RESULT.json');failed=read(FIT/'PREDECESSOR_FAILURE.json')
    decision=dict(status='COMPLETE' if not mismatches else 'PREFIX_MISMATCH_NO_ADOPTION',unix=time.time(),
         positive_development_signal=positive,registered_navigation_gate_passed=gate,
         exact_prefix_prediction_verified=not mismatches,prefix_mismatches=mismatches,
         action_rule_audited_steps=audited,actual_stop_interventions=interventions,
         bias=cal['bias'],checkpoint_sha256=cal['checkpoint_sha256'],calibration_sha256=sha(HERE/'CALIBRATION.json'),
         before=before,after=after,paired=paired,recommendation='ADOPT_AS_DEVELOPMENT_CANDIDATE' if positive else 'RETAIN_UNCALIBRATED_4000',
         compute=dict(fit_environment_actions=fit['environment_actions'],dev_environment_actions=audited,
             fit_wall_seconds=fit['wall_seconds'],dev_wall_seconds=result['wall_seconds'],
             failed_predecessor_wall_seconds=failed['wall_seconds'],gradient_updates=0,scalar_parameters_fitted=1),
         scientific_gain_verified=False,independent_confirmation_done=False,visual_stall_guard_enabled=False,
         special_training_data_used=False,automatic_training=False,all_historical_failures_preserved=True)
    save(HERE/'FINAL_DECISION.json',decision)
    text='# 普通导航停止校准：真实开发核验\n\n'
    text+='固定4,000步基座，只在16个FIT房屋64条路线选择一个全局STOP偏置，再在原100条/5屋INTERNAL_DEV进行真实闭环核验；没有使用DEV选择参数。普通工程校准，不是UAD创新。\n\n'
    text+='| 同100条、batch=1 | SR | SPL | nDTW | OSR |\n|---|---:|---:|---:|---:|\n'
    for label,case in [('原4,000步模型',before),('FIT停止校准后的同一模型',after)]:
        r=case['result'];text+=f"| {label} | {100*r['sr']:.2f}% | {100*r['spl']:.2f}% | {100*r['ndtw']:.2f}% | {100*r['osr']:.2f}% |\n"
    text+=f"\n固定偏置b={cal['bias']}，新增成功{int(paired['wins'])}条、失去{int(paired['losses'])}条；实际触发提前停止{interventions}次。完整动作规则审核{audited}步。因果前缀预测与真实模拟{'一致' if not mismatches else '存在差异，不采用'}。事前开发门槛：{'通过' if positive else '未通过'}。\n\n"
    text+='| 房屋 | ΔSR（百分点） | ΔSPL | ΔnDTW |\n|---|---:|---:|---:|\n'
    for h,d in paired['by_house_delta'].items():text+=f"| {h} | {100*d['success']:+.2f} | {100*d['spl']:+.2f} | {100*d['ndtw']:+.2f} |\n"
    text+='\n五屋bootstrap和留一屋见FINAL_DECISION.json。它们是已暴露开发集上的描述性结果，不是独立泛化证明，也不是完整官方val_unseen成绩。高LR8,000步SR14%、低LR8,000步SR13%和无收益的视觉停滞保护均保留，未启用。\n\n'
    text+='仅提前STOP的因果前缀计算用于FIT拟合和唯一参数的事前预测；以上表格是重新在模拟器实际运行的结果，不把离线预测当真实回放。部署动作只读取原模型四个分数和固定b，目标距离/位置仅用于离线评分。特殊数据未混入，模型架构没有新增记忆模块。\n'
    with (HERE/'REPORT_ZH.md').open('x') as f:f.write(text)
    print(json.dumps({k:decision[k] for k in ('status','positive_development_signal','bias','actual_stop_interventions','recommendation')},ensure_ascii=False))


if __name__=='__main__':main()
