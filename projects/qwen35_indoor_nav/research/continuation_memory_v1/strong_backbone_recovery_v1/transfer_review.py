"""Recompute sealed groups and costs; can finish independently after the job."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
from transfer_pipeline import records, summarize


def percentile(values,q):
    if not values:return None
    values=sorted(values);position=(len(values)-1)*q;lower=int(position)
    return values[lower]+(values[min(lower+1,len(values)-1)]-values[lower])*(position-lower)


def main(run,wait):
    deadline=time.monotonic()+6*3600
    while wait and (not (run/'STATUS.json').exists() or u.read(run/'STATUS.json')['status']=='RUNNING'):
        if time.monotonic()>deadline:raise TimeoutError('REVIEW_WAIT_LIMIT')
        time.sleep(20)
    u.verify_sources(run);result=summarize(run);groups=records(run,'evaluation',True)
    costs={}
    for arm in ['NATIVE','BC','B2','OURS']:
        inference=[];memory=[];actions=collisions=far_stop=missed_stop=conflicts=0
        for item in groups.values():
            path=Path(item['path']).parent/arm/'TRACE.jsonl'
            assert u.sha(path)==item['trace_hashes'][arm],'TRACE_CHANGED'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            moves=[r for r in rows if r['event']=='action']
            generations=[r for r in rows if r['event']=='generation']
            inference.extend(r['seconds'] for r in generations)
            memory.extend(r['seconds'] for r in rows if r['event']=='memory_write')
            actions+=len(moves);collisions+=sum(r['collision'] for r in moves)
            outcome=item['outcomes'][arm]
            far_stop+=bool(moves and moves[-1]['executed_action']==0 and moves[-1]['distance']>=3)
            missed_stop+=bool(any(r['distance']<3 for r in moves) and not outcome['success'])
            conflicts+=sum(max(range(4),key=lambda i:r['native_logits'][i])==0 and max(range(4),key=lambda i:r['method_logits'][i])!=0 for r in generations)
        costs[arm]=dict(environment_decisions=actions,collisions=collisions,far_stop=far_stop,
            entered_range_without_success=missed_stop,native_stop_method_continue_at_queries=conflicts,
            base_generate_calls=len(inference),additional_visual_memory_calls=len(memory),
            generation_p50_seconds=percentile(inference,.5),generation_p95_seconds=percentile(inference,.95),
            dense_memory_p50_seconds=percentile(memory,.5),dense_memory_p95_seconds=percentile(memory,.95))
    status=u.read(run/'STATUS.json')
    result.update(status=status['status'],error=status.get('error'),gpu_hours=status.get('gpu_hours'),costs=costs,
        uncertainty='Descriptive paired results from ten correlated houses and one head-training seed; no broad independent significance claim',
        recovery_efficacy='NOT_MEASURED',training_fit_is_not_closed_loop_gain=True)
    u.write(run/'REVIEW.json',result)
    lines=[status['status'], '',
        f"完整四臂组 {len(groups)}/200；封存 episode {4*len(groups)}/800。完整 1839 条未运行。",
        "这是加入记忆后的普通导航能力检查，不是历史恢复任务收益实验。",
        "", "| 策略 | 成功/已封存 | SR（已封存分母） | 完整计划 |", "|---|---:|---:|---:|"]
    for arm,row in result['arms'].items():
        sr='待测' if row['sr_completed'] is None else f"{row['sr_completed']:.1%}"
        lines.append(f"| {arm} | {row['successes']}/{row['complete']} | {sr} | 200 |")
    lines+=['',f"GPU 会话小时：{status.get('gpu_hours',0):.3f}。错误：{status.get('error','无')}。",
        "旧原生 112/200（56%）保留为参考；因果比较使用本轮同进程 NATIVE。",
        "三臂头仅由两个 FIT 历史组、10 个动作监督点训练；此前 CHECK 没有准入样本，不能声称历史恢复泛化。",
        "胜负路线、按屋结果、缺项识别界与动作/时延成本见 REVIEW.json。小样本与房屋相关性限制结论，不自动采用任一方法。",
        "本轮结束后不自动训练、调阈值或更换测试样本。"]
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    main(u.HERE/'transfer_runs'/a.run_id,a.wait)

