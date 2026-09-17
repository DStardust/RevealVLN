"""Close measured evidence without fabricating unexecuted metrics."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

OUT=Path(__file__).resolve().parent;LINE=OUT.parents[1];ROOT=LINE.parents[1]
def read(name):
    p=OUT/name
    if name in ('WORKER_RESULT.json','OFFLINE_AFTER.json','EXECUTION_RESULT.json','LEASE_RESTORED.json') and (OUT/'recovery_r1'/name).exists():p=OUT/'recovery_r1'/name
    return json.loads(p.read_text()) if p.exists() else None
def lines(name):
    result=[]
    for attempt,d in enumerate([OUT,OUT/'recovery_r1']):
        p=d/name
        if p.exists():result.extend(dict(json.loads(s),attempt=attempt) for s in p.read_text().splitlines())
    return result
def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    execution=read('EXECUTION_RESULT.json');assert execution is not None
    checks=[]
    for d in [LINE/'data_pipeline/ordinary_pilot_v1',LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1']:
        r=subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=d,capture_output=True,text=True)
        checks.append(dict(directory=str(d.relative_to(ROOT)),returncode=r.returncode,output=r.stdout+r.stderr))
    mismatches=[]
    for name,h in read('SOURCE_LOCK.json').items():
        # Shared mutable coordination documents belong to main agent; do not mistake its authorized changes for data corruption.
        if name.endswith('/AGENTS.md') or name=='AGENTS.md':continue
        if sha(ROOT/name)!=h:mismatches.append(name)
    save('FINAL_SOURCE_CHECKS.json',dict(sealed_checks=checks,source_lock_mismatches=mismatches))
    integrity=all(x['returncode']==0 for x in checks) and not mismatches
    worker=read('WORKER_RESULT.json');updates=lines('UPDATE_LEDGER.jsonl');samples=lines('RESOURCE_SAMPLES.jsonl')
    model_steps=lines('MODEL_STEP_LEDGER.jsonl')
    coverage={}
    lengths={r['policy_file']:r['decisions'] for r in read('SPLIT.json')['train']}
    for r in model_steps:
        if r['stage']!='train':continue
        key=(r['attempt'],r['epoch'],r['policy_file'])
        if key not in coverage:coverage[key]=dict(attempt=r['attempt'],epoch=r['epoch'],policy_file=r['policy_file'],decision_times=[],optimizer_updates_receiving_gradients=[])
        coverage[key]['decision_times'].append(r['t'])
        coverage[key]['optimizer_updates_receiving_gradients'].append(r['update_pending'])
    for r in coverage.values():
        r['complete_episode']=r['decision_times']==list(range(lengths[r['policy_file']]))
        r['optimizer_updates_receiving_gradients']=sorted(set(r['optimizer_updates_receiving_gradients']))
    save('TRAIN_EPISODE_COVERAGE.json',list(coverage.values()))
    save('STEP_ACCOUNTING.json',dict(model_steps_by_stage={s:sum(r['stage']==s for r in model_steps) for s in sorted({r['stage'] for r in model_steps})},preflight_forwards=2 if read('TRAINING_PREFLIGHT.json') else None,checkpoint_diagnostic_forwards_per_completed_reload=6,completed_reload_checks=len(lines('RELOAD_CHECKS.jsonl')),note='Checkpoint diagnostics: two causal steps before perturbation, after perturbation, and after disk reload; no optimizer updates. Partial failures retain raw events.'))
    before=read('OFFLINE_BEFORE.json');after=read('OFFLINE_AFTER.json');reloads=lines('RELOAD_CHECKS.jsonl')
    records=lines('OFFLINE_RECORDS.jsonl');pairs=[]
    a={r['policy_file']:r for r in records if r['stage']=='after'}
    for b in [r for r in records if r['stage']=='before']:
        z=a.get(b['policy_file']);pairs.append(dict(policy_file=b['policy_file'],job_id=b['job_id'],before=b,after=z,ce_change=z['ce_sum']/z['decisions']-b['ce_sum']/b['decisions'] if z else None,accuracy_change=z['correct']/z['decisions']-b['correct']/b['decisions'] if z else None))
    save('OFFLINE_PAIRED.json',pairs)
    episodes=lines('EPISODE_LEDGER.jsonl');nav_pairs=[]
    completed={(e['stage'],e['job_id']):e for e in episodes if e['event']=='complete'}
    for row in read('SPLIT.json')['closed_loop']:
        b=completed.get(('before',row['job_id']));a=completed.get(('after',row['job_id']))
        changes={}
        for k in ['stopped_within_3m','ever_within_3m','collisions','motion_actions','geodesic_ndtw','terminal_goal_distance_geodesic_m']:
            changes[k]=float(a[k])-float(b[k]) if a and b and a[k] is not None and b[k] is not None else None
        nav_pairs.append(dict(job_id=row['job_id'],scene_group=row['scene_group'],instruction_episode_id=row['instruction_episode_id'],before=b,after=a,changes=changes))
    save('NAVIGATION_PAIRED.json',nav_pairs)
    paired_complete=all(p['before'] and p['after'] for p in nav_pairs)
    nav_change={k:statistics.mean(p['changes'][k] for p in nav_pairs) if all(p['changes'][k] is not None for p in nav_pairs) else None for k in nav_pairs[0]['changes']} if paired_complete else None
    executions=[json.loads(p.read_text()) for p in [OUT/'EXECUTION_RESULT.json',OUT/'recovery_r1/EXECUTION_RESULT.json'] if p.exists()]
    elapsed=sum(e['gpu_stage_wall_seconds'] for e in executions)
    train_tokens=sum(max((u['cumulative_train_tokens'] for u in updates if u['attempt']==a),default=0) for a in (0,1))
    costs=dict(gpu_stage_wall_seconds=elapsed,gpu_hours=elapsed/3600,optimizer_updates=len(updates),gpu_peak_total_mib=max((s['memory_mib'] for s in samples),default=None),worker_tree_peak_rss_bytes=max((s['worker_tree_rss_bytes'] for s in samples),default=None),new_output_bytes_before_report=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),new_downloads=0,episode_starts=sum(e['event']=='start' for e in episodes),episode_completions=sum(e['event']=='complete' for e in episodes),training_tokens=train_tokens,all_forward_tokens=None,recovery_all_forward_tokens=worker['all_forward_tokens'] if worker else None,original_all_forward_tokens_upper_bound=2000000,note='First process failed before recording all-forward token total; its full GPU wall time and actual update/token ledger are retained, not counted as zero.')
    save('COST.json',costs)
    with (OUT/'TRAINING_CURVE.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cumulative_update','attempt','update','mean_action_ce','grad_norm_before_clip','decisions','lr','cumulative_train_tokens'])
        w.writeheader();w.writerows(dict({k:r[k] for k in w.fieldnames if k!='cumulative_update'},cumulative_update=i+1) for i,r in enumerate(updates))
    # Standalone SVG plot of actual update losses; no invented curve points.
    if updates:
        vals=[r['mean_action_ce'] for r in updates];lo=min(vals);hi=max(vals);span=max(hi-lo,1e-8)
        pts=' '.join(f'{50+700*i/max(1,len(vals)-1):.2f},{250-200*(v-lo)/span:.2f}' for i,v in enumerate(vals))
        with (OUT/'TRAINING_CURVE.svg').open('x') as f:f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300" viewBox="0 0 800 300"><rect width="800" height="300" fill="white"/><text x="50" y="22">Action CE per optimizer update (observed)</text><path d="M50 45V250H750" fill="none" stroke="black"/><polyline points="{pts}" fill="none" stroke="#1262aa" stroke-width="1"/><text x="4" y="55">{hi:.3f}</text><text x="4" y="250">{lo:.3f}</text><text x="50" y="278">1</text><text x="720" y="278">{len(vals)}</text></svg>')
    result=dict(experiment_id='Q35N_ORDINARY_ACTION_SFT_ACCEPTANCE_V1',decision='COMPLETED_BOUNDED_ACCEPTANCE' if worker and execution['returncode']==0 else 'STOPPED_ENGINEERING_FAILURE',
        data_integrity_pass=integrity,training_interface_pass=worker['training_interface_pass'] if worker else None,
        offline_learning_signal=worker['offline_learning_signal'] if worker else None,
        closed_loop_interface_pass=worker['closed_loop_interface_pass'] if worker else (False if lines('CLOSED_LOOP_FAILURES.jsonl') else None),paired_navigation_change=nav_change,
        scientific_pass=False,official_sr=None,official_spl=None,offline_before=before,offline_after=after,optimizer_updates=len(updates),terminal_checkpoint_effective_updates=worker['optimizer_updates'] if worker else None,cost=costs,
        gpu_restored=(read('LEASE_RESTORED.json') or {}).get('restored'),failure=read('LAUNCH_FAILURE.json'),
        scope='5 FIT_PILOT houses; 75 train routes / 24 seen-house route-dev; descriptive engineering results, no unseen-house generalization, SOTA or mechanism evidence',
        engineering_amendment='recovery_r1/AMENDMENT.json',execution_attempts=executions,checkpoint_reload_pass=bool(reloads) and all(r['pass_gate'] for r in reloads),spec_sha256=sha(OUT/'EXPERIMENT_SPEC.json'))
    save('result.json',result)
    def val(x):return 'null（未执行或未完成）' if x is None else str(x)
    text=f'''# 普通 action-only SFT 有界验收

结论：`{result['decision']}`。科学结论固定为 `scientific_pass=false`。

本次只使用 Qwen3.5-2B 固定 revision、8槽记忆、rank8 LoRA 与4动作 CE，无续接 reader、机制Y或状态辅助。75条训练路线/226条原指令，24条 seen-house route-dev/72条原指令；5个房屋都属于已暴露 FIT_PILOT。不可解释为未见房屋泛化。

## 实测结论

- 数据完整性：{val(result['data_integrity_pass'])}；训练技术接口：{val(result['training_interface_pass'])}。
- 离线 CE 下降信号：{val(result['offline_learning_signal'])}；动作准确率不是导航 SR。
- 闭环接口：{val(result['closed_loop_interface_pass'])}。官方 Habitat-Lab SR/SPL 未接入，均为 null。
- 保存重载检查：{val(result['checkpoint_reload_pass'])}，详见 RELOAD_CHECKS.jsonl；仅保存可训练参数及终点优化器状态，原模型保持只读。
- GPU4占位恢复：{val(result['gpu_restored'])}；停止真实任务数0，未操作GPU3及其pane。

## 协议与结果

EXPERIMENT_SPEC.json 在任何新模型结果之前冻结，SHA256 `{result['spec_sha256']}`。AdamW lr=1e-4、TBPTT=4、batch1、每8片段累积、恒定学习率、梯度裁剪1.0，固定400更新上限。不存在dev最佳checkpoint筛选或超参搜索。

工程修订见 recovery_r1/AMENDMENT.json：原运行更新1已发生，但随后的日志函数重复unix参数而退出，未保存该更新参数。因此从初始checkpoint重启399更新（初始logits再次核验），累计计算400更新、最终checkpoint有效399更新，不能称为原计划无中断的400步模型。没有改变算法或数据；原before离线与10个episode直接复用，不消耗额外闭环名额。原失败、账本、GPU恢复记录均保留。

更新前离线结果：`{json.dumps(before,ensure_ascii=False)}`

更新后离线结果：`{json.dumps(after,ensure_ascii=False)}`

10条固定闭环路线的配对平均变化（after-before）：`{json.dumps(nav_change,ensure_ascii=False)}`。停止半径采用3m，分别记录曾到达与到达后停止；nDTW采用仿真geodesic距离及原参考路点，未声称官方指标复现。每episode最多512运动动作，STOP单独记账，服务错误与超限保留。

## 成本、失败与限制

实际优化器更新 {len(updates)} 次；GPU阶段 {costs['gpu_hours']:.4f} 小时；峰值全卡显存 {val(costs['gpu_peak_total_mib'])} MiB；worker树峰值RSS {val(costs['worker_tree_peak_rss_bytes'])} 字节；闭环启动 {costs['episode_starts']} 个、完成 {costs['episode_completions']} 个；新增下载0。

原执行失败：`{json.dumps(result['failure'],ensure_ascii=False)}`。两次执行的完整退出与成本记录：`{json.dumps(executions,ensure_ascii=False)}`。CPU准备首次误用了系统Python3.6，因标准库参数不兼容退出，未写数据、未使用GPU；随后使用已授权项目Python3.10。没有隐藏旧1次更新成本。原进程未保存完整forward-token总计，精确累计值保持null；恢复过程总计与原阶段保守上界分开报告。

这是微型同房屋工程验收，有限loss下降只能支持学习接口；导航变化是固定小样本描述，不确立机制收益、创新性、SOTA或完整论文贡献。未新增未见房屋测试，不读取官方val/test，不使用主agent机制结果筛样本。

## 证据入口

冻结协议与来源：EXPERIMENT_SPEC.json、PREREGISTRATION_LOCK.json、SPLIT.json、SOURCE_LOCK.json、SEALED_SOURCE_CHECKS.json、FINAL_SOURCE_CHECKS.json。

代码来源和全部差异：CODE_PROVENANCE.json、CODE_DIFF.patch、CODE_LOCK.json。训练接口：TRAINING_PREFLIGHT.json、TRAINABLE_PARAMETERS.json、PARAMETER_CHANGES.json、RELOAD_CHECKS.jsonl、checkpoints/。

完整账本：根与recovery_r1各自的MODEL_STEP_LEDGER.jsonl、UPDATE_LEDGER.jsonl、TRAIN_ORDER.jsonl、TRAIN_EPISODE_LEDGER.jsonl、EPISODE_LEDGER.jsonl、SIM_STEP_LEDGER.jsonl；汇总TRAIN_EPISODE_COVERAGE.json、STEP_ACCOUNTING.json。配对结果：OFFLINE_PAIRED.json、NAVIGATION_PAIRED.json；曲线：TRAINING_CURVE.csv（有更新时另有SVG）。资源与恢复：COST.json、两个阶段的RESOURCE_SAMPLES.jsonl、EXECUTION_RESULT.json、LEASE_BEFORE.json、LEASE_ACTIVE.json、LEASE_RESTORED.json。终点checkpoint及更新后评估在recovery_r1中。

结果只写 sft_acceptance/v1，交主agent审核合并；未修改根或本线STATUS/README，不自行进入下一方法实验。未生成的文件或指标以实际目录及result.json为准，不将本证据清单当作已执行证明。
'''
    with (OUT/'REPORT_ZH.md').open('x') as f:f.write(text)
    # Runtime caches may remain writable for restored occupancy; exclude them from immutable evidence seal.
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS' and not any(x in ('cache','tmp','live') for x in p.relative_to(OUT).parts):
                f.write(sha(p)+'  '+str(p.relative_to(OUT))+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
