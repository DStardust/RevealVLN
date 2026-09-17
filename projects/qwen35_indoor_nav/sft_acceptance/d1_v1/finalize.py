"""Read completed logs, preserve partial failures, and seal this diagnostic only."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]

def read(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else None

def rows(name):
    p=OUT/name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def main():
    assert not (OUT/'result.json').exists()
    execution=read('EXECUTION.json');assert execution is not None
    lock=read('LOCK.json');sources={}
    for rel,h in lock['sources'].items():sources[rel]=hashlib.sha256((LINE/rel).read_bytes()).hexdigest()==h
    assert all(sources.values())
    save('FINAL_SOURCE_CHECKS.json',sources)
    subset=read('SUBSET.json');counts=collections.Counter()
    for r in subset['rows']:
        counts.update(json.loads((LINE/'data_pipeline/ordinary_pilot_v1'/r['supervision_file']).read_text())['actions'])
    total=sum(counts.values());entropy=-sum(n/total*math.log(n/total) for n in counts.values())
    workers=[read(f'rank{r}_RESULT.json') for r in range(2)]
    eq=[read(f'rank{r}_REAL_EQUIVALENCE.json') for r in range(2)]
    toy=[read(f'rank{r}_NCCL_TOY.json') for r in range(2)]
    events=[rows(f'rank{r}_events.jsonl') for r in range(2)]
    updates=[[e for e in ev if e['event']=='update'] for ev in events]
    steps=[x for r in range(2) for x in rows(f'rank{r}_train_steps.jsonl')]
    exposures=collections.Counter(str(x['target']) for x in steps)
    diagnostics={}
    for rank in range(2):
        d=read(f'rank{rank}_D0.json')
        if not d:continue
        changes={mode:max(abs(a-b) for n,m in zip(d['normal'],d[mode]) for a,b in zip(n['logits'][0],m['logits'][0]))
                 for mode in ['zero_memory','mismatched_instruction','mismatched_images']}
        diagnostics[str(rank)]=dict(normal_memory_RMS=[x['memory_rms'] for x in d['normal']],
                                   embedding_weight_rms=d['embedding_weight_rms'],logit_sensitivity_max_abs=changes,
                                   largest_gradients=sorted(d['module_gradients'].items(),key=lambda x:x[1]['grad_norm'] or 0,reverse=True)[:6])
    save('D0_SUMMARY.json',diagnostics)
    ddp=all(x is not None and x['pass_gate'] for x in eq) and all(x is not None and x['pass_gate'] for x in toy)
    complete=all(x is not None for x in workers) and execution['error'] is None
    learned=all(x['learnability_pass'] for x in workers) if complete else None
    decision=('PASS_TEN_ROUTE_LEARNABILITY_NOT_GENERALIZATION' if learned else 'FAIL_BOUNDED_TEN_ROUTE_LEARNABILITY') if complete else 'BLOCKED_ENGINEERING_GATE_NOT_SCIENTIFIC_FAILURE'
    resource=rows('RESOURCES.jsonl')
    result=dict(decision=decision,DDP_interface_pass=ddp,learnability_pass=learned,scientific_pass=False,
                source_integrity_pass=True,completed_training_updates_per_rank=[len(u) for u in updates],
                unique_shared_DDP_training_updates=min(map(len,updates)),
                diagnostic_update_records=eq,training_decisions=len(steps),training_action_exposures=exposures,
                subset_routes=10,subset_decisions=total,subset_action_counts=counts,subset_constant_frequency_CE=entropy,
                subset_always_forward_accuracy=counts['move_forward']/total,
                before=read('rank0_EVAL_before.json'),after=read('rank0_EVAL_after.json'),D0_summary=diagnostics,
                execution=execution,restoration=read('RESTORATION.json'),
                peak_total_GPU_MiB=max((s['memory_mib'] for row in resource for s in row['gpus']),default=None),
                peak_worker_RSS_bytes=max((row['rss'] for row in resource),default=None),
                failures=[read(f'rank{r}_FAILURE.json') for r in range(2)],
                navigation_episodes=0,downloads=0,new_mechanism_training=False,
                next_action='Review numerical/input/optimization evidence; no automatic D2' if not learned else 'Freeze matched full-pool D2 repair protocol separately')
    save('result.json',result)
    text=f'''# D0 + D1 双卡小集验收

结论：`{decision}`。DDP接口通过：{ddp}；10训练路线可学性通过：{learned}；scientific_pass=false。

固定10训练路线、{total}决策，只用既有训练split；不读dev挑配置、不运行闭环。旧SFT来源{len(sources)}项哈希通过，原失败不覆盖。

## 实际执行

每rank完成训练更新：{[len(u) for u in updates]}；实际训练决策：{len(steps)}；动作暴露：{dict(exposures)}。单卡参考与DDP诊断更新另外见REAL_EQUIVALENCE，丢弃后才开始D1。工程失败时未执行指标保持null，不记为科学FAIL。

双卡执行记录：`{json.dumps(execution,ensure_ascii=False)}`。所有借用占位恢复：{execution['all_leased_holders_restored']}；真实任务停止0。

## 小集评价

频率常数CE={entropy:.6f}；始终前进accuracy={counts['move_forward']/total:.6f}。

before：`{json.dumps(result['before'],ensure_ascii=False)}`

after：`{json.dumps(result['after'],ensure_ascii=False)}`

阈值在运行前冻结：accuracy>=95%、macro recall>=90%、STOP precision及recall>=90%。200步内未通过不证明无限预算下不可学；通过也只证明小训练集拟合，不能当泛化/导航收益。

## 诊断与边界

D0_SUMMARY.json保留旧terminal的正常记忆尺度、指令/图像/零记忆敏感性和梯度。置换产生分布外输入，不足以证明正确语义利用；梯度大也不独立证明实现错误。

本节点不改原架构、学习率或类别权重。DDP、类别均衡及后续稳定化不当作原创贡献。不进入全量D2或机制训练；下一步：{result['next_action']}。

报告与result记录实际测量，rank日志保留全部失败。缓存/tmp不封存（占位可继续使用其缓存），根目录证据和checkpoint单独SHA256SUMS封存。
'''
    with (OUT/'REPORT_ZH.md').open('x') as f:f.write(text)
    paths=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in paths:f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
    print(json.dumps(dict(decision=decision,DDP_pass=ddp,learnability_pass=learned,restored=execution['all_leased_holders_restored']),ensure_ascii=False))

if __name__=='__main__':main()
