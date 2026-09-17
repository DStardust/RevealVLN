"""Close only completed execution; never convert partial training to PASS."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent;LINE=OUT.parents[1]

def read(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else None

def rows(name):
    p=OUT/name
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

def save(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    assert not (OUT/'result.json').exists()
    execution=read('EXECUTION.json');assert execution is not None
    checks={p:hashlib.sha256((LINE/p).read_bytes()).hexdigest()==h for p,h in read('LOCK.json')['sources'].items()}
    assert all(checks.values());save('FINAL_SOURCE_CHECKS.json',checks)
    worker=[read(f'rank{r}_RESULT.json') for r in range(2)]
    complete=execution['error'] is None and all(x is not None for x in worker)
    updates=[[x for x in rows(f'rank{r}_events.jsonl') if x['event']=='update'] for r in range(2)]
    u=updates[0];steps=[x for r in range(2) for x in rows(f'rank{r}_train_steps.jsonl')]
    profile=[read(f'rank{r}_CACHE_PROFILE.json') for r in range(2)]
    bucket=[read(f'rank{r}_BUCKET_CORRECTNESS.json') for r in range(2)]
    learned=all(x['learnability_pass'] for x in worker) if complete else None
    decision=('PASS_TEN_TRAIN_ROUTE_LEARNABILITY' if learned else 'FAIL_BOUNDED_TEN_TRAIN_ROUTE_LEARNABILITY') if complete else 'BLOCKED_ENGINEERING_NOT_LEARNABILITY_VERDICT'
    times=[b['unix']-a['unix'] for a,b in zip(u,u[1:])]
    median_update=statistics.median(times) if times else None
    steady_decisions_per_sec=sum(x['decisions'] for x in u[1:])/sum(times) if times else None
    train_targets=collections.Counter(str(x['target']) for x in steps)
    train_preds=collections.Counter(str(x['prediction']) for x in steps)
    metrics=dict(median_update_seconds=median_update,steady_effective_decisions_per_second=steady_decisions_per_sec,
                 training_wall_seconds=worker[0]['train_wall_seconds'] if complete else None,
                 warm_cache_speedups=[p['hot_speedup'] if p else None for p in profile],
                 four_GPU_speedup=None,full_epoch_wall_seconds=None)
    result=dict(decision=decision,execution_complete=complete,scientific_pass=False,learnability_pass=learned,
        bucket_correctness_pass=all(b is not None and b['pass_gate'] for b in bucket),
        old_D1_update_equivalence_failure_preserved=True,source_integrity_pass=True,
        shared_training_updates=min(map(len,updates)),local_training_updates=[len(a) for a in updates],
        shared_diagnostic_updates=1 if all(b is not None for b in bucket) else None,
        training_decisions=len(steps),train_target_counts=train_targets,train_prediction_counts=train_preds,
        cache_selection=read('rank0_CACHE_SELECTION.json'),cache_profiles=profile,efficiency=metrics,
        before=read('rank0_EVAL_before.json'),after=read('rank0_EVAL_after.json'),workers=worker,
        execution=execution,restoration=read('RESTORATION.json'),
        failures=[read(f'rank{r}_FAILURE.json') for r in range(2)],
        navigation_episodes=0,foundation_navigation_complete=False,new_downloads=0)
    save('result.json',result)
    report=f'''# 训练效率与小集验收实测

结论：`{decision}`。真实bucket归约：{result['bucket_correctness_pass']}；小集可学性：{learned}。未执行导航闭环，基础导航基座尚未完成。

## 实際执行与效率

固定10训练路线565决策；每rank更新{result['local_training_updates']}，共享训练更新{result['shared_training_updates']}，训练决策{len(steps)}。

缓存选择：`{json.dumps(result['cache_selection'],ensure_ascii=False)}`。

效率实测：`{json.dumps(metrics,ensure_ascii=False)}`。缓存短测为同卡同片段forward+backward，冷启动和构建单列于CACHE_PROFILE；未测四卡，不宣称线性加速。与历史全池单卡耗时并非匹配数据的严格加速比。

## 学习结果

before：`{json.dumps(result['before'],ensure_ascii=False)}`

after：`{json.dumps(result['after'],ensure_ascii=False)}`

所有阈值、最多200更新和终点评价事前冻结，无dev选优。训练集动作拟合即使通过也不是导航成功/泛化；未通过只否证当前有界配置，不宣称整个方向不可行。

## 工程归约与边界

同次反向的原始bucket与实际SUM/world_size结果逐bucket比较，记录BUCKET_CORRECTNESS。此为分布式数学实现验收，与旧D1独立单卡/双卡AdamW更新向量比较不同；旧0.15关卡FAIL不改写。缓存不缓存可训练语言隐藏状态或记忆，无未来标签进入策略。

## 资源和完整性

执行：`{json.dumps(execution,ensure_ascii=False)}`。

受保护来源{len(checks)}项哈希通过。GPU5/6占位恢复：{execution['all_leased_holders_restored']}；停止真实任务0。无下载/仿真/机制训练。checkpoint只在完整200更新后生成并实测重载；未生成时不冒充存在。
'''
    with (OUT/'REPORT_ZH.md').open('x') as f:f.write(report)
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.name!='SHA256SUMS':f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
    print(json.dumps(dict(decision=decision,updates=result['shared_training_updates'],efficiency=metrics,restored=execution['all_leased_holders_restored']),ensure_ascii=False))

if __name__=='__main__':main()
