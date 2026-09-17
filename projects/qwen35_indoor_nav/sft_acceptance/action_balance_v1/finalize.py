"""Final matched report; complete, failed and interrupted executions remain distinct."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
CONTROL=OUT.parent/'efficiency_run_v1'
def read(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else None
def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    execution=read('EXECUTION.json');assert execution and not (OUT/'result.json').exists()
    lock=read('LOCK.json')
    checks={p:hashlib.sha256((LINE/p).read_bytes()).hexdigest()==h for p,h in lock['sources'].items()}
    checks.update({'code:'+p:hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h for p,h in lock['code'].items()})
    checks['weights']=hashlib.sha256((OUT/'WEIGHTS.json').read_bytes()).hexdigest()==lock['weights_sha256']
    checks['subset']=hashlib.sha256((OUT/'SUBSET.json').read_bytes()).hexdigest()==lock['subset_sha256']
    assert all(checks.values());save('FINAL_SOURCE_CHECKS.json',checks)
    workers=[read(f'rank{r}_RESULT.json') for r in range(2)]
    updates=[[x for x in rows(OUT/f'rank{r}_events.jsonl') if x['event']=='update'] for r in range(2)]
    complete=execution['error'] is None and all(workers) and all(len(u)==200 for u in updates) and execution['returncodes']==[0,0]
    if complete:
        assert workers[0]['after']==workers[1]['after']
        assert all(x['reload_max_abs']<=1e-5 for x in workers)
    after=read('rank0_EVAL_after.json')
    learned=after['learnability_pass'] if complete else None
    decision=('PASS_TEN_TRAIN_ROUTE_LEARNABILITY' if learned else 'FAIL_BOUNDED_TEN_TRAIN_ROUTE_LEARNABILITY') if complete else 'BLOCKED_ENGINEERING_NOT_LEARNABILITY_VERDICT'
    control=json.loads((CONTROL/'rank0_EVAL_after.json').read_text())
    comparison=None
    if complete:
        comparison=dict(control=control,treatment=after,
            differences={k:after[k]-control[k] for k in ['CE','accuracy','macro_recall','STOP_precision','STOP_recall']},
            recall_differences=[a-b for a,b in zip(after['recall'],control['recall'])],
            claim='descriptive single-seed matched exposed training-set comparison; not novelty or generalization')
        def keyed(directory):
            rr=[x for r in range(2) for x in rows(directory/f'rank{r}_eval_after.jsonl')]
            d={(x['row'],x['t']):x for x in rr}
            assert len(d)==len(rr)==565
            return d
        a,b=keyed(CONTROL),keyed(OUT)
        assert a.keys()==b.keys()
        for k in a:assert a[k]['target']==b[k]['target'] and a[k]['job_id']==b[k]['job_id']
        comparison['paired_newly_correct']=sum(a[k]['prediction']!=a[k]['target'] and b[k]['prediction']==b[k]['target'] for k in a)
        comparison['paired_newly_wrong']=sum(a[k]['prediction']==a[k]['target'] and b[k]['prediction']!=b[k]['target'] for k in a)
        comparison['per_route']=[]
        for row in range(10):
            keys=[k for k in a if k[0]==row]
            comparison['per_route'].append(dict(row=row,decisions=len(keys),
                control_correct=sum(a[k]['prediction']==a[k]['target'] for k in keys),
                treatment_correct=sum(b[k]['prediction']==b[k]['target'] for k in keys)))
        save('MATCHED_COMPARISON.json',comparison)
    u=updates[0];times=[b['unix']-a['unix'] for a,b in zip(u,u[1:])]
    steps=[x for r in range(2) for x in rows(OUT/f'rank{r}_train_steps.jsonl')]
    result=dict(decision=decision,execution_complete=complete,learnability_pass=learned,scientific_pass=False,
        shared_training_updates=min(map(len,updates)),local_training_updates=list(map(len,updates)),
        training_decisions=len(steps),train_target_counts=collections.Counter(str(x['target']) for x in steps),
        weights=read('WEIGHTS.json'),before=read('rank0_EVAL_before.json'),after=after,comparison=comparison,
        efficiency=dict(median_update_seconds=statistics.median(times) if times else None,
            steady_effective_decisions_per_second=sum(x['decisions'] for x in u[1:])/sum(times) if times else None,
            training_wall_seconds=workers[0]['train_wall_seconds'] if complete else None),
        workers=workers,execution=execution,restoration=read('RESTORATION.json'),source_integrity_pass=True,
        navigation_episodes=0,foundation_navigation_complete=False,mechanism_training=False,new_downloads=0)
    save('result.json',result)
    report=f"""# 同数据动作加权 SFT 实测

结论：{decision}。完整训练：{complete}。可学性：{learned}。

固定10训练路线565决策、同initial、同两rank顺序、最多200更新，唯一训练目标变化为温和类别加权。
这是已知标准工程修复，不是原创算法；无机制监督、无dev选优、无新导航闭环。

权重：{json.dumps(result['weights'],ensure_ascii=False)}

## 实测对照

{json.dumps(comparison,ensure_ascii=False,indent=2)}

正向局部变化与固定通过门槛分别报告。仅同一暴露训练集、单seed的描述性结果，不能当泛化或导航收益。

## 效率与交付

训练更新：{result['shared_training_updates']}，训练决策：{len(steps)}。
效率：{json.dumps(result['efficiency'],ensure_ascii=False)}
执行与恢复：{json.dumps(execution,ensure_ascii=False)}
来源/代码/权重哈希：{len(checks)}项通过。失败旧实验未修改。
导航基座尚不能宣布完成；不自动扩大训练或增加机制损失。
"""
    with (OUT/'REPORT_ZH.md').open('x') as f:f.write(report)
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.name!='SHA256SUMS':f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
    print(json.dumps(dict(decision=decision,after=after,efficiency=result['efficiency'],restored=execution['all_leased_holders_restored']),ensure_ascii=False))
if __name__=='__main__':main()
