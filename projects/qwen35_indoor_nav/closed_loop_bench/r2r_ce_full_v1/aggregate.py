"""Final full-denominator report, independent trace audit and house uncertainty."""
import collections
import csv
import gzip
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

HERE=Path(__file__).resolve().parent;RUN=HERE/'run_001'
s=importlib.util.spec_from_file_location('common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def read(path):return json.loads(path.read_text())
def records(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def exact_ndtw_check(positions,reference):
    unique=[x for i,x in enumerate(positions) if i==0 or x!=positions[i-1]]
    cost=[[math.inf]*(len(reference)+1) for _ in range(len(unique)+1)];cost[0][0]=0.
    for i,a in enumerate(unique,1):
        for j,b in enumerate(reference,1):cost[i][j]=math.dist(a,b)+min(cost[i-1][j],cost[i][j-1],cost[i-1][j-1])
    return math.exp(-cost[-1][-1]/(3*len(reference)))


def stats(rows):
    return dict(n=len(rows),successes=int(sum(x['success'] for x in rows)),**{
        output:statistics.mean(x[key] for x in rows) if rows else None for output,key in
        [('sr','success'),('spl','spl'),('ne_m','navigation_error_m'),('osr','oracle_success'),('ndtw','ndtw'),('sdtw','sdtw')]})


def main():
    c.verify_lock();p=read(HERE/'PROTOCOL.json');launch=read(RUN/'LAUNCH_RESULT.json')
    scheduled=read(HERE/'EPISODES_PRIVILEGED.json');gt=json.load(gzip.open(p['gt_path']))
    rows=[];audited=0
    for lane in range(p['lanes']):
        folder=RUN/'lanes'/f'lane_{lane:02d}'
        completed=[read(x) for x in sorted(folder.glob('episode_*.json'))];lookup={x['index']:x for x in completed}
        for e in completed:
            truth=scheduled[e['index']];assert e['episode_id']==truth['episode_id'] and e['house']==truth['scene_id'].split('/')[-2]
            assert len(e['positions'])==len(e['distances'])==e['steps']+1 and 1<=e['steps']<=500
            assert e['stopped'] or e['steps']==500
            success=float(e['stopped'] and e['distances'][-1]<3.)
            path=sum(math.dist(a,b) for a,b in zip(e['positions'],e['positions'][1:]))
            spl=success*e['distances'][0]/max(e['distances'][0],path)
            assert success==e['success'] and e['navigation_error_m']==e['distances'][-1]
            assert e['oracle_success']==float(min(e['distances'])<3.)
            assert math.isclose(path,e['path_length_m'],rel_tol=1e-5,abs_tol=1e-5)
            assert math.isclose(spl,e['spl'],rel_tol=1e-5,abs_tol=1e-5)
            ndtw=exact_ndtw_check(e['positions'],gt[str(e['episode_id'])]['locations'])
            assert math.isclose(ndtw,e['ndtw'],rel_tol=1e-9,abs_tol=1e-9)
            assert math.isclose(success*ndtw,e['sdtw'],rel_tol=1e-9,abs_tol=1e-9)
        counters=collections.defaultdict(collections.Counter);last_steps=collections.defaultdict(int)
        if (folder/'POLICY_STEPS.jsonl').exists() and (folder/'STEPS_PRIVILEGED.jsonl').exists():
            for policy,step in itertools.zip_longest(records(folder/'POLICY_STEPS.jsonl'),records(folder/'STEPS_PRIVILEGED.jsonl')):
                if policy is None or step is None:
                    assert launch['status']!='COMPLETE';continue
                assert policy['index']==step['index'] and policy['step']==step['step'] and policy['action']==step['action']
                index=step['index'];st=step['step'];assert st==last_steps[index]+1;last_steps[index]=st
                assert policy['images']==min(2,st) and policy['executed_history']==min(8,st-1)
                counters[index][step['action']]+=1;counters[index]['collisions']+=int(step['collided'])
                if index in lookup:
                    e=lookup[index];assert step['position']==e['positions'][st] and step['distance_to_goal']==e['distances'][st]
                    assert step['action']!='STOP' or st==e['steps']
                audited+=1
        for index,e in lookup.items():
            assert last_steps[index]==e['steps']
            assert all(counters[index][a]==n for a,n in e['action_counts'].items())
            assert counters[index]['collisions']==e['collisions']
        rows.extend(completed)
    assert len(set(x['index'] for x in rows))==len(rows)
    rows.sort(key=lambda x:x['index']);full=launch['status']=='COMPLETE' and len(rows)==p['episode_count']
    if full:
        assert [x['index'] for x in rows]==list(range(p['episode_count']))
        inference=read(RUN/'INFERENCE_RESULT.json')
        assert inference['completed']==p['episode_count'] and inference['trainable_unchanged'] and inference['optimizer_updates']==0
        assert audited==inference['total_actions']==sum(x['steps'] for x in rows)
    overall=stats(rows);by_house={h:stats([x for x in rows if x['house']==h]) for h in p['houses']}
    old_ids=set(p['previous_tiny_episode_ids'])
    split=dict(previously_exposed_8=stats([x for x in rows if x['episode_id'] in old_ids]),
               other_1831=stats([x for x in rows if x['episode_id'] not in old_ids]))
    ci=None;loo=None
    if full:
        rng=random.Random(1109);boot={key:[] for key in ('sr','spl','ne_m','osr','ndtw','sdtw')}
        houses=p['houses']
        for _ in range(2000):
            selected=[by_house[rng.choice(houses)] for _ in houses];n=sum(x['n'] for x in selected)
            for key in boot:boot[key].append(sum(x[key]*x['n'] for x in selected)/n)
        ci={k:[sorted(v)[49],sorted(v)[1949]] for k,v in boot.items()}
        loo={h:stats([x for x in rows if x['house']!=h]) for h in houses}
    result=dict(status='COMPLETE' if full else 'INCOMPLETE',unix=time.time(),checkpoint_updates=p['checkpoint_updates'],
        checkpoint_sha256=p['checkpoint_sha256'],benchmark='R2R-CE val_unseen v1-3 local preprocessed copy',
        planned=1839,completed=len(rows),missing=1839-len(rows),
        **{k:overall[k] if full else None for k in ('sr','spl','ne_m','osr','ndtw','sdtw')},
        partial_only=overall,house_count=11,by_house=by_house,development_exposure=split,
        house_bootstrap_95_interval=ci,house_bootstrap_replicates=2000 if full else 0,
        uncertainty_note='Descriptive 11-house cluster bootstrap, not a matched-model significance test',
        leave_one_house_out=loo,stopped=sum(int(x['stopped']) for x in rows),
        failure_categories=dict(collections.Counter(x['failure_category'] for x in rows)),
        collisions=sum(x['collisions'] for x in rows),environment_actions=sum(x['steps'] for x in rows),
        trace_audit_passed=True,audited_actions=audited,source_lock_verified=True,
        selected_batch_size=read(RUN/'BATCH_PARITY_GATE.json').get('selected_batch_size'),
        exact_ndtw_FDTW_false=True,optimizer_updates=0,official_full_trainer_or_leaderboard_run=False,
        scientific_gain_verified=False,wall_seconds=launch['wall_seconds'],peak_gpu_mib=launch['peak_gpu_mib'],
        training_processes_signaled=[],run_dir=str(RUN))
    fields=['index','episode_id','trajectory_id','house','success','spl','navigation_error_m','oracle_success','ndtw','sdtw','steps','stopped','collisions','failure_category']
    with (RUN/'episodes.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:x[k] for k in fields} for x in rows)
    def fmt(x,pct=False):return '未完成' if x is None else f'{x*100:.2f}%' if pct else f'{x:.3f}'
    text=f'''# 普通导航基座：完整R2R-CE验证

状态：{result['status']}。固定检查点 {p['checkpoint_updates']} 更新，已完成 {len(rows)}/1839 条，11个房屋。

| 指标 | 完整验证结果 |
|---|---:|
| SR | {fmt(result['sr'],True)} |
| SPL | {fmt(result['spl'],True)} |
| NE | {fmt(result['ne_m'])} m |
| OSR | {fmt(result['osr'],True)} |
| nDTW（精确DTW） | {fmt(result['ndtw'],True)} |
| SDTW（精确DTW） | {fmt(result['sdtw'],True)} |

本轮不是隐藏测试集或榜单提交。官方v0.1.7 SR/SPL类主体未改；独立轻量仿真适配器；原始指令、2帧RGB、8个已执行动作输入，0.25m/15°/500步/STOP且严格<3m，无自动停止、目标提示或最短路回退。nDTW对应官方FDTW=False定义，不是fastdtw近似版本。

## 分房屋

| 房屋 | 完成数 | SR | SPL | NE(m) |
|---|---:|---:|---:|---:|
'''
    for h,x in by_house.items():text+=f'| {h} | {x["n"]} | {fmt(x["sr"],True)} | {fmt(x["spl"],True)} | {fmt(x["ne_m"])} |\n'
    text+='''
## 解释边界

所有原始条目保留，包括同一物理路线的不同指令；主结果按episode等权。旧8条开发暴露和其余1831条分别见RESULT.json。所有11个val房屋均排除在当前训练/内部验证房屋之外；旧研究和基础预训练暴露未知。本地预处理数据副本来源限制见SPEC_ZH.md和旧小测ASSET_AUDIT。

不能把本轮1839条与旧检查点8条均值直接相减当作模型进步。与论文比较还需匹配版本、传感器、历史长度、路点控制器、额外训练数据和评测配置。本轮不判断创新或SOTA。

逐步动作/位置/计分独立审计，以及精确nDTW独立重算通过；未进行参数更新。结果缺失时保持null，不能将部分均值作为全量结果。房屋bootstrap和留一房屋结果仅描述场景波动，11个房屋不支持过强统计结论。

完整机器结果：RESULT.json；逐条表：episodes.csv；原始轨迹/图片/动作logits：lanes/lane_XX。最终以RESULT.json与LAUNCH_RESULT.json为准，原训练网页18766未改。本次只读监控127.0.0.1:18768。
'''
    (RUN/'REPORT_ZH.md').write_text(text)
    c.write(RUN/'RESULT.json',result,True)
    c.write(RUN/'PROGRESS.json',dict(status=result['status'],unix=time.time(),completed=len(rows),total=1839,
        checkpoint_updates=p['checkpoint_updates'],total_actions=result['environment_actions'],final_report=str(RUN/'REPORT_ZH.md')))
    print(json.dumps({k:result[k] for k in ('status','completed','sr','spl','ne_m','osr','ndtw','sdtw','trace_audit_passed')},ensure_ascii=False))


if __name__=='__main__':main()
