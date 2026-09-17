"""Post-run, CPU-only aggregation and trace audit; never imports or calls the policy."""
import collections
import csv
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

HERE=Path(__file__).resolve().parent
EVAL=HERE.parent;RUN=EVAL/'run_001'
s=importlib.util.spec_from_file_location('common',EVAL/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def load(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
def quantile(values,q):
    values=sorted(values)
    if not values:return None
    i=(len(values)-1)*q;lo=int(i);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(i-lo)


def main():
    assert (RUN/'LAUNCH_RESULT.json').exists(),'WAIT_FOR_LAUNCHER_CLEANUP'
    assert not (HERE/'RESULT.json').exists(),'POSTRUN_RESULT_ALREADY_WRITTEN'
    c.verify_lock()
    protocol=load(EVAL/'PROTOCOL.json');launch=load(RUN/'LAUNCH_RESULT.json')
    scheduled=load(EVAL/'EPISODES_PRIVILEGED.json')
    episodes=[load(x) for x in sorted(RUN.glob('episode_*.json'))]
    policy=lines(RUN/'POLICY_STEPS.jsonl');steps=lines(RUN/'STEPS_PRIVILEGED.jsonl')
    errors=[];rows=[]
    for e in episodes:
        index=e['index'];truth=scheduled[index]
        assert e['episode_id']==truth['episode_id'] and e['trajectory_id']==truth['trajectory_id']
        pp=[x for x in policy if x['index']==index];ss=[x for x in steps if x['index']==index]
        assert len(pp)==len(ss)==e['steps']
        assert [x['step'] for x in pp]==[x['step'] for x in ss]==list(range(1,e['steps']+1))
        assert [x['action'] for x in pp]==[x['action'] for x in ss]
        assert dict(collections.Counter(x['action'] for x in ss))=={k:v for k,v in e['action_counts'].items() if v}
        assert len(e['positions'])==len(e['distances'])==e['steps']+1
        assert all(x['position']==p and x['distance_to_goal']==d for x,p,d in zip(ss,e['positions'][1:],e['distances'][1:]))
        assert not any(x['action']=='STOP' for x in ss[:-1])
        stopped=ss[-1]['action']=='STOP';assert stopped==e['stopped']
        assert stopped or e['steps']==500
        distance=e['distances'][-1];success=float(stopped and distance<3.)
        osr=float(any(x<3. for x in e['distances']))
        traveled=sum(math.dist(a,b) for a,b in zip(e['positions'],e['positions'][1:]))
        spl=success*e['distances'][0]/max(e['distances'][0],traveled)
        assert success==e['success'] and osr==e['oracle_success'] and distance==e['navigation_error_m']
        assert math.isclose(spl,e['spl'],abs_tol=1e-5,rel_tol=1e-5)
        assert math.isclose(traveled,e['path_length_m'],abs_tol=1e-5,rel_tol=1e-5)
        assert sum(x['collided'] for x in ss)==e['collisions']
        assert all(row['images']==min(2,row['step']) and row['executed_history']==min(8,row['step']-1) for row in pp)
        longest=0;run=0;previous=None
        for x in ss:
            run=run+1 if x['action']==previous else 1;longest=max(longest,run);previous=x['action']
        row={k:e[k] for k in ('index','episode_id','trajectory_id','house','steps','stopped','success','spl',
            'navigation_error_m','oracle_success','min_distance_m','start_distance_m','path_length_m','collisions','failure_category')}
        row.update(longest_identical_action_run=longest,action_counts=e['action_counts'])
        rows.append(row)
    n=len(episodes);successes=sum(int(x['success']) for x in episodes)
    complete=launch['status']=='COMPLETE' and n==8
    inference=load(RUN/'INFERENCE_RESULT.json') if (RUN/'INFERENCE_RESULT.json').exists() else {}
    if complete:
        assert inference['completed']==8 and inference['optimizer_updates']==0 and inference['trainable_unchanged']
        assert inference['total_actions']==len(policy)==len(steps)==sum(x['steps'] for x in episodes)
        assert not launch['cleanup']['remaining'] and launch['training_processes_unchanged']
    latencies=[x['inference_seconds'] for x in policy]
    mean=lambda k:statistics.mean(x[k] for x in episodes) if n else None
    result=dict(created_unix=time.time(),status='COMPLETE' if complete else launch['status'],
        benchmark='R2R-CE val_unseen local preprocessed v1-3 copy',checkpoint_updates=34800,
        checkpoint_sha256=protocol['checkpoint_sha256'],planned_episodes=8,completed_episodes=n,
        uncompleted_episodes=8-n,successful_episodes=successes,houses=4,
        success_rate=successes/8 if complete else None,success_rate_completed_only=successes/n if n else None,
        planned_success_bounds=[successes/8,(successes+8-n)/8],
        spl=mean('spl'),navigation_error_m=mean('navigation_error_m'),oracle_success_rate=mean('oracle_success'),
        stopped_episodes=sum(int(x['stopped']) for x in episodes),total_actions=len(steps),
        total_collisions=sum(x['collisions'] for x in episodes),
        failure_categories=dict(collections.Counter(x['failure_category'] for x in episodes)),
        inference_p50_seconds=quantile(latencies,.5),inference_p95_seconds=quantile(latencies,.95),
        first_forward_seconds=latencies[0] if latencies else None,
        wall_seconds=launch['wall_seconds'],peak_gpu_mib=launch['peak_gpu_mib'],
        training_processes_unchanged=launch['training_processes_unchanged'],optimizer_updates=0,
        trace_audit_passed=True,official_metric_class_bodies_unchanged=True,
        official_full_trainer_or_leaderboard_run=False,ndtw=None,sdtw=None,
        infer_statistical_improvement=False,records=rows)
    c.write(HERE/'RESULT.json',result,True)
    fields=['index','episode_id','house','success','spl','oracle_success','navigation_error_m','min_distance_m','steps','stopped','collisions','failure_category']
    with (HERE/'episodes.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:x[k] for k in fields} for x in rows)
    # Raw trace plot, not generated room imagery. Identical axes across eight panels.
    ymax=max(5.,math.ceil(max((max(e['distances']) for e in episodes),default=5.)))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" viewBox="0 0 1200 820">',
         '<rect width="1200" height="820" fill="#f5f7fb"/>',
         '<g font-family="sans-serif" fill="#14253b"><text x="30" y="32" font-size="22">R2R-CE val_unseen: fixed 8-episode pilot / checkpoint 34,800</text>',
         '<text x="30" y="57" font-size="14">Geodesic distance to goal (m) vs executed actions. Green line: 3 m threshold; STOP is still required.</text>']
    for i in range(8):
        x0=34+(i%2)*590;y0=105+(i//2)*176;w=530;h=117
        svg.append(f'<rect x="{x0-15}" y="{y0-32}" width="570" height="164" rx="8" fill="white"/>')
        e=next((x for x in episodes if x['index']==i),None)
        if e is None:
            svg.append(f'<text x="{x0}" y="{y0}">Episode {scheduled[i]["episode_id"]}: not completed</text>');continue
        color='#147d64' if e['success'] else '#b34f45'
        title=f'{i+1}. ep {e["episode_id"]} | {"SUCCESS" if e["success"] else "FAIL"} | {e["steps"]} actions | NE {e["navigation_error_m"]:.2f} m'
        svg.append(f'<text x="{x0}" y="{y0-13}" font-size="15">{title}</text>')
        for d in (0,3,ymax):
            yy=y0+h-d/ymax*h
            svg.append(f'<path d="M{x0},{yy} h{w}" stroke="{"#38a56e" if d==3 else "#d5dce6"}" stroke-dasharray="4 3"/><text x="{x0-5}" y="{yy+4}" text-anchor="end" font-size="10">{d:g}</text>')
        points=' '.join(f'{x0+step/500*w:.2f},{y0+h-d/ymax*h:.2f}' for step,d in enumerate(e['distances']))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        xx=x0+e['steps']/500*w;yy=y0+h-e['distances'][-1]/ymax*h
        svg.append(f'<circle cx="{xx}" cy="{yy}" r="4" fill="{color}"/>')
        for step in (0,100,250,500):svg.append(f'<text x="{x0+step/500*w}" y="{y0+h+12}" font-size="10" text-anchor="middle">{step}</text>')
    svg.append('</g></svg>');(HERE/'distance_traces.svg').write_text(''.join(svg))
    category={'success':'成功停止','entered_goal_but_no_successful_stop':'曾到目标附近，但未正确停止',
              'stopped_without_reaching_goal':'尚未到目标附近就停止',
              'budget_exhausted_without_reaching_goal':'未到目标附近，耗尽步数'}
    table=['| 路线ID | 房屋 | 成功 | SPL | 曾到3m内 | 终点距离(m) | 最小距离(m) | 动作数 | 碰撞 | 结束 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for x in rows:
        table.append(f'| {x["episode_id"]} | {x["house"]} | {int(x["success"])} | {x["spl"]:.3f} | {int(x["oracle_success"])} | {x["navigation_error_m"]:.2f} | {x["min_distance_m"]:.2f} | {x["steps"]} | {x["collisions"]} | {"STOP" if x["stopped"] else "500步耗尽"} |')
    heading=f'完成 {n}/8 条；成功 {successes}/{n} 条已完成路线。'
    if complete:heading=f'8 条真实闭环全部完成，成功 {successes}/8（{successes/8:.1%}），SPL {result["spl"]:.3f}。'
    md=f'''# 普通导航基座：首轮真实基准小测

{heading} 固定检查点为 34,800 更新。本次不训练、不使用最短路回退或自动停止。

这是 R2R-CE val_unseen 的 8 条不同物理路线、4 个房屋，只用于判断当前模型是否已经能自主完成基本导航。不是完整官方测试，不足以证明相对之前检查点提升或与其他论文的排名。

## 结果

平均终点距离：{result['navigation_error_m']:.3f} m；曾进入目标 3 m 内：{sum(int(x['oracle_success']) for x in episodes)}/{n}；主动 STOP：{result['stopped_episodes']}/{n}；实际动作：{len(steps)}。

{chr(10).join(table)}

失败分类：{'; '.join(category[k]+' '+str(v)+' 条' for k,v in result['failure_categories'].items())}。

![逐步到目标距离](distance_traces.svg)

## 如何理解

训练网页的动作准确率或 STOP 召回是单步指标；这里的成功必须是模型实际走完并自己发出 STOP，且最终测地距离严格小于 3 m。

“曾到目标附近”但未成功，才直接提示停止或离开目标附近的问题；如果从未接近目标，不能只归因于 STOP。官方 SR 的 3 m 容差也不保证逐句路线严格遵循；本轮未计算 nDTW/SDTW，不能据此宣称指令每一步都已正确执行。

## 核验与边界

- 8 条路线按预先冻结的哈希规则选出，运行中未替换失败路线、未追逐更新的检查点。
- 当前训练 FIT51 和内部 DEV/CONFIRM 房屋与官方 val_unseen 11 屋均无交集；其他旧研究暴露及基础模型预训练暴露未知。
- 使用项目既有 R2R_VLNCE_v1-3_preprocessed_xlmr 副本中的原始文本/几何；其全部 10,819 条 train 文本/几何与登记的官方 minimal train 一致。val 副本未因本机下载 TLS 失败而独立重新下载验证；来源、文件哈希与选择清单已固定。
- 运行 Habitat-Sim 0.1.7；相机 224×224、90°、1.25 m；前进 0.25 m、转向 15°、allow_sliding=True。该 sliding 配置遵循官方基准，与训练示范的 False 有差异，未事后调整。
- 使用原封不动的官方 Habitat-Lab v0.1.7 DistanceToGoal、Success、SPL 类主体，独立轻量仿真执行接口；不是完整官方 trainer 或榜单提交。
- 模型只接收原始指令、最近两张 RGB、最近八个已执行动作名。位姿、目标、距离与参考路径只留在仿真/计分端。
- 14 项 CPU 接口与指标测试通过；真实起点/朝向/测地距离核验通过；策略动作与仿真步账逐条一致，独立重算 SR/SPL/OSR 通过；0 次参数更新。
- 单卡运行 {result['wall_seconds']:.1f} 秒（含初始化），显存峰值 {result['peak_gpu_mib']/1024:.2f} GiB。动作推理耗时 p50={result['inference_p50_seconds']:.3f} 秒，p95={result['inference_p95_seconds']:.3f} 秒；首步 {result['first_forward_seconds']:.2f} 秒包含冷编译。仅本次硬件/接口耗时，不是论文速度对比。
- 三卡训练进程身份未改变：{result['training_processes_unchanged']}。仅本节点自身进程清理，无占位借用，无外部进程信号。

数据文件：[逐路线CSV](episodes.csv)、[聚合结果](RESULT.json)、[实际画面与原始指令](index.html)。原始轨迹和每步 logits 保留在 ../run_001，SOURCE_LOCK.json 绑定全部输入和运行代码。

一手定义：[官方配置](https://raw.githubusercontent.com/jacobkrantz/VLN-CE/master/habitat_extensions/config/vlnce_task.yaml)、[官方数据](https://jacobkrantz.github.io/vlnce/data)、[Habitat-Lab指标](https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py)。
'''
    (HERE/'REPORT_ZH.md').write_text(md)
    parts=['<!doctype html><meta charset="utf-8"><title>普通导航基座：R2R-CE小测</title>',
           '<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:35px auto;padding:0 18px;color:#16283d;background:#f5f7fb}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.frames{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}figcaption{font-size:13px}img.rgb{width:224px;height:224px;image-rendering:auto}h1,h2{line-height:1.3}</style>',
           '<h1>普通导航基座 · 首轮真实闭环小测</h1><p>'+html.escape(heading)+'</p>',
           '<p>固定 34,800 更新检查点；8 条 / 4 屋；非完整基准成绩。下面均为本次仿真原始画面，不是生成示意图。</p>',
           '<object data="distance_traces.svg" type="image/svg+xml" style="width:100%"></object>']
    for e in episodes:
        parts.append(f'<section><h2>{e["index"]+1}. 路线 {e["episode_id"]}：{category[e["failure_category"]]}</h2>')
        parts.append('<p>'+html.escape(e['instruction'])+'</p>')
        parts.append(f'<p>动作 {e["steps"]}；SPL {e["spl"]:.3f}；终点距离 {e["navigation_error_m"]:.2f} m；最小距离 {e["min_distance_m"]:.2f} m；碰撞 {e["collisions"]}。</p><div class="frames">')
        available=sorted((RUN/'frames').glob(f'ep_{e["index"]:02d}_step_*.png'))
        chosen=sorted(set([available[0],available[len(available)//2],available[-1]]))
        for path in chosen:
            label='动作 '+str(int(path.stem.split('_')[-1]))
            parts.append(f'<figure><img class="rgb" src="../run_001/frames/{path.name}" alt="{label}"><figcaption>{label} · 原始 RGB</figcaption></figure>')
        parts.append('</div></section>')
    parts.append('<p>完整口径见 REPORT_ZH.md；逐路线数据见 episodes.csv。</p>')
    (HERE/'index.html').write_text(''.join(parts))
    evidence={str(x):c.sha(x) for x in sorted(RUN.glob('*.json'))+sorted(RUN.glob('*.jsonl'))}
    evidence[str(HERE/'build_report.py')]=c.sha(HERE/'build_report.py')
    c.write(HERE/'EVIDENCE_BINDINGS.json',evidence,True)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
