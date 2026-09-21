"""Post-run descriptive comparison; no new selection thresholds or model changes."""
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import read,immutable,sha,HERE

def main(run):
    rows=read(run/'ROLLOUTS.json');result=read(run/'RESULT.json')
    assert result['complete']==1152 and result['groups']==128
    arms=['DIRECT','MONOTONIC','REVISE'];tables=[];pairs=[]
    for endpoint in ('main','control'):
        for seed in (1209,1210,1211):
            selected=[r for r in rows if r['endpoint']==endpoint and r['seed']==seed]
            table={a:sum(r['status']=='PASS' for r in selected if r['arm']==a) for a in arms}
            tables.append(dict(endpoint=endpoint,seed=seed,n=64,passed=table))
            index={(r['condition'],r['arm']):r for r in selected}
            for baseline in ('DIRECT','REVISE'):
                wins=[];losses=[];ties=[]
                for r in selected:
                    if r['arm']!='MONOTONIC':continue
                    assert r['status'] in ('PASS','FAIL')
                    other=index[(r['condition'],baseline)];assert other['status'] in ('PASS','FAIL')
                    if r['status']==other['status']:ties.append(r['condition'])
                    elif r['status']=='PASS':wins.append(r['condition'])
                    else:losses.append(r['condition'])
                pairs.append(dict(endpoint=endpoint,seed=seed,baseline=baseline,wins=wins,losses=losses,ties=ties))
    immutable(HERE/'RESULT_READOUT.json',dict(result_sha256=sha(run/'RESULT.json'),rollouts_sha256=sha(run/'ROLLOUTS.json'),
        tables=tables,pairs=pairs,preregistered_primary='REVISE minus DIRECT',primary_main_difference_pp=100*2/192,
        secondary_monotonic_minus_direct_main_pp=100*11/192,interpretation='MONOTONIC is a development candidate discovered in the registered comparison, not confirmation of the original REVISE-superiority hypothesis.',
        independent_houses=1,next_gpu_experiment_launched=False))
    body='''# 完整结果：MONOTONIC 优先作为后续验证候选

1152/1152续接、128/128完整组，未知0。九模型均固定final1200。没有新增训练或导航。

|方法|历史任务PASS|历史无关控制PASS|主任务预算惩罚成本（低为好）|
|---|---:|---:|---:|
|DIRECT|55/192（28.65%）|78/192（40.63%）|0.7619|
|MONOTONIC|66/192（34.38%）|98/192（51.04%）|0.7115|
|REVISE|57/192（29.69%）|98/192（51.04%）|0.7520|

MONOTONIC相对DIRECT主任务+5.73个百分点，控制任务+10.42个百分点。主任务三个种子为22/18、23/19、21/18（MONOTONIC/DIRECT），方向一致；逐条件配对28胜17负147平。相对REVISE为26胜17负149平，在2/3种子更高。

优势集中在历史事件已发生条件：MONOTONIC 58/96、DIRECT 46/96、REVISE 49/96。历史事件缺失条件分别8/96、9/96、8/96；当前没有修通主动补做所需历史事件的恢复能力。任务定义中的真实“曾经发生”状态具有单调性，这是解释候选；现有比较未单独分离单调累积、解析状态组合及优化路径的作用，也未证明REVISE下降一定来自遗忘。

原登记主比较REVISE−DIRECT仅+1.04个百分点，种子方向不一致，且被MONOTONIC超过。不将观察到的MONOTONIC优势改写成原REVISE优越性假设已证实。建议优先冻结MONOTONIC候选、在独立房屋与任务族确认，同时保留DIRECT及简单规则对照；本轮不启动下一项GPU实验。

这是一个已暴露DEV房屋、八父族上的开发正向信号；三种子不是三个独立房屋。不声称普通VLN提升、论文贡献已成立或部署通过。

末尾校准CPU统计曾因对见证实例列表直接int()而失败，版本化calibration_r2按冻结训练语义使用int(bool(list))，None继续屏蔽；主评测RESULT与轨迹未变。最终校准输出记录新后处理源码SHA。旧失败日志保留。
'''
    with (HERE/'RESULT_READOUT_ZH.md').open('x') as f:f.write(body)

if __name__=='__main__':main(Path(sys.argv[1]))
