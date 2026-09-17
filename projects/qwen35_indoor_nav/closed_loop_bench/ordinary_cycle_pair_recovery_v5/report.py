"""One handoff report built from actual engineering and pilot records."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c


def main():
    r=c.read(HERE/'RESULT.json')
    pilot=c.LINE/'research/continuation_memory_v1/pilot'
    pilot_runs=sorted(pilot.glob('run_*'))
    p_result=None
    for folder in pilot_runs:
        if (folder/'RESULT.json').exists(): p_result=c.read(folder/'RESULT.json')
    pilot_seconds=sum(c.read(folder/'LAUNCH_RESULT.json')['gpu_seconds'] for folder in pilot_runs if (folder/'LAUNCH_RESULT.json').exists())
    launches=[c.read(path) for path in sorted((HERE/'sessions').glob('session_*/LAUNCH_RESULT.json'))]
    lines=['ENGINEERING_COMPLETE' if r['experiment_complete'] else 'ENGINEERING_PARTIAL', '',
        f"实际加载模型：{r['model_loaded']}；完整有效pair：{r['complete_valid_pairs']}/100；待补：{r['pending_pairs']}。",
        f"工程累计GPU小时：{r['gpu_hours']:.6f}；pilot累计GPU小时：{pilot_seconds/3600:.6f}。",'',
        'GPU小时按本任务launcher运行窗口计入预算，包含加载、编译与失败启动，不是GPU内核活跃时长。',
        '5→20→100为同一个冻结实验的累计进度，对应MILESTONE_005/020/100.json；前段没有因分数被重选。',
        '本轮使用V5行为配对协议，替代旧专用窗口、4100秒及一次启动限制；旧V3/V4 UNKNOWN不追溯改写。',
        '实验完成、输入一致、动作一致、数值逐位一致、干预收益和是否采用分别记录。','',
        '|指标|原生A|恢复B|B−A|','|---|---:|---:|---:|']
    for key in ('sr','spl','ndtw','osr'):
        a,b,d=r['arms']['A'][key],r['arms']['B'][key],r['delta'][key]
        lines.append(f'|{key}|{a}|{b}|{d}|')
    if r['experiment_complete'] and r['delta']['sr']<=0:
        lines += ['', '完整100对未观察到SR收益，本轮不采用该恢复规则。重复决策减少不能解释为任务成功率提高。']
    lines += ['',f"配对胜episode：{r['wins']}；负episode：{r['losses']}；保留成功数：{r['retained_successes']}。",
        f"输入前缀一致：{r['audit']['input_prefix_matched']}；动作前缀一致：{r['audit']['action_prefix_matched']}；",
        f"logits逐位一致：{r['audit']['logits_bitwise_equal']}；最大logit差：{r['audit']['max_logit_delta']}；干预前argmax翻转：{r['audit']['argmax_flip_count']}。",
        '运行身份及加载后/分段末全参数和持久buffer指纹在sessions；无覆盖pair检查完整轨迹与终态。',
        f"行为比较有效：{r['behavioral_comparison_valid']}；完整100对效应可评估：{r['full100_effect_evaluable']}；",
        f"开发正向信号：{r['development_positive_signal']}；采用：{r['adopted']}。",
        'SR效果、路径质量、重复动作和资源成本分别报告，不自动上线。','',
        '|成本|A|B|','|---|---:|---:|']
    for key in ('total_decisions','repeated_decisions','overrides','collisions'):
        lines.append(f"|{key}|{r['arms']['A'][key]}|{r['arms']['B'][key]}|")
    lines += ['', '|屋|ΔSR|ΔSPL|ΔnDTW|','|---|---:|---:|---:|']
    for house,delta in r['by_house_delta'].items():lines.append(f"|{house}|{delta['sr']}|{delta['spl']}|{delta['ndtw']}|")
    lines += ['',f"描述性配对区间：{r['paired_uncertainty']}。",
        'episode/屋内相关且只有5屋；INTERNAL_DEV已暴露，不能据此主张盲测泛化。','']
    if r['experiment_complete'] and not r['wins'] and not r['losses']:
        lines += ['100对成功差值全为0，使经验重采样区间退化为[0,0]；这不意味着总体效应已被精确确定为0。','']
    for arm in ('A','B'):
        x=r['arms'][arm]
        lines += [f"{arm} STOP分型：{x['failure_classes']}。",f"{arm} 时延秒（p50/p95）：{x['latency']}。",
            f"{arm} 按资源竞争分开的推理时延：{x['inference_latency_by_observed_competition']}。",'']
    lines += ['资源是共享使用；竞争按5秒监视快照分类，预处理耗时含审计哈希，不能宣传为独占或真机时延。',
        f"数值/服务问题：{len(r['issues'])}，逐次原因及100对完整清单见RESULT.json。",
        '半对只保留为尝试，不拼接单臂。已完成且有参数核验的pair可续用；没有按分数重试。','',
        '研究pilot：']
    lines[-1:-1]=[f"工程session数：{len(launches)}；基础设施失败尝试：{sum(x['infrastructure_retry'] for x in launches)}；"
        f"本任务峰值显存MiB：{max((x['peak_own_memory_mib'] for x in launches),default=0)}；"
        f"进程组峰值RSS GiB：{max((x['peak_group_rss_bytes'] for x in launches),default=0)/1024**3:.3f}。",'']
    for folder in pilot_runs:
        if (folder/'FAILURE.json').exists():
            failure=c.read(folder/'FAILURE.json')
            correction=c.read(folder/'CORRECTION.json') if (folder/'CORRECTION.json').exists() else {}
            lines += [f"保留失败尝试{folder.name}：{failure['error']}；定位：{correction.get('root_cause','尚未定位')}。"]
    if p_result:
        g=p_result['gradient_update']
        lines += [f"实际结果：{p_result['status']}。真实冻结Qwen特征、249观察递归展开、关键事件写入梯度、动作读出和参数更新均已测。",
            f"关键H_B/task_B第14步写入到第248步监督的梯度范数：{g['critical_event']['writer_output_gradient_norm']}。",
            f"动作loss对运行memory梯度范数：{g['action_memory_gradient_norm']}；memory替换使动作logits改变的最大值：{g['action_memory_effect_max_abs']}。",
            f"诊断更新{p_result['diagnostic_updates']}次；B2/Ours各{p_result['arms']['B2']['updates']}次，使用同一初始状态、因果特征、492动作owner及预算。",'']
        for arm,row in p_result['arms'].items(): lines.append(f"{arm}：初始{row['initial']}；结束{row['final']}；训练秒{row['seconds']}。")
        lines += [f"结束时有效query拟合准确率：B2 {p_result['arms']['B2']['final']['effective_query_accuracy']:.2%}，"
            f"Ours {p_result['arms']['Ours']['final']['effective_query_accuracy']:.2%}；本次没有Ours优于精确状态监督的证据。"]
        lines += ['', '上述数值是单个已暴露SEE2族的拟合/实现检查，没有独立族或记忆闭环评测。',
            'B2的有效query指标由预测精确状态加合法续接事件经原任务逻辑计算，Ours用其训练读出器；不拿B2未训练的交叉读出头充当弱对照。',
            '精确状态监督更密集，辅助反向成本也不同，实际调用与耗时已披露；这个小pilot不支持Ours优越性，也不裁定整个方向失败。',
            '底层Qwen/视觉编码器未训练；原始证书的training_admission=false保持不变。',
            'SEE2不冒充到访房间，既有规范化与负控制缺口保留。未来查询只进入训练读出器；科学主张仍UNTESTED。']
    else:
        lines += ['完整研究pilot未完成；逐次失败/资源记录保留在pilot/run_*。']
        for folder in pilot_runs:
            for name in ('FEATURE_RESULT.json','GRADIENT_UPDATE_RESULT.json','FAILURE.json'):
                if (folder/name).exists():lines.append(f"已发生的{folder.name}/{name}：{c.read(folder/name)}。")
        lines += ['CPU检查不替代上述真实前向、梯度或更新证据；未执行项不填PASS。']
    lines += ['',f"工程CPU检查：{c.read(HERE/'CPU_TEST_RESULT.json')}；pilot CPU检查：{c.read(pilot/'CPU_TEST_RESULT.json')}。"]
    if (pilot/'CPU_REPAIR_RESULT.json').exists():
        lines += [f"pilot接口修复CPU回归：{c.read(pilot/'CPU_REPAIR_RESULT.json')}。"]
    if (pilot/'SAVED_EVIDENCE_AUDIT.json').exists():
        lines += ['pilot/SAVED_EVIDENCE_AUDIT.json已从实际保存的参数文件复核两臂共同初始化、更新后的参数指纹、训练步数及FEATURES哈希。']
    if (HERE/'SAVED_EVIDENCE_AUDIT.json').exists():
        lines += ['',f"已保存全量证据独立复核：{c.read(HERE/'SAVED_EVIDENCE_AUDIT.json')}。",
            '实际编译出的Triton配置见sessions/*/TRITON_CACHE_METADATA.json；它不是逐决策kernel选择轨迹，未把不可获得的选择信息补成已验证。']
    lines += ['', '实际运行命令（vla根目录）：','```bash',
        '.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/closed_loop_bench/ordinary_cycle_pair_recovery_v5/launch.py --gpu 1 --target 100',
        '.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/pilot/launch.py',
        '```','',
        '入口、协议、CPU检查、逐pair结果及原生/实际动作日志均在本目录；pilot实现及真实更新记录在research/continuation_memory_v1/pilot。',
        '审阅起点7f65fd04b54cea7f3844ec128a493f75501bbeb2；源文件身份见SOURCE_LOCK与每session/source，best4k和冻结模型未改。',
        'ROOT现有.gitignore/AGENTS/README修改保留。CURRENT_STATUS更新V5事实，不覆盖旧报告、旧checkpoint或部署配置。',
        'GitHub交付代码、配置、逐步JSON日志、参数指纹与汇总。原始RGB、编译缓存、FEATURES.pt及pilot的INITIAL/B2/Ours参数文件保留本地；参数文件路径和SHA见pilot/SAVED_EVIDENCE_AUDIT.json。复跑仍需当地原有模型、场景和数据资产。',
        '下一判断应依据本轮完整效应及成本，并另定记忆独立族/闭环测量范围。没有SR40、真机部署或论文创新完成声明。',
        '本轮交付后停止，不自动长训、扩LoRA、扩历史或安装方向B环境。','']
    (HERE/'REPORT_ZH.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
