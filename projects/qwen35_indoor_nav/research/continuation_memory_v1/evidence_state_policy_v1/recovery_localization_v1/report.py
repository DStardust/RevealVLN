"""Summarize measured positives and localize the next repair without inventing efficacy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import BASE,HERE,LINE,read,sha,immutable,write

def main(out):
    out=out.resolve()
    probe=read(out/'CAUSAL_PROBE.json');live=read(out/'ORIGINAL_RECOVERY.json');result=read(out/'RESULT.json')
    dev=read(BASE/'gpu_runtime_r1/runs/gpu_001/RESULT.json');held=read(BASE/'monotonic_holdout_v1/runs/holdout_001/RESULT.json')
    repair=read(BASE/'teacher_alignment_v1/runs/aligned_001/RESULT.json')
    def hist(split,history):return next(r for r in probe['summary'] if r['seed'] is None and r['split']==split and r['history']==history)
    def ls(history):return next(r for r in live['summary'] if r['seed'] is None and r['history']==history)
    missing=ls('missing');fit=hist('FIT','all');check=hist('DEV','all')
    evidence=[dict(claim='Original development MONOTONIC vs DIRECT',status='MEASURED_DEV_POSITIVE_SIGNAL',
        source=str((BASE/'gpu_runtime_r1/runs/gpu_001/RESULT.json').relative_to(LINE)),
        main=dict(direct=55,monotonic=66,n_per_arm=192,delta=11/192),control=dict(direct=78,monotonic=98,n_per_arm=192,delta=20/192),
        scope='One exposed DEV house; method comparison under matched training. Not independent ordinary VLN-CE evidence.'),
        dict(claim='Four-house follow-up',status='WEAK_INCOMPLETE_SIGNAL',
        source=str((BASE/'monotonic_holdout_v1/runs/holdout_001/RESULT.json').relative_to(LINE)),
        complete=1488,planned=1536,main=dict(direct=125,monotonic=131,executed_per_arm=372,planned_per_arm=384,unidentified_per_arm=12,delta_bounds=[-6/384,18/384]),
        scope='Two houses positive, two negative. Missing conditions and dependence preclude robust claim; houses now exposed.'),
        dict(claim='Task-aligned teacher intervention',status='VALID_NEGATIVE_RESULT_DO_NOT_ADOPT',
        source=str((BASE/'teacher_alignment_v1/runs/aligned_001/RESULT.json').relative_to(LINE)),
        complete=repair['complete'],planned=repair['planned'],main=dict(original=66,aligned=47,n_per_arm=192,delta=-19/192),
        scope='Earlier teacher STOP plus reduced movement action masks harmed performance; effects of the two changes were not separately identified.'),
        dict(claim='Current causal localization',status='MEASURED_DIAGNOSIS_NOT_NEW_METHOD_GAIN',
        source=str((out/'RESULT.json').relative_to(LINE)),teacher_sequence_forwards=result['teacher_sequence_forwards'],
        causal_step_forwards=probe['counts']['causal_step_forwards'],live_main_rollouts=192,gpu_hours=0,optimizer_updates=0)]
    for e in evidence:e['source_sha256']=sha(LINE/e['source'])
    immutable(out/'EVIDENCE_LEDGER.json',evidence)
    init=[]
    for seed in (1209,1210,1211):
        a=read(BASE/'gpu_v1/runs/gpu_001/train'/f'MONOTONIC_{seed}'/'RESULT.json')
        b=read(BASE/'teacher_alignment_v1/runs/aligned_001/train'/f'ORIGINAL_{seed}'/'RESULT.json')
        init.append(dict(seed=seed,same_initial=a['initial']==b['initial'],same_final=a['final']==b['final']))
    immutable(out/'REPRODUCTION.json',dict(models=init,note='Identical rerun weights and same benchmark outcome show reproducibility, not independent statistical replication.'))
    support=read(out/'EVENT_SUPPORT.json') if (out/'EVENT_SUPPORT.json').exists() else None
    next_step=dict(status='LOCALIZATION_COMPLETE_NEXT_REPAIR_TARGET_IDENTIFIED',retain='Original MONOTONIC',reject='ALIGNED earlier-teacher-STOP intervention',
        priority='Task-conditioned event recognition under the existing input contract, with explicit measurement of accumulated false evidence.',
        evidence=dict(missing_history_false_belief_at_takeover=missing['takeover_history']['fp'],missing_history_denominator=96,
            failed_stop_with_false_history=missing['failure_types']['STOP_MISSING_HISTORY_WITH_FALSE_BELIEF'],
            failed_stop_despite_correct_missing_history=missing['failure_types']['STOP_DESPITE_CORRECT_MISSING_HISTORY'],
            first_witness_detection=dict(fit=[fit['first_witness_detected'],fit['witnessed_n']],dev=[check['first_witness_detected'],check['witnessed_n']]),
            detected_then_lost=check['detected_then_lost']),
        immediate_next_work=['Audit existing FIT event support by target words, visibility and house; distinguish unseen composition/wording from raw data scarcity.',
            'Build a bounded FIT-only event diagnostic/repair pool with actual SEE2 positives and legal hard negatives, preserving original full action teachers.',
            'Before another navigation training run, require event recall/false-positive measurements on a held-out FIT house or registered development partition, then matched full-denominator evaluation.'],
        not_supported=['Increasing memory length or model size as the next response to this evidence','Further early-STOP teacher truncation','Claiming a new algorithm, paper-ready gain or ordinary VLN-CE success'],
        unresolved=['DEV jointly holds out house and instruction template: this diagnosis does not isolate visual vs language shift.',
            '0.5 is descriptive only; no new runtime threshold is selected.',
            'No observed loss after a recognized event follows MONOTONIC structure, not a learned long-horizon generalization theorem.',
            'Correct-state action errors remain; event repair alone is not guaranteed to recover them.'],
        new_gpu_training_started=False,automatic_model_adoption=False)
    if support:
        next_step['existing_event_inventory']=dict(unique_input_role_labels=support['distinct_feature_role_labels'],physical_source_traces=support['source_traces'],new_data_collected=0,
            dev_word_room_combinations=len(support['dev_lexical_coverage']),dev_combinations_absent_from_fit=sum(not r['word_room_seen_in_fit'] for r in support['dev_lexical_coverage']),
            dev_words_absent_from_fit=sum(not r['word_seen_in_fit'] for r in support['dev_lexical_coverage']))
        next_step['immediate_next_work'][0]='Existing source audit and executable event index completed: EVENT_INDEX.json / EVENT_SUPPORT.json. Use this pool before assuming all difficult negatives are missing.'
    immutable(out/('NEXT_STEP_WITH_EVENT_AUDIT.json' if support else 'NEXT_STEP.json'),next_step)
    lines=['# 当前正向证据与 ORIGINAL 恢复定位','',
        '结论：存在真实、有限的开发集正向信号；尚无稳健的独立泛化或普通 VLN-CE 方法收益。保留 ORIGINAL/MONOTONIC，关闭 ALIGNED 教师截短修复。','',
        '|证据|实际结果|可以支持什么|','|---|---|---|',
        '|完整开发集对照|MONOTONIC 66/192，DIRECT 55/192；+5.73pp|受控 SEE2 任务上的候选正向信号|',
        '|历史无关控制|MONOTONIC 98/192，DIRECT 78/192；+10.42pp|同一开发协议下的控制任务改善|',
        '|四屋后续|主任务131/372 vs125/372；全计划每臂384，识别差值界[-1.56,+4.69]pp|弱信号，不能当稳健独立确认|',
        '|最新教师修复|768/768完成，ALIGNED47/192 vsORIGINAL66/192；−9.90pp|明确不采用该修复|','',
        '主任务与控制任务不能合并成更大的成功率；房屋内历史、变体及三个种子不是独立泛化样本。此前开发集正向不能抵消后续负结果，后续负结果也不能改写已发生的开发集改善。','',
        '## 本轮真实执行','',
        f"本轮在CPU读取三个冻结 ORIGINAL 模型，完成 {result['teacher_sequence_forwards']} 次完整教师序列前向、{probe['counts']['causal_step_forwards']} 次因果递归步骤，并复核192条已封存主任务轨迹。耗时{result['seconds']:.1f}秒；GPU小时、新Qwen前向、环境动作和优化器更新均为0。",
        f"三个模型参数前后不变。缓存CPU前向与现场接管状态最大差 {live['cached_cpu_live_state_max_delta']:.3g}；四位状态分类和动作翻转均为0。此处没有要求浮点逐位相等。",
        '三份重新训练的 ORIGINAL final 状态与此前 MONOTONIC 完全一致；主任务再次为66/192。这是复现性证据，不是增加三个独立统计种子。','',
        '## 新定位：先修事件识别，不能先认定记忆容量不足','',
        '|教师路径主任务诊断（模型×轨迹计数）|FIT|DEV|','|---|---:|---:|',
        f"|首次真实事件检出|{fit['first_witness_detected']}/{fit['witnessed_n']}|{check['first_witness_detected']}/{check['witnessed_n']}|",
        f"|真历史在接管点仍漏记|{fit['missed_history_at_cutoff']}/{fit['true_history_n']}|{check['missed_history_at_cutoff']}/{check['true_history_n']}|",
        f"|缺失历史却预测已经发生|{fit['false_history_at_cutoff']}/{fit['missing_history_n']}|{check['false_history_at_cutoff']}/{check['missing_history_n']}|",
        f"|检出后又丢失|{fit['detected_then_lost']}|{check['detected_then_lost']}|",'',
        'DEV的68例错误历史全部可追溯到事件预测累计；初始先验没有0.5以上的假阳性。没有观察到已识别事件随后丢失，与单调累计结构一致；不把这个结构性质当新学术贡献。','',
        '当前事件的去重输入诊断（仍有屋/族相关性）：','',
        '|区域|事件|正例检出|负例误报|','|---|---|---:|---:|']
    for row in probe['event_calibration']:
        if row['seed'] is None and row['scale']=='all':lines.append(f"|{row['split']}|{row['role']}|{row['tp']}/{row['tp']+row['fn']}|{row['fp']}/{row['fp']+row['tn']}|")
    lines+=['','DEV的清晰前置事件同样存在漏检（16/42检出）；因此不能仅归因于目标接近像素阈值。DEV房屋与语言模板同时留出，尚未分别识别视觉与语言迁移因素。','',
        '## 实际闭环的缺失历史96条','',f"成功{missing['passed']}/96；在尚无历史的错误信念下STOP有{missing['failure_types']['STOP_MISSING_HISTORY_WITH_FALSE_BELIEF']}条，在正确判断历史缺失时仍STOP有{missing['failure_types']['STOP_DESPITE_CORRECT_MISSING_HISTORY']}条。预算耗尽{missing['failure_types']['BUDGET_EXHAUSTED']}条。",
        '这些类型使用已观测终态分型；状态分类来自固定0.5描述阈值，未改变部署动作。某次访问中过早STOP、误报累计、运动偏离会相互影响，不据此声称单一原因已完成因果隔离。','',
        '## 下一步','',
        '转向任务条件事件识别的迁移与累计误报，优先审核/补充原FIT的真实正例及困难负例，保留原始完整动作教师。先验证当前输入下的事件检出和误报，再决定一次匹配闭环试验；本轮没有启动新训练。',
        '不继续缩短教师，不增加记忆槽或换更大底模。正确状态下仍存在动作错误，事件修复不保证解决全部失败；不能承诺CVPR录用或刷高普通VLN-CE分数。','',
        '具体证据见 CAUSAL_PROBE.json、ORIGINAL_RECOVERY.json、INPUT_MANIFEST.json、EVIDENCE_LEDGER.json 和 NEXT_STEP.json。原实验、权重、成功定义及旧UNKNOWN均未修改。']
    if support:
        lines+=['','## 已完成的事件样本准备','',
            f"已从{support['source_traces']}条既有真实物理轨迹核验并导出{support['distinct_feature_role_labels']}个去重输入×角色标签（EVENT_INDEX.json），不是新采集的独立样本。FIT/DEV仍分开。",
            '原FIT前置事件包含465个正例和2292个负例，负例中有717个单帧见证、375个低于像素阈值、13个跨帧不同实例。困难负例并非完全缺失；不能简单继续增加同类绕圈数据。',
            'DEV中的9组目标词×房间组合，有5组未在FIT出现，但目标词本身均出现过。跨组合、房屋及措辞变化同时存在；不能把错误只归为新词或数据总量。',
            '下一候选应先使用这个真实索引验证原FIT内的跨屋事件泛化与负例误报，再决定精确补充哪些组合/观察状态，保持完整动作教师。更新后的执行指向见NEXT_STEP_WITH_EVENT_AUDIT.json。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write(out/'STATUS.json',dict(status='COMPLETE',stage='report_complete',report=str((out/'REPORT_ZH.md').relative_to(LINE)),gpu_hours=0,new_optimizer_updates=0))

if __name__=='__main__':main(Path(sys.argv[1]))
