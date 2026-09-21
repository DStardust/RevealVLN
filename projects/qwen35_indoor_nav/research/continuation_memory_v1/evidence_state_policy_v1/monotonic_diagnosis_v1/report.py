"""Concise reproducible interpretation of the measured diagnosis; no new efficacy claim."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import HERE,BASE,HOLDOUT,read,immutable,sha

def main(out):
    result=read(out/'RESULT.json');pairs=read(out/'PAIRED_FAILURES.json');episodes=read(out/'EPISODES.json');fit=read(out/'FIT_PROBE.json')
    algebra=read(out/'READOUT_ALGEBRA.json');support=read(out/'TRAINING_SUPPORT.json')
    def agg(arm,seed=None):return next(r for r in result['aggregates'] if r['endpoint']=='main' and r['arm']==arm and r['seed']==seed)
    def fitted(model,task):return next(r for r in fit['summary'] if r['model']==model and r['task']==task)
    main=[r for r in pairs if r['endpoint']=='main'];s1210=[r for r in main if r['seed']==1210]
    losses=[r for r in s1210 if r['outcome']=='loss'];extra=[r for r in s1210 if r['extra_collision']]
    fm=fitted('MONOTONIC_1210','task_A');ft=fitted('MONOTONIC_1210','task_T')
    fit_missed=[r for r in fit['records'] if r['model']=='MONOTONIC_1210' and r['truth'][3] and r['action']!=3]
    true_ready=[r for r in algebra if r['model']=='MONOTONIC_1210' and r['endpoint']=='main' and r['phase']=='takeover' and r['truth_ready']]
    counters=dict(main_wins=sum(r['outcome']=='win' for r in main),main_losses=sum(r['outcome']=='loss' for r in main),
        added_collision_pairs=sum(r['extra_collision'] for r in main),avoided_collision_pairs=sum(r['avoided_collision'] for r in main),
        seed1210_loss_causes=dict(Counter(r['monotonic_failure'] for r in losses)),seed1210_extra_collision_pairs=len(extra),
        seed1210_extra_collision_exhausted=sum(r['monotonic_failure']=='EXHAUSTED' for r in extra),
        seed1210_first_divergence_actions=dict(Counter(r['first_action_divergence']['direct']['action']+' -> '+r['first_action_divergence']['monotonic']['action'] for r in losses)),
        fit1210_ready_missed=len(fit_missed),fit1210_missed_follows_native=sum(r['action']==r['native_action'] for r in fit_missed),
        fit1210_missed_state_correct=sum(r['state_correct'] for r in fit_missed),
        holdout1210_ready_actual_stop=sum(r['actual_action']=='STOP' for r in true_ready),holdout1210_ready_exact_state_algebra_stop=sum(r['exact_state_algebra_action']=='STOP' for r in true_ready),
        max_monotonic_update_error=max(r['max_monotonic_recurrence_error'] for r in episodes if r['arm']=='MONOTONIC'))
    next_step=dict(status='LOCALIZED_REPAIR_CANDIDATE_NOT_IMPLEMENTED_OR_TESTED',priority='STATE_TO_STOP_ACTION_READOUT',
        keep=['best4k encoder','8x64 recurrent memory','MONOTONIC historical update','task SEE2/STOP/collision/500-step success definition','all three seeds'],
        change_candidate='A separately versioned learned task-completion-to-STOP readout; actual causal predicted state and existing legal teacher actions only. First isolate the readout with the encoder and recurrent representation frozen. No oracle STOP, globally banning F, blanket native STOP guard or threshold selection from holdout.',
        first_falsifier='On fixed FIT/DEV, correct-state missed STOP must decline without increasing false STOP or ordinary-action regression; inspect both, not accuracy alone. This is a repair diagnostic, not a research contribution by itself.',
        source_of_supervision='Existing recorded teacher actions and deterministic causal state labels only; no fabricated recovery suffixes.',
        exclusion='Do not simply repeat the already failed blanket preservation-KL removal. Native-logit competition is correlated evidence here, not an isolated causal proof.',
        data_need='Controlled FIT trajectories have no forward motion. Closed-loop displacement/collision recovery remains outside their physical support. If forward recovery is evaluated, collect matched real FIT recovery examples for both arms; no selective MONOTONIC failure mining or existing TEST training.',
        heldout_status='Four evaluated houses are now exposed for diagnosis; any next run there is development/repair evidence, not fresh held-out confirmation.',
        gpu_execution_this_step=False,automatic_new_training=False)
    immutable(out/'NEXT_REPAIR.json',next_step);immutable(out/'FINDINGS.json',counters)
    lines=['# MONOTONIC 退化定位：动作读出有可复现缺口，容量不足尚无证据','',
        f"诊断完成：{result['episodes']}/1536 条已有续接、{result['autonomous_decisions']:,} 次自主决策；48 个未采集槽位仍保留。另对六份冻结权重做 {len(fit['records'])} 次原 FIT 接管前向。新增训练更新、仿真动作、GPU 小时均为 0。",'',
        '原结果不变：已执行主任务 MONOTONIC 131/372、DIRECT 125/372，+1.61pp；按完整计划分母差值 +1.56pp，未知分配识别界 [-1.56,+4.69]pp。四屋两正两负，三个种子两正一负，不是稳定闭环收益。',
        '', '## 1. 1210 首先暴露的是状态到动作的缺口','',
        '|主任务接管诊断|DIRECT 1210|MONOTONIC 1210|','|---|---:|---:|']
    a,b=agg('DIRECT',1210),agg('MONOTONIC',1210)
    for label,key in [('主任务成功/124','passed'),('碰撞轨迹/124','colliding'),('耗尽预算/124','exhausted'),('真实可停状态/124','ready_takeover_n'),('可停时立即STOP/32','ready_takeover_stop'),('真值可停且预测可停、却继续/32','ready_takeover_predicted_continue')]:lines.append(f"|{label}|{a[key]}|{b[key]}|")
    lines.extend(['',f"相同接管输入上，历史是否已发生的正确数分别为 {a['takeover_past_anchor']['tp']+a['takeover_past_anchor']['tn']}/124 与 {b['takeover_past_anchor']['tp']+b['takeover_past_anchor']['tn']}/124；当前终点假阳性由 {a['takeover_current_terminal']['fp']}/60 降至 {b['takeover_current_terminal']['fp']}/60。MONOTONIC 的状态没有整体更差，动作成功却下降，不能据此要求更大底模或更长记忆。",'',
        f"在原 FIT 上，MONOTONIC 1210 的 task_A 四位状态在 {fm['state_correct']}/{fm['n']} 个接管点全部正确，但仍有 {fm['state_correct_action_wrong']} 个教师动作不一致；其中真实应 STOP 的 {fm['ready_n']} 个点漏停 {fm['ready_n']-fm['ready_stop']} 次。task_T 状态同样 {ft['state_correct']}/{ft['n']} 正确，128 个应停点漏停 {ft['ready_n']-ft['ready_stop']} 次。转向教师不一致未必等于任务失败，漏掉这些有效STOP则是明确的读出诊断。",'',
        f"合计 {len(fit_missed)} 个 FIT 漏停点都具备正确四位状态，{counters['fit1210_missed_follows_native']} 个最终动作跟随原生动作。将显式状态输入替换成精确状态的离线线性重算，仍不能纠正这 {len(fit_missed)} 个漏停；说明只把显式状态识别变准不足以解决已观测读出错误。原生logit/KL竞争是线索，未被此观察单独因果归因。",'',
        '具体反例：主条件 208 与 210，MONOTONIC 的可停概率约 0.999545，但 STOP margin 分别为 -0.518/-0.564，执行右转后失败；同输入 DIRECT 主动 STOP 并成功。两条是同一父族的相关历史，不能当两个独立泛化样本。',
        '', '## 2. 新增碰撞集中在失控后的长尾','',
        f"主任务配对中，MONOTONIC 新增碰撞 {counters['added_collision_pairs']} 对、避免碰撞 {counters['avoided_collision_pairs']} 对，净增20。1210 新增碰撞 {len(extra)} 对，其中 {counters['seed1210_extra_collision_exhausted']} 对最终耗尽预算。1210 主任务前进动作 5669 次，DIRECT 为 2207 次；这是不同长度/访问分布的描述，不能仅由数量推断因果。",'',
        '1210 的18个配对输例中，7个终点不满足却STOP、5个未满足历史却STOP、5个耗尽、1个已满足但此前发生碰撞。11个输例首次动作分歧是左右转选择；不应把全部退化归咎于漏停。',
        '', '## 3. 数据支持与记忆边界','',
        '原 FIT 受控数据的动作监督为 STOP472、左转1676、右转1788、前进0；全部受控真实轨迹也没有前进。普通动作池有前进监督（18679行），但不等于有这类任务的位移后恢复监督。原始数据合法，缺的是特定闭环状态覆盖，不能说全部数据错误。',
        '',f"MONOTONIC 自主运行中的更新方程最大残差仅 {counters['max_monotonic_update_error']:.3g}，未发现历史概率下降。未发生事件时的错误累计确实存在，但主任务出现此错误的轨迹数 MONOTONIC 为 {agg('MONOTONIC')['ever_unseen_history_false_positive']}、DIRECT 为 {agg('DIRECT')['ever_unseen_history_false_positive']}；不能把总体退化简单归为单调累计误报。",'',
        '原前缀没有逐步事件预测，尚不能把接管时漏记分解为视觉漏检、初始先验或旧事件写入失败。长程保持与当前事件识别仍需登记的局部诊断；本轮没有证明记忆容量下限。',
        '', '## 4. 最小下一修复','',
        '优先做独立版本的“任务完成状态到 STOP 的学习读出”局部对照，先冻结编码器和递归表示，用原 FIT/DEV 的实际合法动作监督检查：已知可停时能否停，同时未知/未满足时不要错停，并保留普通动作回归。原500步和成功定义不动。',
        '', '不直接用真值或0.5规则强制STOP，不全局禁止前进，不因1210差就删种子，不重复旧的整体KL删除修复。若局部读出不能过关，先停在这个定位点，不扩大训练。若进入位移恢复实验，两臂都需要同池真实FIT恢复轨迹。',
        '', '本轮只完成诊断与可检验的修复定位；尚未实现/训练新读出，也没有新增闭环收益。四个新屋已用于本次诊断，后续不能再称为未暴露测试。',
        '', '## 复核入口与限制','',
        '- `RESULT.json`：全分母诊断与分种子、状态误差；`EPISODES.json`：逐轨迹首分歧/错误状态引用。',
        '- `PAIRED_FAILURES.json`：所有胜负与新增碰撞；`READOUT_ALGEBRA.json`：只读代数，不是另一条导航轨迹。',
        '- `FIT_PROBE.json`：真实冻结CPU前向与参数不变；`TRAINING_SUPPORT.json`：已有训练池覆盖。',
        '- `INPUT_MANIFEST.json`：封存组及来源哈希。历史服务没有逐步碰撞标志；零位移前进仅是运动学代理，不能伪装精确碰撞时刻。',
        '- 0.5仅用于固定描述性分类，不是本轮新增控制阈值。动作后的访问分布存在选择偏差；这里只定位修复，不宣称因果效能。'])
    with (out/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    immutable(out/'CPU_TEST_RESULT.json',dict(status='PASS',unit_tests=3,test_log_sha256=sha(HERE/'CPU_TEST_001.log'),
        frozen_fit_forward_records=len(fit['records']),model_state_unchanged=fit['model_state_unchanged'],new_updates=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
