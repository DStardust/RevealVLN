# RevealNav v1.3：公开室内数据上的可恢复分支承诺（历史方案）

> 本文已被 [FROZEN_SPEC.md](FROZEN_SPEC.md) 取代。2026-08-22 新公开的 CondVLN 等工作改变了 novelty 边界；实施与投稿只以冻结规格为准。

## 1. 结论先行

本项目不能承诺论文一定被 CVPR 或其他 CCF-A 会议接收，也不能在实验前保证方法新颖性和有效性。能够做到的是设置一套可审计的“学术风险防火墙”：已有工作覆盖的能力不宣称为贡献；主张只建立在统一复现和可证伪实验上；公开数据阶段通过后才扩展到自采数据和真实机器狗。

当前最稳妥的论文命题不是“提出拓扑 VLN”或“让智能体学会回退”。这两点已有大量先例。建议保留 RevealNav 名称，将主命题收紧为：

> 在有限视场下，分支候选及其语言身份所需证据会逐步显现。导航智能体应在不可逆选择前估计证据是否闭合，并用可恢复的 option checkpoint 保留选择权。

最终方法只保留三个有明确职责的部分：`Factorized Reveal Estimator`、`Evidence-Contingent Option Graph (ECOG)` 和统一的 `Counterfactual Recoverability Value (CRV) Policy`。

拓扑图是承载“选择权”的系统机制，不是单独贡献。论文的因果链必须是：

```text
候选/证据未显现
        ↓
普通动作置信度产生 false-ready
        ↓
Reveal belief 判断是否可承诺
        ↓
option checkpoint 保存可恢复选择
        ↓
更低错误承诺，或相同风险下更低延迟
```

### CVPR 硬门槛

“首轮全部正分”只能作为内部投稿门槛，不能作为外部结果保证。当前方案在公开实验完成前尚未达到该门槛。只有同时满足以下条件才提交 CVPR：

1. 相比最强 history-aware baseline，在匹配承诺延迟下将 premature commitment 相对降低至少 25%，且 paired bootstrap 的 95% 置信区间不跨零；
2. 在 RxR-CE 与 R2R-CE、Oracle 与 Frozen 两种候选前端上方向一致；标准 SR/SPL 在两个数据集均统计非劣，并至少在一个数据集上显著改善；
3. factorized Reveal、counterfactual checkpoint 和 Option Graph 三项消融均有独立贡献，不能由更大模型、更多历史帧或更多推理次数解释；
4. 五名未参与项目、熟悉 VLN 的内部审稿人盲审均给出 weak accept 或以上，且没有未回答的 novelty objection。

## 2. 唯一主研究问题

给定指令 \(I\)、截至时刻 \(t\) 的允许观测 \(O_{\le t}\)、当前候选集 \(B_t\) 和在线 option graph \(G_t\)，学习策略：

\[
\pi(a_t, c_t, r_t \mid I,O_{\le t},B_t,G_t),
\]

其中：

- \(a_t\)：FOLLOW、INSPECT、EXPLORE、BACKTRACK、ENTER、STOP；
- \(c_t\)：是否对某个语义分支作不可逆承诺；
- \(r_t\)：是否把当前决策位置提升为可回访 checkpoint。

核心评价不是单一 SR，而是错误承诺、承诺延迟、错过最后安全机会和回退成本的联合 Pareto。

## 3. 创新边界与禁止性表述

### 可以主张

1. 将 false-ready 分解为 `Candidate Absence` 与 `Referential Evidence Absence`，并用严格截断前缀验证两者是否独立存在。
2. 用可因子分解的 Set–Separation–Constraint Reveal estimator 预测“何时可承诺”，U/A/D 由这些可验证事件导出，而非独立黑盒分类。
3. 用统一的 Counterfactual Recoverability Value 决定 checkpoint 创建、分支探索、回退和承诺，在同一决策损失下保存选择权。
4. 提供针对上述问题的公开室内逐前缀 benchmark、指标和强基线。

### 不应主张

1. “首次在 VLN 中使用拓扑图、历史节点或回退”。
2. “首次主动观察缺失信息”或“首次从历史候选选择分支”。
3. “低熵意味着安全承诺”之外的宽泛 uncertainty 结论。
4. 未经统一输入、候选前端和控制器复现就断言某篇方法必然失败。

### 与最邻近方法的最小差异

| 方法族 | 已覆盖能力 | RevealNav 必须额外证明的部分 |
|---|---|---|
| FAST / Topological Planning / SSM / DUET | 图搜索、全局候选、回退、拓扑规划 | 回退节点为何在证据未闭合时有反事实价值 |
| AdaNav | 动作熵触发额外推理 | 低熵但 target 不在集合或证据未闭合 |
| ProFocus | 主动获取缺失视觉信息、历史 waypoint 搜索 | 最后安全机会与可恢复承诺，而非一般主动感知 |
| AwareVLN | 关键节点的场景/进度/纠错推理 | candidate/evidence closure 的显式监督和校准 |
| DRIVE-Nav | persistent directions、inspection、verification | 同方向多个语义入口及历史序数证据 |
| Meta-Explore / Lookahead | 候选探索、未来路径评估 | 不可逆承诺风险和可审计 evidence state |

## 4. 状态与标签定义

候选集必须相对于一个声明的前端定义，不能由模型置信度反推。每个结果都分别报告 Oracle Current Candidates 与 Frozen Automatic Candidate Frontend。

- U：目标分支不在该前端当前输出的候选集合。
- A：目标在集合中，但竞争消歧或决定性语言约束至少一项未闭合。
- D：目标在集合中、竞争候选可区分、所有决定性语言约束均可由允许历史验证。

`referential_evidence_complete` 是 A/D 的辅助监督，不增加第四状态。环境真值只用于生成 target membership、分支拓扑和 last-safe label，不进入模型输入。

不可解样本标为 `UNRESOLVABLE_BEFORE_SPLIT`，不强迫进入 D，也不混入标准可解事件的可靠 grounding 成功率。

### 防止标签泄漏

- 标注员只能看截止到 \(t\) 的图像历史、完整指令和前端候选，不得看未来帧。
- `target_in_set` 由声明的候选前端与路线真值计算，不由标注员猜测。
- 人工只标“该历史是否足以验证关系约束”和感知可辨性。
- train/val/test 按 scene 划分；同一路线的反事实版本必须在同一 split。
- 每个标注显式记录 `candidate_frontend_id`；同一事件的 target、候选前端和决定性约束不可跨 prefix 改变。
- 每个事件只能有一个 last-safe-prefix；D onset 定义为首次连续 (K) 个 prefix 保持 D 的起点，避免逐帧抖动改变指标。
- 发布派生 annotation 与 episode id，不重新分发受 Matterport3D 条款约束的图像或 mesh。

## 5. Factorized Reveal：避免 U/A/D 只是一个新分类头

先将指令解析成决定性约束集合 \(\mathcal C(I)\)，只保留会改变分支身份的 direction、ordinal、exclusion、temporal 和 landmark-relation 约束。模型分别预测：

\[
p_t^{\mathrm{set}},\qquad
p_t^{\mathrm{sep}},\qquad
e_{t,k}=P(c_k\ \text{resolved}\mid O_{\le t}),\ c_k\in\mathcal C(I).
\]

再通过单调 evidence aggregator \(g_\theta\) 导出：

\[
q_t(U)=1-p_t^{\mathrm{set}},
\]

\[
q_t(D)=p_t^{\mathrm{set}}p_t^{\mathrm{sep}}
g_\theta(\{e_{t,k}\}),
\qquad
q_t(A)=1-q_t(U)-q_t(D).
\]

`g_\theta` 对每项决定性证据单调，并在任一必需约束未闭合时抑制 D。这样 U/A/D 有明确语义和可审计误差来源；直接三分类 U/A/D 只作为消融。

## 6. 统一指标：Counterfactual Recoverability Value

不能直接用 action entropy 创建 checkpoint。低熵可能来自不完整候选集，而且走廊中的感知噪声会产生大量无用节点。

定义统一决策损失：

\[
\mathcal L=
\lambda_w\mathbf 1[\text{wrong commit}]
+\lambda_m\mathbf 1[\text{miss last-safe opportunity}]
+\lambda_d C_{\mathrm{path}}
+\lambda_t C_{\mathrm{time}}.
\]

用 simulator counterfactual rollout 监督动作 cost-to-go：

\[
Q_\theta(h_t,G_t,a)=
\mathbb E[\mathcal L\mid h_t,G_t,a],
\]

其中动作 \(a\) 统一包括 INSPECT、EXPLORE 某分支、BACKTRACK 某节点、COMMIT 和 STOP。运行时选择满足安全约束的最小 \(Q_\theta\) 动作，而不是组合多个手工分数。

对候选 checkpoint \(v\)，定义反事实可恢复价值：

\[
\operatorname{CRV}_t(v)=
\min_a Q_\theta(h_t,G_t\setminus v,a)
-
\min_a Q_\theta(h_t,G_t\cup v,a).
\]

冻结的拓扑前端只负责提出稳定分支事件；当至少两个分支持续可匹配且 \(\operatorname{CRV}_t(v)>\tau_{cp}\) 时才保存节点。节点存储最后安全位姿、当前 instruction phase、未闭合约束、候选分支及 `UNTRIED/ACTIVE/EXHAUSTED/COMMITTED` 状态。

这个同一指标也回答“探索哪条分支”：

\[
b_t^*=\arg\min_{b\in\mathcal B_t^{\mathrm{safe}}}
Q_\theta(h_t,G_t,\operatorname{EXPLORE}(b)).
\]

因此不再争论概率、熵或手工 information gain 哪个最好：概率、证据闭合、路径成本和不可逆风险共同作为 \(Q_\theta\) 的输入与监督来源，最终比较的是该动作将减少多少预期任务损失。

只有同时满足以下条件才 `COMMIT`：

\[
q_t(D)>\tau_D,\quad
e_t^{ref}>\tau_e,\quad
p_{target}(b)>\tau_b.
\]

否则只能进行可恢复的 EXPLORE/INSPECT。当前分支的最小风险高于某个历史 checkpoint 时，执行 BACKTRACK；目标/物品被确认且 STOP 的预测风险最低时才停止。

## 7. 第一阶段公开 benchmark

### 数据范围

1. **RxR-CE-en：主数据。** 第一阶段明确使用英语子集；其指令更长、语言更密集，适合筛选序数、排除、时序与地标后关系。
2. **R2R-CE：交叉验证。** 检验方法不是只适配 RxR 的长指令风格，并作为较成熟连续 VLN 基线入口。
3. **REVERIE/SOON：暂不作为主证据。** 它们适合后续物品搜索扩展，但离散全景设置会削弱有限视场 candidate absence 的核心假设。
4. **双浦与机器狗：公开阶段通过后再启动。** 不进入首轮方法选择和超参数调节。

### Phase A：可行性扫描，不训练大模型

Phase A 先限定英语指令，解析公开 train 中预先划出的 scene-disjoint pilot/calibration split，筛选：

- ordinal：first/second/next/another；
- exclusion：skip/pass/not the first；
- temporal：after/before/once；
- landmark-conditioned branch：after the wall/stairs/door；
- 同方向连续分支和有限视场渐显事件。

然后在 Habitat 中沿 ground-truth continuous path 重放，生成固定 FOV 前缀、候选集合和 last-safe-prefix。先抽样 300 个事件进行三人盲标。协议冻结前不查看 val-unseen 的事件统计或 baseline 结果。

### Phase A 的硬门槛

- pilot split 中可解的非平凡事件不少于 300，或自动筛选后人工有效率至少 25%。
- U/A/D Fleiss' \(\kappa\ge0.65\)，evidence-complete \(\kappa\ge0.70\)。
- `UNRESOLVABLE_BEFORE_SPLIT` 不高于 pilot 候选事件的 20%。
- 强 baseline 至少出现足够的 false-ready：低熵子集的 premature commitment rate 不低于 10%。

Pilot 通过后，最终目标是构建不少于 2,000 个 scene-disjoint Reveal Event，并保留不少于 600 个三人标注的 Gold test events；若公开数据无法支持该规模，不把 benchmark 作为 CVPR 主贡献。

未过门槛时不堆模块：优先 pivot 到“可恢复承诺/last-safe decision”，弱化 ordinal grounding；若公开数据根本缺事件，则停止宣称新 benchmark。

## 8. 指标

### 表示层

- U/A/D macro-F1、NLL、ECE、Brier score；
- target-in-set AUROC/AUPRC；
- evidence-completeness F1/ECE；
- D onset MAE 与 interval coverage；
- false-ready rate，分别报告 Candidate Absence 与 Referential Evidence Absence。

### 决策层

- Premature Commitment Rate (PCR)；
- Missed Opportunity Rate (MOR)；
- Accidental Correct Commitment (ACC)；
- Commitment Delay (CD)，只在可解事件上计算；
- Backtrack Success / Backtrack Distance / repeated-branch rate；
- Risk-Delay AUC 与固定风险/固定延迟 operating points。

### 完整导航

保留 SR、SPL、nDTW、SDTW、NE，并增加每条路线 checkpoint 数、内存、推理调用次数和 wall-clock latency。普通无歧义路线单独报告，防止策略靠普遍减速换取风险下降。

## 9. 决定性实验与消融

1. **严格同当前帧反事实。** 仅替换历史中“是否看过第一个入口”，验证模型是否真正使用历史证据。
2. **低熵 false-ready。** 比较 max-softmax、entropy、energy/no-match、target-visible、history-aware evidence closure。
3. **checkpoint 价值。** Every-N-step、distance-based、intersection-only、entropy-gated、完整 CRV gate。
4. **分支调度。** target probability、entropy、information gain、完整 counterfactual \(Q_\theta\)。
5. **恢复闭环。** no memory、stack-only、topological option graph；相同候选前端和底层控制器。
6. **Oracle→Frozen frontend gap。** 收益若主要被候选召回率吞噬，应如实归因为 perception bottleneck。
7. **普通路线非退化。** 相同 SR 下检查额外路径、等待和推理成本。

所有阈值只在 val-seen/held-out calibration split 选择；val-unseen 不反复调参。至少报告 3 个随机种子、置信区间和 paired episode bootstrap。

## 10. 实施路线

### 现有探索原型（不作为最终方法实现）

- `toporeveal/types.py`：U/A/D belief、branch、checkpoint proposal 和动作接口。
- `toporeveal/memory.py`：可回访 checkpoint、分支状态和最短回退路径。
- `toporeveal/policy.py`：反事实价值门控、risk-aware utility、commit/explore/inspect/backtrack/stop 状态机。
- `toporeveal/benchmark.py`：逐前缀标签的可执行一致性约束。
- `toporeveal/screening.py`：对官方 RxR guide metadata 做高召回语言事件筛选。
- `tests/test_toporeveal.py`：false-ready、证据缺失、回退、STOP 和标签防泄漏测试。

该原型早于 CRV 方案，只验证接口可行性；最终实现必须改为 factorized Reveal 与 counterfactual \(Q_\theta\)，不能把当前手工 utility 当作论文方法。

### 下一阶段

1. 接入官方 VLN-CE/Habitat 数据读取与轨迹 replay，仅生成 metadata，不改基线训练代码。
2. 完成指令事件筛选器和 300-event annotation manifest。
3. 复现 entropy、target-visible 和 history-aware 三个最小 baseline。
4. 数据门槛通过后，接入 AwareVLN/NaVILA 类 backbone 的多任务 heads。
5. 最后才做自动候选、弱标注扩量和真实机器狗。

## 11. 投稿决策门

满足以下三点，才把目标维持在 CVPR/CCF-A 主会：

1. 在统一候选前端下，false-ready 是稳定且规模足够的独立失败模式；
2. factorized Reveal 相比直接 U/A/D 与 history-aware 强 baseline 有显著增益；
3. CRV Option Graph 形成统计显著的 Risk-Delay Pareto 改善，同时普通路线不退化。

若只满足第一点，贡献更像 benchmark/analysis；若只在 Oracle candidates 下成立，应先解决候选前端或降低系统性主张；若拓扑回退带来收益但 reveal supervision 无增益，则应转为规划论文，不能继续使用当前 novelty 叙事。
