# RevealNav Frozen Specification

**冻结版本：2026-08-22 / Method-Freeze-2**  
**目标：CVPR 2027 或同等级 CCF-A 会议**  
**状态：方法、主张、数据协议和投稿门槛冻结，进入公开数据可行性扫描与实施**  
**研究判断：截至冻结日的一手文献审计未发现同时覆盖 evolving candidate set、Reveal–Expiry timing 与 pre-error option preservation 的工作；该组合具有足以进入 CVPR/CCF-A 实施与验证阶段的可辩护 novelty。此判断不是接收或正分保证，竞争力只有通过第 9 节全部实证门槛后才视为成立。**

## 1. 最终题目与一句话贡献

暂定标题：

> **RevealNav: Learning When to Commit under Evolving Candidate Sets in Vision-and-Language Navigation**

核心问题：

> 在有限视场连续 VLN 中，正确分支可能尚未成为可执行候选，识别它所需的语言证据也可能尚未闭合；智能体需要在分支显现之前继续获取证据，同时避免越过最后安全选择机会。

最终方法由且仅由三部分组成：

1. **Reveal–Expiry Estimator (REE)**：预测正确候选/决定性证据何时充分显现，以及当前选择权何时到期；
2. **Evidence-Contingent Option Graph (ECOG)**：只保存能够保留未来选择权的安全 checkpoint；
3. **Option Preservation Policy (OPP)**：在统一任务损失下选择 FOLLOW、INSPECT、EXPLORE、BACKTRACK、COMMIT 或 STOP。

U/A/D 是 REE 的可解释状态读出，不再作为独立三分类创新。动作 entropy 只作为 baseline。

## 2. 2026-08-22 新颖性复核后的定位

### 2.1 直接近邻与不可再主张的内容

| 近邻工作 | 已覆盖内容 | 本文必须保持的边界 |
|---|---|---|
| CondVLN, 2026-08 | 显式 if/then/otherwise 指令、证据可观测性、分支一致性诊断 | 本文研究普通路线指令下候选随 ego-FOV 出现的时间与 last-safe deadline，不把 conditional branching 当贡献 |
| GC-VLN, CoRL 2025 | 指令图约束、导航树、无解/多解时回退 | 约束解析只是 REE 的输入；贡献是证据显现与选择权到期的在线预测 |
| ProFocus, CVPR 2026 | 主动获取缺失视觉信息、从历史候选中选 top-k waypoint | 本文必须证明“继续观察还来不来得及”和预错误 option preservation，而非一般主动感知 |
| AwareVLN / Progress-Think, CVPR 2026 | 关键节点推理、任务进度和单调进展 | 本文预测的是 candidate/evidence reveal time 与 option expiry time，而非一般 instruction progress |
| FAST / Topological Planning / SSM / DUET | 拓扑记忆、全局候选、规划和回退 | ECOG 不是通用地图；节点存在的唯一理由是保留一个尚未闭合的选择权 |
| SmartWay, IROS 2025 | 增强 waypoint proposal、历史推理和出错后 backtracking | 不把 waypoint 或 backtracking 本身作为贡献；本文评价错误发生前是否以及何时保存选择权 |
| ETP-R1, 2025-12 | 强图式 VLN-CE、拓扑规划、R2R/RxR 联合预训练和在线 RFT | 仅作为冻结 backbone；不把图式规划或 RFT 作为本文贡献 |
| Beyond Waypoints, 2026-06 | 将候选 waypoint 绑定到可执行轨迹，改善 reachability 和 planning-control 一致性 | 候选/控制前端固定且共享；本文不主张候选生成或低层控制创新 |
| AdaNav | 动作熵触发额外推理 | 低熵不代表候选完整；entropy 不进入主决策定义 |
| CoRe, 2026-08 | 操作任务失败后的 counterfactual recovery | 本文只研究错误发生前的 option preservation，不使用“counterfactual recovery”作为方法命名 |
| SAS-MDP 文献 | 随机/变化动作集合的通用 MDP | 不宣称首次提出 changing-action-set MDP；只主张其在有限视场 VLN 中的可观测性与决策时机问题 |

### 2.2 冻结后的 novelty moat

本文只保留三个审稿时可逐项验证的差异：

1. **Evolving candidate set**：正确路线分支可能在当前视觉候选集合中不存在，而不是已经给出后做逻辑选择。
2. **Reveal–Expiry timing**：同时学习“何时首次足以承诺”和“何时将失去安全选择机会”，而不是只估计进度、置信度或缺失信息。
3. **Pre-error option preservation**：checkpoint 在错误发生前保存尚未兑现的选择权，其价值由是否降低未来任务损失定义。

若 2026 年后续工作同时覆盖以上三点，则必须重新做 novelty 审计；仅出现新的拓扑、回退、条件分支或 uncertainty 工作不触发换题。

## 3. 冻结的问题定义

令 \(h_t=(I,O_{\le t},\widehat{\mathcal B}_{\le t})\) 表示完整指令、严格截断的视觉历史和声明候选前端产生的候选历史。

对一个路线分支事件定义两个时间变量：

### 3.1 Reveal time

\[
T_R=\min\{t:\ b^*\in\widehat{\mathcal B}_t,
\ b^*\text{ 可与竞争候选区分，且决定性语言证据闭合}\}.
\]

实际标注使用区间和连续 \(K\) 个 prefix 的稳定条件，不要求主观地指定唯一边界帧。

### 3.2 Expiry time

\[
T_X=\max\{t:\ \text{仍存在能够进入目标分支或返回已保存候选的安全控制序列}\}.
\]

\(T_X\) 是 last-safe opportunity，不等于路口中心或目标分支首次可见时刻。

### 3.3 U/A/D 读出

- U：目标不在声明候选集合中；
- A：目标已在集合中，但竞争消歧或决定性语言证据至少一项未闭合；
- D：目标、竞争消歧和所有决定性语言证据均稳定成立。

`UNRESOLVABLE_BEFORE_SPLIT` 表示 \(T_R>T_X\) 或在传感器协议下不存在可观测的 \(T_R\)。这类事件单独评价安全 fallback，不强迫模型进入 D。

## 4. Reveal–Expiry Estimator

### 4.1 输入

- 完整 instruction；
- 截止到 \(t\) 的 RGB history；
- Oracle 或 Frozen frontend 的当前/历史候选；
- ECOG 检索出的至多 \(M\) 个相关 checkpoint token；
- 不向模型提供未来帧、目标真值、navmesh 或 simulator pose。

### 4.2 可监督输出

\[
p_t^{\mathrm{set}},\qquad
p_t^{\mathrm{sep}},\qquad
e_{t,k}=P(c_k\text{ resolved}\mid h_t),
\]

以及离散时间 cause-specific hazards：

\[
h_t^R=P(T_R=t\mid T_R\ge t,h_t),
\qquad
h_t^X=P(T_X=t\mid T_X\ge t,h_t).
\]

指令约束 \(c_k\) 只覆盖会改变分支身份的 direction、ordinal、exclusion、temporal 和 landmark relation。解析器可复用现有方法或冻结 VLM，不作为贡献。

U/A/D 由 set、separation 和 evidence outputs 单调导出。训练使用：

- target-in-set BCE；
- candidate separation ranking loss；
- constraint closure BCE；
- interval-censored reveal/expiry likelihood；
- 时间单调正则，只约束接近事件且未发生遮挡回退的有效区间。

第一实现固定为 **ETP-R1 官方公开 checkpoint**，只增加小型 temporal heads；不重新进行大规模预训练或在线 RFT，避免算力和数据规模掩盖方法贡献。选择 ETP-R1 是因为它已经提供 R2R-CE/RxR-CE、候选 waypoint、拓扑状态和低层控制的统一接口，而非因为其拓扑本身构成本文贡献。若截至 2026-08-31 无法在现有 Habitat/VLN-CE 环境中严格复现该 checkpoint，则触发 no-go，不静默更换 backbone。

## 5. Evidence-Contingent Option Graph

ECOG 不保存每一帧。冻结的候选前端先提出持续 \(K\) 帧可匹配的分支事件，REE 再判断是否保存最后安全位姿。

每个节点只记录：

- checkpoint id 与可由公共控制器回访的引用；
- 代表 RGB/history embedding；
- instruction phase 与未闭合证据；
- 候选分支 id、代表观测和 `UNTRIED/ACTIVE/EXHAUSTED/COMMITTED` 状态；
- 创建时的 reveal/expiry prediction。

节点不存环境真值，不自行建立新的 SLAM。Oracle return controller 与 Frozen return controller 分开报告，且所有方法共享同一个底层控制器。

## 6. Option Preservation Policy

定义统一任务损失：

\[
\mathcal L=
\lambda_w\mathbf 1[\text{wrong commitment}]
+\lambda_m\mathbf 1[\text{missed opportunity}]
+\lambda_d C_{\mathrm{path}}
+\lambda_t C_{\mathrm{time}}
+\lambda_r C_{\mathrm{return}}.
\]

对动作 \(a\in\{\text{FOLLOW, INSPECT, EXPLORE}(b),\text{BACKTRACK}(v),\text{COMMIT}(b),\text{STOP}\}\)，预测：

\[
Q_\theta(h_t,G_t,a)=\mathbb E[\mathcal L\mid h_t,G_t,a].
\]

候选 checkpoint \(v\) 的 Option Preservation Value 为：

\[
\operatorname{OPV}_t(v)=
\min_a Q_\theta(h_t,G_t\setminus v,a)
-
\min_a Q_\theta(h_t,G_t\cup v,a).
\]

当分支事件稳定且 \(\operatorname{OPV}_t(v)>\tau_{cp}\) 时创建节点。运行时在安全动作中选择最小预测损失：

- 未达到 D 且 \(h^X\) 低：FOLLOW/INSPECT；
- 某分支预计降低未来损失且可返回：EXPLORE；
- 当前分支风险高于历史未尝试候选：BACKTRACK；
- D 成立且 COMMIT 风险最低：COMMIT；
- 目标/物品确认且 STOP 风险最低：STOP；
- \(T_R>T_X\)：报告 unresolved，并执行预先声明的安全 fallback。

OPV/Q labels 由 Habitat 中的 bounded counterfactual rollout 生成；模型输入不包含 rollout 真值。训练固定为归一化 cost-to-go 的 Huber regression，并用同一 batch 内 checkpoint/action pair 的 margin ranking 作为辅助损失；OPV 始终由预测 Q 的差值导出。消融只比较去掉 ranking loss 和无 OPV 的启发式版本，不在主实验中事后选择学习目标。

## 7. RevealBench-CE

### 7.1 数据范围

- **RxR-CE-en**：主 benchmark；
- **R2R-CE**：跨指令风格验证；
- **CondVLN**：若官方代码与协议可复现，仅作为邻近任务的零样本/迁移诊断，不与其争夺 conditional branching benchmark 主张；
- 双浦、户外与机器狗在公开数据主结论成立后再作为域外验证。

### 7.2 样本单位

一个 Reveal Event 包含：

- dataset/scene/episode/event id；
- `candidate_frontend_id`；
- 严格截断 RGB prefixes；
- 每个 prefix 的候选 id；
- 固定 target branch 与决定性语言约束；
- \(T_R\) 区间、唯一 \(T_X\)、U/A/D、resolvability；
- oracle/Frozen candidate provenance；
- 可复现的 counterfactual action costs。

同一路线及其反事实必须处于同一 scene split。协议冻结前不查看 val-unseen 的事件统计或 baseline 结果。

### 7.3 决定性反事实

1. 相同当前帧，仅历史是否看过前置入口不同；
2. 相同历史和当前帧，指令在 first/second 或 before/after 间最小替换；
3. 低熵但 target-in-set=false；
4. target-in-set=true 但 decisive evidence=false；
5. 相同 reveal state，不同 \(T_X-t\)；
6. 移除 ECOG 节点后最优恢复成本显著增加。

## 8. 固定实验矩阵

### 8.1 候选协议

1. Oracle Current Candidates：隔离核心假设；
2. Frozen Automatic Frontend：固定为 Hong et al. (CVPR 2022) 的 VLN-CE candidate waypoint predictor 及 ETP-R1 既有低层控制，冻结 waypoint proposal、traversability 和 temporal matching；REE 只接收候选结果，不接收该前端的 depth、panorama 或 simulator feature；
3. 所有方法共享 observation、candidate frontend、return controller、速度和动作预算。

### 8.2 必须比较的基线

- Unmodified ETP-R1 policy；
- AwareVLN official checkpoint，作为不同传感协议下的现代性能参照，不进入匹配输入的因果比较；
- max-softmax / action entropy；
- target-visible head；
- history-aware direct U/A/D；
- progress-aware baseline；
- ProFocus-style historical candidate ranking；
- branch memory without expiry；
- REE without ECOG；
- ECOG with intersection-only checkpoint；
- Full REE + ECOG + OPP。

### 8.3 指标

表示层：U/A/D macro-F1、NLL、ECE、Brier、\(T_R/T_X\) interval coverage 与 onset error。  
事件层：False-Ready Rate、Premature Commitment Rate、Missed Opportunity Rate、Accidental Correct Commitment、Risk–Delay AUC。  
恢复层：Backtrack Success、Return Distance、Repeated-Branch Rate、checkpoint count。  
完整导航：SR、SPL、nDTW、SDTW、NE、wall-clock、VLM calls。

所有结果报告至少 3 个种子、paired episode bootstrap 95% CI，并在匹配 compute、delay 或 risk 的 operating point 比较。

## 9. 投稿硬门槛

以下任一项不满足，不以当前主张投稿 CVPR：

1. pilot 中至少 300 个可解非平凡事件，人工有效率至少 25%；
2. U/A/D Fleiss' \(\kappa\ge0.65\)，evidence closure \(\kappa\ge0.70\)；
3. 最终至少 2,000 个 scene-disjoint Reveal Events，Gold test 至少 600 个三人标注事件；
4. 相比最强 history-aware baseline，在匹配 delay 下 PCR 相对降低至少 25%，95% CI 不跨零；
5. RxR-CE-en/R2R-CE 与 Oracle/Frozen 四个设置效果方向一致；
6. 标准 SR/SPL 在两个数据集均统计非劣，并至少一个设置显著提升；
7. REE、expiry supervision、ECOG/OPV 均有独立消融贡献；
8. 五名未参与项目的 VLN 内部审稿人全部给出 weak accept 或以上，且没有未回答的 novelty objection。

## 10. 可行性与实施顺序

截至 2026-08-22，CVPR 2027 官方只公布会议将在 Seattle 举行，尚未公布论文截止日。项目采用更早的内部截止，避免依赖未确认日期。

### Phase 0：2026-08-22 至 2026-08-31

- 获取官方 RxR/R2R-VLNCE metadata 与现有 Habitat 资产；
- 跑英语指令高召回筛选和 50 条轨迹人工复核；
- 验证 candidate evolution、\(T_R\)、\(T_X\) 可自动生成；
- 输出 event count、人工有效率和不可解比例。

**8 月 31 日 go/no-go：** 若无可用 Habitat/MP3D 资产、ETP-R1 checkpoint 无法严格复现、预计不足 300 个有效事件，或 \(T_X\) 不能稳定定义，则停止 CVPR 2027 时间表，不用加模块掩盖。

### Phase 1：2026-09-01 至 2026-09-20

- 冻结 benchmark schema、split 和候选协议；
- 建成数据生成器、事件级 validator 和 entropy/visible/history baselines；
- 完成 300-event 三人 pilot 标注。

### Phase 2：2026-09-21 至 2026-10-15

- 接入冻结的 ETP-R1 checkpoint；
- 实现 REE temporal heads、ECOG 和 OPP；
- 跑 Oracle/Frozen 主实验与核心消融。

### Phase 3：2026-10-16 至 2026-10-31

- 统计检验、失败分解、普通路线非退化实验；
- 写作、图表和第一次五人盲审；
- 完成投稿前最新论文检索。

若官方截止晚于内部截止，额外时间只用于复现、审稿意见和稳定性，不新增主模块。

## 11. 固定实现边界

第一阶段只实现：

```text
official episodes -> prefix/event generator -> event validator
                  -> frozen VLN encoder -> REE heads
                  -> ECOG memory -> OPP/Q head -> evaluation
```

不在第一阶段实现：新 SLAM、新候选检测器、联合训练大 VLM、户外数据、真实机器狗、多语言扩展、开放词汇物品搜索。

当前 `toporeveal/` 代码是接口探索原型，不是冻结方法实现。实施时保留可复用类型和测试，替换手工 utility，并先修复已审计出的安全候选过滤、不可达 checkpoint、事件级 schema 和 Enum 输入边界。

## 12. 冻结决策登记表

| 决策项 | 冻结选择 |
|---|---|
| 主任务 | 普通路线指令中的 evolving candidate set 与 commit timing |
| 主数据 | RxR-CE-en；R2R-CE 只做跨风格迁移 |
| Backbone | ETP-R1 官方 checkpoint；冻结大编码器与原有低层控制 |
| 自动候选前端 | Hong et al. (CVPR 2022) candidate waypoint predictor |
| 核心状态 | target-in-set、candidate separation、constraint closure、reveal/expiry hazards |
| 时间稳定条件 | 连续 \(K=3\) 个 prefix；遮挡回退区间不施加单调约束 |
| 图记忆 | 保存全部 ECOG 节点，REE 每步最多检索 \(M=8\) 个节点 |
| checkpoint 触发 | 稳定分支事件且预测 \(\operatorname{OPV}>\tau_{cp}\) |
| 分支/动作选择 | 安全可执行集合内最小预测 cost-to-go；entropy 只作 baseline |
| Q/OPV 学习 | Huber cost regression + pairwise margin auxiliary；OPV 为 Q 差值 |
| 调参与锁定 | 仅在 scene-disjoint train/val-seen 上选择阈值和损失权重，协议冻结后禁止查看 Gold test 统计 |
| 第一阶段明确不做 | 新 SLAM、新候选器、大 VLM 联训、真实机器人、户外、多语言、开放词汇物搜 |

## 13. 冻结与变更规则

从 Method-Freeze-2 开始，主问题、三模块、U/A/D 语义、两个公开主数据集、两种候选协议、固定实验矩阵和投稿硬门槛不再变化。实现可以补全代码、修复错误并在已声明的 train/val-seen 范围内选择数值超参数，但不得改变论文因果链或把 baseline 升格为贡献。

只允许三类变更：

1. 2026 年新论文直接同时覆盖 evolving candidate set、Reveal–Expiry timing 和 pre-error option preservation；
2. Phase 0/1 的硬数据门槛失败；
3. 实现或形式化存在可复现错误。

触发上述任一情况时，必须新建版本号并记录变更理由；不得直接覆盖 Method-Freeze-2 的主张。其他改进全部进入 backlog，不得在主线中继续堆模块。
