# SecondThought：当前方法、实验证据与研究方向

**Revision：** `secondthought_method_and_results_v1_zh`
**日期：** 2026-09-03
**状态：** 探索性 novelty revision；不是冻结投稿协议
**暂定标题：** *SecondThought: Risk-Controlled Proposal Deliberation for Frozen Vision-Language Navigation Agents*

**英文版本：** [`METHOD_NOVELTY_REVISION_SECONDTHOUGHT_V1.md`](METHOD_NOVELTY_REVISION_SECONDTHOUGHT_V1.md)

## 0. 与冻结项目的关系

本文档记录一条独立的 candidate-correction 研究线，不修改、不取代、
也不重新解释 [`FROZEN_SPEC.md`](FROZEN_SPEC.md)。其中 Method-Freeze-2
已经冻结的主张、模块、数据集、协议与门槛继续保持 immutable。

本文总结的 candidate-correction 实验 artifact 当前不在这个自包含仓库中。
下文的精确数字来自已经报告的实验记录，但尚未根据当前仓库内的 artifact
重新推导。任何结果在用于投稿前，都必须在本项目内使用从权威来源独立获取的
资产重新复现，并补齐冻结 protocol、immutable result、代码哈希、checkpoint
provenance 和 raw-scene-disjoint split manifest。

## 1. 核心结论

现有实验形成了一条范围较窄、但逻辑连贯的证据链：

1. Frozen ETP-R1 通常已经生成了正确的局部候选：在 exact-rankable
   decisions 中，gold candidate 进入 Top-3 的比例约为 99.4%--99.7%。
2. ETP 自身的 score distribution 能识别大量高风险决策：使用
   Top-1 probability 区分 Top-1 wrong/correct 的 AUROC 为 0.803，最高风险
   20% decisions 能捕获全部 Top-1 errors 的 47.4%。
3. 因此，主要问题不是 candidate generation，而是在少量合理候选中稳定选对。
4. Independent linear scoring、pairwise comparison、当前帧 raw evidence、
   ResNet、冻结 DINOv2 以及简单短历史 pooling 都没有产生可靠的正向纠错收益。
5. 在扩充数据上训练的 relational Top-3 Transformer，是目前唯一把结果从负向
   推到微弱正向的 selector，但正式 held-scene 收益仍远低于预先封存的门槛。
6. 当前决定性瓶颈是 intervention harm：selector 能修复许多 baseline errors，
   但也会破坏数量几乎相同的原本正确决策。

因此，当前方法假设不能只表述为“对 uncertain candidates 重新排序”，而应是：

> 只有当 relational deliberator 不仅提出了替代候选，而且具有经过校准的证据，
> 表明本次干预更可能修复错误而不是破坏正确决策时，才允许覆盖 frozen
> navigation policy。

## 2. 问题定义

在 timestep \(t\)，冻结导航策略给出 proposal set
\(\mathcal C_t=\{c_1,\ldots,c_K\}\)、native scores \(s_t\) 和 baseline
decision：

\[
b_t=\arg\max_{c_i\in\mathcal C_t}s_t(c_i).
\]

纠错系统只能读取 deployment-time 可用信息：

- 当前 observation；
- instruction；
- 当前 proposal representation 与 native score；
- proposal geometry 或 action semantics；
- 过去的 frozen policy state 与已经执行的 action；
- 禁止 future frame、outcome、shortest-path distance、scene ID 和 episode ID。

系统输出 `KEEP` 或 replacement proposal \(r_t\)。科学目标不是孤立的
reranker accuracy，而是净纠错收益：

\[
u_t=
\mathbf 1[r_t=y_t]-\mathbf 1[b_t=y_t]
\in\{-1,0,+1\},
\]

其中 \(y_t\) 只能作为训练或评测监督：

- \(+1\)：rescue，即 wrong \(\rightarrow\) correct；
- \(0\)：neutral intervention；
- \(-1\)：harm，即 correct \(\rightarrow\) wrong。

只有满足下式时才执行干预：

\[
P(u_t=+1\mid x_t)-\lambda P(u_t=-1\mid x_t)>\tau,
\]

其中 \(\lambda\) 表示破坏一个原本正确动作的代价，\(\tau\) 必须在不使用
held-scene 信息的情况下确定。

## 3. 当前方法

### 3.1 冻结 proposal policy

基础策略完全冻结。已完成的 probes 使用 ETP-R1 作为 discovery backbone，
不更新其 candidate generator 或 checkpoint。

### 3.2 Uncertainty trigger

当前 trigger 使用较低的 native Top-1 probability。在五折
raw-scene-disjoint evaluation 中，每一折只根据 training scenes 确定固定的
20% training-risk threshold，然后原样应用到 held-out scenes。未触发的
decision 必须完全保留基础策略动作。

### 3.3 Relational Top-3 deliberator

只有 triggered decisions 会被重新考虑。目前最强的 selector 是作用于三个
最高分 proposal 的小型 Transformer。它可使用的证据包括：

- 冻结 instruction representation 与 navigation/history representation；
- 冻结 candidate/proposal representation；
- 可获得时的 frozen visual evidence；
- native ETP score；
- relative heading 和 relative distance；
- proposal mask 以及 Top-3 之间的关系比较。

所有 proposal position 共享同一个 scorer，其输出作为对 native ranking 的
residual correction。所有 frozen-policy encoders 均保持不变。关系建模很重要：
独立为每个 candidate 打分的模型没有表现出相同的数据扩展趋势。

### 3.4 待验证的 counterfactual utility gate

下一项方法组件是一个使用 out-of-fold proposal outcome 训练的独立
intervention gate。它预测 `RESCUE`、`NEUTRAL`、`HARM`，或者等价地估计上文的
expected utility。它不负责寻找新候选，而只负责决定是相信 deliberator，
还是保留 frozen policy 的原始决策。

为防止 leakage，一个样本的 utility label 必须来自没有在该样本或其 raw scene
上训练过的 selector。所有 threshold 与 cost 必须在 nested training folds
内固定，然后才能进行 held-scene evaluation。

### 3.5 通用 proposal interface

实现需要从 ETP-specific tensor width 中解耦，改成统一的
`FrozenPolicyAdapter`：

```text
instruction evidence
current observation evidence
last K policy/history states
K executable proposals
native proposal logits
proposal geometry or action semantics
STOP proposal, when available
proposal-to-environment executor
```

对于 graph navigation agent，proposal 是 waypoint 或 topology node；对于现代
VLA agent，proposal 是 executable action token。允许每个 agent 使用不同的
input projection，但 uncertainty trigger、relational deliberator、utility
objective 与 evaluation protocol 应当保持共享。

## 4. 已完成的实验证据

### 4.1 STOP / termination 分支

#### Oracle headroom：PASS

Revision：`stop_oracle_headroom_v1`

| 指标 | 结果 |
|---|---:|
| Train-development episodes | 156 |
| Baseline success / failure | 134 / 22 |
| Baseline SR | 85.90% |
| FALSE_STOP | 10 |
| MISSED_STOP | 1 |
| PURE_NAVIGATION_FAILURE | 11 |
| Oracle-recovered successes | 10 |
| Oracle SR | 92.31% |
| Absolute SR headroom | +6.41 percentage points |
| Baseline failures 中可恢复比例 | 45.45% |

结果：`STOP_ORACLE_HEADROOM_PASS`。Termination error 具有明确的 oracle value，
并且主要来自 false STOP，而不是 missed STOP。

#### Tiny STOP veto：FAIL

Revision：`tiny_stop_veto_v1`

收集到 137 个 STOP proposals，其中 127 个 valid、10 个 invalid。
Held-scene STOP-invalid recall 为 0.60，低于固定的 0.70 gate。由于 invalid
样本过少，无法支持可靠的 scene-disjoint verifier，因此该分支没有继续声称
rollout improvement。

结果：`TINY_STOP_VETO_FAIL`。Oracle opportunity 真实存在，但第一个监督式
verifier 尚未验证成功。

### 4.2 Candidate availability 与 reranking headroom

Revision：`candidate_rerank_headroom_v1`

| 指标 | 结果 |
|---|---:|
| Exact-rankable decisions | 1,428 |
| Top-1 correct / wrong | 1,179 / 249 |
| Top-1 accuracy | 82.56% |
| Top-3 accuracy | 99.72% |
| Top-1 errors 中 gold 位于 Top-3 | 245 / 249 |
| Top-3-recoverable error rate | 98.39% |
| Recoverable errors 覆盖 scenes | 54 / 59 |

结果：`CANDIDATE_RERANK_HEADROOM_PASS`。在这个 population 中，candidate
generator 不是主要瓶颈；只要能够在现有 Top-3 内正确选择，几乎所有 Top-1
error 都有局部恢复空间。

### 4.3 已有 secondary signals

Revision：`candidate_secondary_signal_probe_v1`

Frozen instruction-candidate compatibility 与可用 local-score signals 在固定的
245-error population 上均未达到预先封存的 65% gold-preference 门槛。

结果：`SECONDARY_SIGNAL_PROBE_FAIL`。现有 scalar signal 无法支持
zero-training reranker。

### 4.4 Candidate uncertainty

Revision：`candidate_uncertainty_headroom_v1`

在固定的 1,428-decision population 上，native Top-1 probability 是最好的
单一 uncertainty signal：

- wrong-versus-correct AUROC：**0.803**；
- 最高风险 20% 捕获了全部 Top-1 errors 的 **47.4%**。

结果：`CANDIDATE_UNCERTAINTY_HEADROOM_PASS`。Frozen policy 经常知道“什么时候
危险”，但 uncertainty 本身并不能指出哪个替代候选正确。

### 4.5 Selective correction probes

所有 probes 都遵循相同的五折 raw-scene-disjoint 原则，并且只修改 triggered
Top-3 decisions。

| Probe | Evidence / structure | 结果 | 科学含义 |
|---|---|---|---|
| Independent linear Top-3 | Frozen candidate features、score、geometry | FAIL | 独立 candidate separability 不足 |
| Pairwise MLP | 显式 candidate-vs-candidate differences | FAIL | 单纯 pairwise structure 不能解决问题 |
| Raw current evidence MLP | Candidate-facing 当前视觉证据 | FAIL | 只重新读取当前 observation 不足 |
| Frozen ResNet arm | 替代冻结视觉特征 | Held-scene 无正收益 | 常规 backbone replacement 不足 |
| Frozen DINOv2 arm | 更强的冻结自监督视觉特征 | Held-scene 无正收益 | 更强静态视觉 representation 仍不足 |
| Simple temporal pooling | 短期 frozen history | FAIL | 简单平均短历史不足 |
| Relational Top-3 Transformer | 使用 frozen evidence 联合比较 Top-3 | 微弱正向，但 gate FAIL | Relational reasoning 可学习，但还不安全 |

当前仓库没有 ResNet 与 DINOv2 的本地 result artifact，因此本文不复述其精确
per-fold 数值。目前能够严格保留的结论只限于：它们没有产生 held-scene 正向
收益。

### 4.6 扩充数据后的 relational Transformer

扩充 collection 后的数据规模为：

| 指标 | 结果 |
|---|---:|
| Episodes | 6,446 |
| Exact-rankable decisions | 65,031 |
| Raw scenes | 59 |
| Frozen ETP Top-1 errors | 12,248 |
| Gold 位于 Top-3 的 decisions | 64,654 |
| Gold-in-Top-3 rate | 99.42% |

正式 epoch-20 held-scene 结果：

| 指标 | Frozen ETP | Selective Transformer | 变化 |
|---|---:|---:|---:|
| Triggered exact-candidate accuracy | 55.170% | 55.639% | +0.469 pp |
| Global exact-candidate accuracy | 81.166% | 81.260% | +0.094 pp |
| Wrong \(\rightarrow\) correct | -- | 1,489 | -- |
| Correct \(\rightarrow\) wrong | -- | 1,428 | -- |
| Net corrections | -- | +61 | -- |
| Recovery/new-error ratio | -- | 1.043 | -- |

该结果没有达到预先声明的门槛：triggered gain 至少 +5 pp、global gain 至少
+1.5 pp，并且 recovery/new-error ratio 至少 1.5。

结论：正式模型为 **FAIL**，但收益符号为正。

仅用于诊断，观测到的最好 checkpoint 是 epoch 11：

- triggered gain：+0.953 pp；
- global gain：+0.191 pp；
- net corrections：+124；
- recovery/new-error ratio：1.103。

这个 post-hoc checkpoint 同样未通过任何实质性 improvement gate，不能作为
经过选择的 primary result 对外报告。

### 4.7 Scalar intervention-confidence diagnostic

由 Transformer 相对于 native ETP Top-1 的 advantage 构造的 scalar，无法区分
beneficial intervention 与 harmful intervention：

- benefit-versus-harm AUROC：0.538；
- 在选出的最高风险 20% 中，benefit/harm ratio：0.975；
- net corrections：-2。

因此，直接对 reranker advantage 设置 threshold 不能形成有效 safety gate。
下一步 utility gate 必须显式学习 intervention outcome，并且必须进行
out-of-fold evaluation。

## 5. 实验已经证明和没有证明的内容

### 5.1 已支持的结论

1. **Proposal coverage 很高。** 在 1,428-decision 和扩充 population 中，
   几乎所有 gold action 都已位于 Top-3。
2. **可以定位高风险决策。** Native probability 对 wrong/correct 具有较强的
   区分能力。
3. **静态视觉特征强度不是核心缺口。** 从 ResNet 换到 DINOv2 并没有让
   candidate correction 变成正向。
4. **Relational structure 有作用。** 数据规模上升后，Top-3 Transformer 是目前
   唯一稳定得到正号的模型族。
5. **Safe intervention 是主要瓶颈。** Recoveries 与 newly introduced errors
   的数量仍然接近。

### 5.2 尚不支持的结论

当前证据不能证明：

- selective correction 能提高完整 navigation rollout 的 SR 或 SPL；
- 方法能泛化到 ETP-R1 之外；
- proposed utility gate 能有效区分 rescue 与 harm；
- 继续调节 DINOv2 或其他 backbone 就能解决问题；
- tiny STOP verifier 已经可以部署；
- 当前 Transformer 的微弱正收益足以形成 CVPR-level empirical claim。

## 6. 修正后的创新定位

“由 uncertainty 触发第二阶段 reasoning”已经不能作为独立 novelty。
[AdaNav](https://www.microsoft.com/en-us/research/publication/adanav-adaptive-reasoning-with-uncertainty-for-vision-language-navigation/)
已经使用 action uncertainty 触发 adaptive reasoning；
[AwareVLN](https://github.com/GWxuan/AwareVLN) 会在关键节点于稀疏的
`[REASON]`/`[ACT]` 模式之间切换；[ATENA](https://github.com/kuai-lab/NeurIPS25_att_vln)
则在 DUET 和 ETPNav backbone 上进行 uncertainty-aware test-time adaptation。

更可辩护的方法假设是以下组合：

1. **Frozen-policy post-hoc correction**：不重新训练基础 agent；
2. **Proposal-space abstraction**：统一 graph node 与 VLA action token；
3. **Counterfactual intervention utility**：显式区分 rescue 与 harm，而不是只预测
   action correctness；
4. **Risk-controlled abstention**：当 expected intervention value 不为正时保留
   native action；
5. 如果 STOP 分支以后获得足够 invalid-example support，可以进一步形成
   **统一 MOVE/STOP verification**。

现代 agent evaluation 是泛化证据，而不是可以分别计数的多个独立贡献。

## 7. 现代 Agent 验证计划

DUET/ScaleVLN 一类较旧 agent 可以保留为受控 legacy baselines，但不应作为
2027 投稿的 headline evidence。

| 定位 | Agent | 相关性 | 接口与成本判断 |
|---|---|---|---|
| 工程桥接 | [TagaVLM，ICRA 2026](https://github.com/APEX-BJUT/Taga-VLM) | Qwen2-0.5B/7B、公开权重、topology-aware global action | 最接近当前 Top-K proposal correction；中等成本，但仍由 DUET 派生 |
| 现代主目标 | [StreamVLN，ICRA 2026](https://github.com/InternRobotics/StreamVLN) | LLaVA-Video + SlowFast streaming context，支持 R2R/RxR 并提供 checkpoint | 最适合检验 temporal correction，需要 action-token adapter |
| 直接竞品/压力测试 | [AwareVLN，CVPR 2026](https://github.com/GWxuan/AwareVLN) | Llama-3 8B + SigLIP，显式 sparse reasoning/action switching | 必须比较，但其自身已包含 selective reasoning，不是中性 base |
| Generalist extension | [OctoNav-R1，CVPR 2026](https://github.com/buaa-colalab/OctoNav-R1) | 连续环境中输出低层 action sequence 的 VLA policy | 泛化范围最广，集成成本最高 |

建议顺序：

1. 先在 TagaVLM 上运行 zero-training headroom audit，低成本验证通用 proposal
   adapter。
2. 将 StreamVLN 作为主要现代跨 agent 实验。
3. 将 AwareVLN/AdaNav 作为同期 selective-reasoning competitor 比较。
4. 只有前两个 base 获得实质正结果后，才接入 OctoNav-R1。

每个新 agent 在训练前都先进行有界的 train-development audit：

- deterministic replay 与 executable proposal correspondence；
- exact gold proposal/action availability；
- Top-1/Top-3 coverage；
- uncertainty AUROC；
- recoverable-error count 与 raw-scene coverage。

只有 native proposal space 具有足够 recoverable headroom 的 agent，才允许进入
成本更高的 correction experiment。

## 8. 下一项必要实验

下一个 feasibility gate 应验证：explicit intervention utility 能否在已经收集的
expanded population 上减少 harm。

### 8.1 固定比较

```text
A. Frozen ETP-R1
B. Uncertainty-triggered relational Transformer
C. B + out-of-fold counterfactual utility gate
```

### 8.2 必须满足的 leakage control

对于每个 outer held-scene fold：

1. proposal selector 只能在 outer-training scenes 上训练；
2. training scenes 内的 selector choice 必须通过 inner scene-disjoint
   out-of-fold prediction 生成；
3. 根据这些 out-of-fold choices 产生 rescue/neutral/harm label；
4. 使用这些 labels 训练 utility gate；
5. 在 outer held scenes 上评测前固定 operating point。

### 8.3 主要指标

- triggered/global exact-candidate accuracy；
- wrong \(\rightarrow\) correct；
- correct \(\rightarrow\) wrong；
- net corrections；
- recovery/new-error ratio；
- intervention coverage；
- scene-by-scene degradation；
- rescue/harm probability 的 expected calibration error。

Utility gate 没有实质改善 recovery/new-error trade-off 之前，不应开始 full
rollout 或现代 agent port。如果显式 out-of-fold utility prediction 仍然失败，
应停止 lightweight frozen-policy correction 方向，而不是换用更大的 Transformer
继续 rescue。

## 9. 条件成立时的论文贡献结构

如果 utility gate 与现代 agent experiments 均通过，论文可以形成三个连贯贡献：

1. **Diagnosis：** 大规模揭示 frozen VLN agent 的 proposal-coverage/risk-control
   paradox：正确替代候选通常已经存在，错误也可以被定位，但朴素纠错会造成
   数量接近的新增错误。
2. **Method：** 使用 counterfactual rescue-versus-harm utility 优化的
   risk-controlled relational proposal deliberator。
3. **Generality：** 一个统一的 correction abstraction，覆盖 waypoint-candidate
   policy、streaming VLM action policy，并可能扩展到 STOP decision。

以上贡献仍然是条件式目标。当前 Transformer 的正号只能证明存在一定
learnability，尚不能证明已经得到成功的 navigation method。
