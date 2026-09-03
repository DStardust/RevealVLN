# SecondThought 方案合理性讨论提示词

下面的提示词可以直接复制给导师、合作者或具有联网能力的审稿型模型。建议同时
附上：

- [`METHOD_NOVELTY_REVISION_SECONDTHOUGHT_V1_ZH.md`](METHOD_NOVELTY_REVISION_SECONDTHOUGHT_V1_ZH.md)
- [`METHOD_NOVELTY_REVISION_SECONDTHOUGHT_V1.md`](METHOD_NOVELTY_REVISION_SECONDTHOUGHT_V1.md)

## 完整提示词

```text
请你以 CVPR 2027 embodied AI / vision-language navigation 领域的严格 Area
Chair 和方法学审稿人身份，评估下面的研究方案是否科学合理、是否还有足够创新
空间，以及下一项实验是否值得执行。

你的任务是找出最可能导致拒稿或实验失败的问题，而不是帮助我包装结果。请明确
区分：

1. 已由实验支持的事实；
2. 根据结果作出的推断；
3. 尚未验证的方法假设；
4. 只有获得进一步正结果后才能成立的论文主张。

如果可以联网，请优先检索并引用 2025--2026 年的一手论文、官方项目页或官方
代码，重点比较 AdaNav、AwareVLN、ATENA、StreamVLN、TagaVLM 和
OctoNav-R1。不要把“uncertainty-triggered reasoning”“Top-K reranking”或
“使用历史信息”本身误判为新颖贡献。

【研究背景】

基础模型是完全冻结的 ETP-R1。当前研究对象是 train-development、
raw-scene-disjoint 的 exact-rankable navigation decisions。模型训练和评测输入
禁止 future frame、outcome、shortest-path distance、scene ID 和 episode ID。

小规模固定 population 的结果：

- 1,428 exact-rankable decisions；
- Frozen ETP Top-1 accuracy = 82.56%；
- Top-3 accuracy = 99.72%；
- 249 个 Top-1 errors 中，245 个的 gold candidate 位于 Top-3；
- recoverable errors 覆盖 54/59 scenes；
- 使用 native Top-1 probability 识别 Top-1 wrong 的 AUROC = 0.803；
- 最高风险 20% decisions 捕获全部 Top-1 errors 的 47.4%。

已经尝试但失败的 selector/evidence：

- independent linear Top-3 scorer；
- pairwise one-hidden-layer comparator；
- candidate-facing current raw visual evidence；
- frozen ResNet feature；
- frozen DINOv2 feature；
- 简单的短期 temporal pooling。

扩充数据后：

- 6,446 episodes；
- 65,031 exact-rankable decisions；
- 59 raw scenes；
- 12,248 Frozen ETP Top-1 errors；
- 64,654 decisions 的 gold 位于 Top-3，即 99.42%。

目前唯一得到正号的是 relational Top-3 Transformer。正式 epoch-20
held-scene 结果为：

- triggered accuracy：55.170% -> 55.639%，+0.469 pp；
- global accuracy：81.166% -> 81.260%，+0.094 pp；
- wrong -> correct = 1,489；
- correct -> wrong = 1,428；
- net corrections = +61；
- recovery/new-error ratio = 1.043。

该结果没有通过预先固定的 +5 pp triggered、+1.5 pp global、ratio >= 1.5
门槛，因此正式结论是 FAIL，而不是成功方法。事后最好 epoch 11 也仅有
+0.953 pp triggered、+0.191 pp global、net +124、ratio 1.103，同样 FAIL。

此外，简单使用“Transformer 相对 ETP Top-1 的 score advantage”识别某次
干预是 rescue 还是 harm，AUROC 只有 0.538；最高风险 20% 中 benefit/harm
ratio 为 0.975。因此不能靠一个 score threshold 解决 intervention harm。

STOP 分支存在独立 oracle headroom：156 episodes 中 baseline SR 85.90%，
10 个 FALSE_STOP，termination oracle 恢复 10 个失败，oracle SR 92.31%。但是
tiny STOP verifier 只有 10 个 invalid STOP training examples，STOP_INVALID recall
为 0.60，因此该学习模块 FAIL，尚不能纳入有效方法。

【当前方法假设】

暂定名称：SecondThought: Risk-Controlled Proposal Deliberation for Frozen
Vision-Language Navigation Agents。

方法由以下部分组成：

1. Frozen base policy 输出 K 个 executable proposals 和 native logits；
2. 使用 frozen policy uncertainty 定位高风险 decisions；
3. 只对高风险 Top-3 运行小型 relational Transformer；
4. 再训练一个显式 counterfactual intervention utility gate，预测该 selector
   choice 会产生 RESCUE、NEUTRAL 还是 HARM；
5. 只有当

   P(RESCUE | x) - lambda * P(HARM | x) > tau

   时才覆盖 frozen base action，否则 KEEP；
6. Utility labels 必须来自 inner scene-disjoint out-of-fold selector
   predictions，outer held scenes 完全不参与 threshold、cost 或模型选择；
7. 将 ETP-specific implementation 抽象为 FrozenPolicyAdapter，使 proposal 可以
   是 topology waypoint，也可以是现代 VLA 的 executable action token；
8. 若 ETP 上通过，再依次验证 TagaVLM 和 StreamVLN。AwareVLN/AdaNav 主要作为
   同期竞争方法，OctoNav-R1 只作为高成本 extension；
9. Full rollout 只在 decision-level utility gate 得到实质正收益后进行。

单步 utility 定义为：

u = 1[reranker choice = gold] - 1[base choice = gold]，

即 u 属于 {-1, 0, +1}，分别表示 HARM、NEUTRAL、RESCUE。

【请重点回答】

1. 当前最合理的科学解释，究竟是“relational evidence 可学但 intervention
   control 缺失”，还是仅仅“出现了接近噪声水平的正号”？请给出支持与反对
   两方面证据。
2. Counterfactual utility gate 是否构成独立且合理的方法贡献，还是一个容易
   被审稿人视为普通 selective classification / learning-to-defer 的包装？
3. 使用 inner out-of-fold selector choice 构造 rescue/harm labels 是否足以避免
   leakage 与 circularity？还需要哪些严格控制？
4. 在 recoveries 和 harms 几乎相等、简单 confidence AUROC 只有 0.538 的条件
   下，utility 是否在统计上可能学习？请指出可识别性、class balance、scene
   shift 和 calibration 风险。
5. “FrozenPolicyAdapter 跨 waypoint proposals 和 action tokens”在数学与执行
   语义上是否真的统一？哪些部分必须 agent-specific，哪些部分必须共享，才能
   支持 cross-model generality，而不是只包装多个独立实现？
6. 对照 AdaNav、AwareVLN、ATENA 以及其他最新工作后，哪些 novelty claim 已被
   占据？最窄且仍可辩护的 novelty moat 是什么？
7. TagaVLM 和 StreamVLN 是否是合理的现代验证对象？请分别评价 checkpoint/
   code availability、proposal probability 可提取性、gold correspondence、
   adaptation cost 和实验解释力。如果不是，请提出更合适且有公开权重的
   2025--2026 模型。
8. STOP verification 是否能自然统一到 proposal intervention utility，还是会
   让论文变成贡献堆砌？
9. 请为“下一次、也是最后一次几小时到一天级 feasibility experiment”给出一个
   预先固定、可证伪的最小协议。只能选一项主实验，并明确 population、split、
   labels、features、model capacity、metrics 和 PASS/FAIL gate。不要建议
   architecture sweep 或根据 held-out 结果调 threshold。
10. 如果该实验失败，是否应停止整个 frozen Top-3 correction 方向？如果不应
    停止，请说明还有哪个关键假设尚未被排除；如果应停止，请给出明确停止理由。

【要求的回答格式】

A. 一句话总判定：
   - GO：方案合理，值得执行下一 gate；
   - CONDITIONAL GO：只有满足明确条件才值得继续；
   - STOP/PIVOT：现有证据不足以支持继续。

B. 事实 / 推断 / 未验证主张三列表。

C. Novelty collision 表：已有工作、已覆盖内容、SecondThought 尚可主张内容。

D. 方法学审查：label、split、leakage、calibration、统计功效、跨模型接口。

E. 最小下一实验：给出一套唯一协议和固定 PASS/FAIL gate。

F. CVPR 论文潜力：分别按 novelty、technical soundness、empirical strength、
   significance、reproducibility 以 1--5 分评分，并解释最低分项。

G. 三个最可能导致拒稿的致命问题，以及每个问题所需的最小证据。

请不要因为 Top-3 oracle headroom 很高就默认 learner 一定能实现收益，也不要把
post-hoc 最好 epoch 当成正式结果。若关键数字不足以作判断，请明确指出缺失的
artifact 或统计量，不要自行补造。
```

## 简短讨论版

```text
请作为严格的 CVPR VLN 审稿人，评估 SecondThought 是否值得继续：冻结导航策略
的 gold proposal 几乎总在 Top-3（99.42%--99.72%），Top-1 probability 能定位
错误（AUROC 0.803），但 linear、pairwise、raw-current、ResNet、DINOv2 和简单
temporal probes 均失败；扩充到 65,031 decisions 后，relational Transformer
仅将 global exact-candidate accuracy 从 81.166% 提至 81.260%，产生 1,489 次
修复和 1,428 次新增错误，正式 gate 仍 FAIL。简单 reranker advantage 对
rescue/harm 的 AUROC 只有 0.538。

下一假设是用 inner scene-disjoint out-of-fold predictions 构造
RESCUE/NEUTRAL/HARM labels，训练 counterfactual utility gate，并仅在
P(RESCUE)-lambda*P(HARM)>tau 时干预；随后通过统一 proposal interface 在
TagaVLM 和 StreamVLN 上验证。

请回答：这个解释是合理的可证伪假设，还是对接近噪声的结果继续包装？它相对
AdaNav/AwareVLN/ATENA 的最窄 novelty 是什么？utility gate 是否存在 circularity
或不可识别问题？请只给出一个最小下一实验及预先固定 gate，并明确实验失败时
是否应终止整个方向。结论必须是 GO、CONDITIONAL GO 或 STOP/PIVOT 之一。
```
