# Q35N G0：贡献边界审查 V1

日期：2026-09-09。顺序机器审查；没有第二位独立评审、人工审核或新导航实验。

## 结论

- 原泛化表述“受观测约束的教师 → 示范 → SFT”：不能作为原创算法；已有基础明确，不再保留这种首创主张。
- [冻结的具体成对示范算子](../CONTRIBUTION_BOUNDARY_V1.md)：`PASS_FOR_SPECIFIED_CLAIM`，含义仅为值得进入数据可行性检验的狭义候选。
- 总体 G0：`REFORMULATE`，剩余缺口是物理数据接口与标签证书；不是继续找一个更漂亮的名字。训练和方法实验未准入。
- 科学支持、稳定泛化、贡献充分、SOTA/近 SOTA：均未验证。本报告没有新增正收益。

## A. Failure Scientist：只从现有证据判断

| 假设 | 证据 | 状态 | 最便宜的判别检查 |
|---|---|---|---|
| 限制教师可用信息有理论/实证先例 | DESPOT、GPO、EgoPush 的实际方法 | Supported，指问题有依据，不指本候选有效 | 不重复证明这一宽泛命题 |
| 普通全知专家示范在所有 VLN 情形都有害 | 无；指令可已经消歧，BC 也可学条件动作分布 | Refuted，否定的是“所有”这一表述 | 保留普通 SFT 及已消歧任务对照 |
| 本项目已有可靠像素语义真值可直接造数据 | 前序 B15 诊断存在全零语义和对象身份歧义 | Unobservable，针对已查资产接口 | 验收对象身份、可见性和成功条件；不能从 RGB 观感补真值 |
| 真实室内任务可以规模化产生反转样本 | 目前只有数学构造，没有本线采样数据 | Untested | 一个合法世界族的恢复、渲染与完整有限求解检查 |
| 成对提取优于同规划器均匀示范 | 无测量 | Untested | 同求解预算、同训练预算的匹配消融 |
| 从 Qwen 训练就能避免历史失败并达到强结果 | 无；旧组件失败不能推出该结论 | Untested | 普通 Qwen SFT 独立场景基线，随后同 backbone 消融 |

历史失败不在本版本重新命名为成功；旧数据暴露不因创建独立目录而消失。

## B. Idea Compressor：删除非必要内容

| 选项 | 标签/实现负担 | 新颖性与潜在收益判断 | 处理 |
|---|---|---|---|
| 观测约束规划器直接生成普通示范 | 仍需合法世界和规划，网络简单 | 重要基线；没有足够具体增量 | 必需对照，不作主贡献 |
| 预算 token + 多预算训练 | 容易实现，普通轨迹即可 | AdaTurn 等已明确训练预算边界；单独贡献不足 | 仅接口/对照 |
| 回报证书支持的反转成对示范 + 无用观察反例 | 生成端较难，策略端保持普通 SFT | 与已核算法具体操作不同；收益未知，必须超过均匀教师 | 唯一候选 |

本轮不引入新地图、时序标签体系、额外记忆网络、长 CoT 或 RL。小模型不是创新，完整系统也不靠模块数量定义厚度。

## C. 最近方法：公式与代码拆件

下表是相关方法范围的审查，不声称完整复现其全部仓库。来源在行内。

| 工作 | 已核验的实际操作 | 与冻结算子的具体关系 |
|---|---|---|
| [DESPOT](https://arxiv.org/abs/1609.03250)，[despot.cpp](https://github.com/AdaCompNUS/despot/blob/37a3e2175eabe6d351914b6c00e10ec29f462bb5/src/solver/despot.cpp) | 完整相关源文件：采样 belief particles；同动作推进多场景；Expand 使用 partitions[obs] 分组；Backup 汇总分支及动作界限 | 提供已知耦合规划核心；该文件没有训练数据配对选择。不是因为“在线改离线”而新；均匀离线蒸馏是最强简单对照 |
| [GPO](https://arxiv.org/abs/2505.15418)，[GPO.py](https://github.com/liyheng/GPO/blob/d7a955ddff9ccb7e447f74e78a177791a1abcc9d/GPO.py) | 完整文件：同一 PyTorch policy 用于 full/partial observation；双向 detached KL、裁剪/惩罚方式，guider 与 learner surrogate，训练 full rollout、评估 partial rollout | 已处理可模仿教师与信息不匹配；不能简化成“只做 KL”或称两个独立网络。未在该文件见物理同前缀族与预算反转示范编译 |
| [CAST](https://arxiv.org/abs/2508.13446)，[counterfactual.py](https://github.com/catglossop/CAST/blob/ec7a214e76167e0f844800ea91d6664863a3d9b1/cast/data/utils/counterfactual.py)、[action_generation.py](https://github.com/catglossop/CAST/blob/ec7a214e76167e0f844800ea91d6664863a3d9b1/cast/data/utils/action_generation.py) | 完整两文件：当前帧/原子历史与候选指令生成不同指令和原子动作，再以原子 diffusion policy 生成 waypoint chunk | 反事实动作数据已有；不是不可区分隐藏世界的共享决策与回报反转证书。不能说 CAST 只是文字增强 |
| [CFNBC](https://arxiv.org/abs/2607.27261) | 正文方法：任务保持 nuisance 对，专家动作不变；名义策略 action drift/response feature；加权 facility-location 贪心覆盖；原标签 BC 修复 | “反事实 + 同预算选数据”已有。其预算是修复样本量，非导航剩余动作；其目标动作保持不变。我们的差异必须是任务回报支持的标签反转，不能只换筛选名词。未审计其代码 |
| [AdaTurn](https://arxiv.org/abs/2607.14547) | 方法正文：预算显式输入、随机 rollout budgets、FA-DAPO 将超预算工具调用转为可训练的最终回答、调度负载平衡 | 预算条件主动视觉与边界训练已有；本候选不宣称首次预算感知，也不把换成导航当差异。具体增量是成对示范构造而非预算输入 |
| [EgoPush](https://arxiv.org/abs/2602.18071) | 正文：可见性限制稀疏 keypoint teacher，蒸馏到 egocentric depth student，阶段预算及时间衰减奖励 | “教师只看可见信息会教出主动感知”已有；不把该问题或 teacher constraint 本身列为原创 |
| [LSP-AIG](https://arxiv.org/abs/2403.03269) | 正文 Eq.3–4：frontier 代价加入 VOI；用已知地图评价观测前后 base-LSP 决策差，累积后训练 GNN 属性预测 | 自动导航信息价值标签已有；本候选比较有限耦合树回报并构造训练对，不宣称首次 VOI 标签 |
| [Ask When It Pays / TANDEM](https://arxiv.org/abs/2606.03175) | 正文：实例目标歧义下，cost-aware 问 oracle，EIG 代理与问题类型成本 | “何时值得获得信息”的故事已有；本候选任务须由物理观测可解，运行期没有问 oracle |
| [CCC-VLN](https://arxiv.org/abs/2203.16586) | 正文：特征混合 counterfactual creator 保持 instruction/action pair，speaker/follower cycle consistency | 场景反事实训练已有；实际方法不是物理重排真实世界，不能只按摘要归类 |
| [GLiDE](https://arxiv.org/abs/2403.17124)，[代码](https://github.com/yanweiw/glide) | augment_demo.py：平滑 Gaussian 轨迹扰动，task.validify_traj 分成功/失败示范池 | 成功失败增强已有；不是共享观测的多世界决策反转编译 |

DESPOT 的深度叶上界钳制是本候选的具体实现风险：有限搜索返回 gap=0 不保证全任务最优。
源码经 alphaXiv 按默认分支阅读；随后经 GitHub ls-remote 固定提交并独立获取文件 SHA256。没有将两次来源响应逐字节比对，因此不声称 MCP 响应已通过哈希同一性证明。固定源登记见 [SOURCE_AUDIT_V1.json](SOURCE_AUDIT_V1.json)。

VisualThink-VLA、EgoCoT-Bench、PRODEN 的前序实现/数据审查在 [来源登记](../REFERENCES.md)；本轮不把前序读过的内容伪称重新完整复现。
Undermind 用于公共概念近邻检索，alphaXiv 用于论文相关正文和具体代码，GitHub 固定版本取证。SciSpace 未可用、未使用；拆件表由本次顺序审查完成。不上传私有草稿或轨迹。

## D. Adversarial Reviewer：最危险的否定解释

1. **“这只是 POMDP 规划蒸馏。”** 部分成立：教师核心确实继承。只有成对编译超过同一教师均匀采样，具体训练算法才有独立贡献。
2. **“这只是预算条件策略。”** AdaTurn 已有，不能作为新意；同预算下两种证据价值情形必须被区别。
3. **“这只是 curriculum / hard-example mining。”** 这是最强贡献风险。必须隔离任务回报反转证书相对随机、entropy、动作漂移选样的收益，并计入生成成本。
4. **“你构造了对自己有利的小谜题。”** 受控对只能做机制诊断，主结论必须来自独立自然任务分布。模板反转比例不是自然事件频率。
5. **“相同画面给相反标签根本学不了。”** 预算要在策略输入内；若预算相同且历史真的相同，不得赋不同确定性动作。共享前缀中的隐藏目标身份不能泄漏给模型。
6. **“全知专家有害的前提不成立。”** 不做全称主张，保留已消歧任务。失败复盘不能替代新底座对照。
7. **“自动标签不等于正确标签。”** 必须有渲染、状态恢复、对象身份、任务语义、动作成本与界限的独立机器可复算记录；不依赖人工挑图裁决。

## E. 可宣称与禁止宣称

现在可说：已完成一轮有界的核心算法比较，选定一个具体且可证伪的数据算法候选，冻结了继承边界与关键对照。

现在不能说：文献绝无相同算法、论文贡献已充分、从 Qwen 重训必胜、已有正结果、可直接部署、已经有统一 VLN 泛化或保证接收。
“找不到完全相同的标题”未作为准入依据。准入依据是上述实际操作差异及可隔离的必要实验；审稿人仍可能认为收益不足以支持贡献。

后续只推进 [G0B 数据接口任务](../CONTRIBUTION_BOUNDARY_V1.md#8-下一步的唯一任务)。核心不变时停止例行泛搜；实质遗漏/直接同构新先例仍须如实重审。

## 方法来源

本次使用 scientific-critical-thinking 技能区分假设、证据、可观察性和因果对照，并保持 Failure Scientist / Idea Compressor / Adversarial Reviewer 三次顺序判断；这不是三位独立审核者。

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。该引用不证明机器审查具有独立同行评审效力。
