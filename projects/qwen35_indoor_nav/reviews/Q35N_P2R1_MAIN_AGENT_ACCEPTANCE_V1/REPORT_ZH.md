# P2R1 主 agent 接收结论

日期：2026-09-09。只复核原 R1–R7，不新增方向审查。

## 裁决

`ACCEPTED_FOR_RUNTIME_GATE_WITH_PRECONDITIONS`。
**P2 的主体静态规划完成，不再退回一轮 P2 文档。** 保留论文主线、Qwen3.5-2B 与 Habitat/MP3D/R2R-CE 首选；下一步为 G0R 独立环境与渲染验收。
接受的是已形成可实现/可证伪规格，不是全部示例细节正确、数据已可用、算法创新已获经验支持或已具备导航收益。

当前 G0R 仍待所有者确认数据使用依据、资源预算及 GPU 分配方式。本文不执行安装、下载或渲染，不自动把草案 executable 改为 true。

## 七项接收范围

| 项 | 主 agent 结论 | 后续验证归属 |
|---|---|---|
| R1 | 接受独立 runtime 与 replay 的资源/权限拆分；零 GPU 渲染及 5 GiB 环境矛盾已消除 | G0R 测实际环境、renderer 和成本 |
| R2 | 接受版本化的 SEE2 观察任务作为小型数据契约；不再冒称床边/进入房间 | G1F 核验可见实例及语义映射；自然任务能力仍须另测 |
| R3 | T/F/U 和跨字段检验已可实施；不再将可靠负例一律当 unknown | G1F 的真实日志、正式 schema 与语义 validator；时间索引按下节统一 |
| R4 | query 已有真实运动/观察内容和隔离规则，不再只有后缀名字 | G1F 查 trace 一致性；G2 查张量隔离 |
| R5 | M2 任务进度监督与物理状态证书已分开；条件化 action 去重明确 | G1F/G2 查严格时序标签、loss mask 与去重 |
| R6 | 全长 token/动作/位置契约及 frozen-feature probe 已明确 | G2 按固定源码核验对象、MRoPE、cache 和梯度 |
| R7 | 候选搜索/失败分层/物理与任务计数/许可分层明确 | G0R 使用前置条件；G1F 有界发现与冻结重放 |

上述接收不将原执行者自报的 7/7 当作已独立运行证明。未测的路线、标签和模型可运行性继续 DEFERRED_RUNTIME。

## 主 agent 直接给定的实现澄清，不另开 P2 修订轮

这些是原关闭项内部的索引/表述修正，不改变方法或扩大本次 G0R；后续 G1F/G2 写执行配置时必须纳入，并保留原文：

1. **动作索引统一（R3）**：V2 定义 O_t 已观察、a_<t 已执行，接下来预测 a_t。因此从 prefix_cutoff=t 开始的续接，其首动作决策索引应为 t，后续目标索引不小于 t。X03 的严格大于和示例 [101,102] 不能在 t=100 且只有 forward/STOP 两动作时直接用作真实 trace。后续声明式配置改为 decision-index 口径；执行后帧另记为 O_(t+1)。不允许漏掉首个需要历史区分的动作监督。
2. **严格先后（R5）**：READY_TO_STOP 必须满足某个 anchor 时刻严格小于当前 terminal witness 时刻；首次 anchor 与 terminal 同时出现不自动 READY。M2 标签和 Y 检查器使用同一已冻结时序语义。
3. **位置 API 表述（R6）**：原“get_rope_index 内部 compute_3d_position_ids”调用方向文字不作为接口事实。按固定源码由 compute_3d_position_ids 调 get_rope_index，或显式直接调用后者并解包 (position_ids, rope_deltas)。G2 以实际对象签名/返回和完整长度断言验收。
4. **示意记录不得转正（R3）**：synthetic_spec_only 样例永久保留为示例。真实记录必须从实际执行、真实哈希与证书重新产生；两级 validator 通过也不能把占位哈希或纸面标签直接翻成实测。

此外，256 像素是操作性标注阈值，不是自然语言“看见”的普适真值；两帧观察任务仅为接口/机制起点，不能替代最终 VLN 自然路线和目标导航评估。这是现有范围约束，不增加 G0R 的任务。

## 本轮证据与完整性

实际读取了修订报告、关闭矩阵、结果、两个 gate、资源、数据族、模型规格、schema、样例及来源范围。
P2R1 交付 SHA256SUMS 13/13；原 P2 13/13；P2R1 的 before/after 相同；23 个受保护文件在主 agent 更新前逐项匹配。
原 P2/P2R1 产物均未改写。主 agent 随本次接收更新 README/STATUS 属于新的显式状态变更，不是执行者违反保护约定。
本轮没有运行 Draft 2020-12 validator、模拟器或模型；不继承执行者的“示例断言通过”为完整数据语义测试结论。

## 下一步与授权边界

唯一下一节点：`Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1`。

- 范围：独立 Habitat 环境、固定源码/依赖记录、一场景加载、对齐 RGB/semantic 和一个动作的接口 smoke。
- 拟议上限：总新增磁盘 40 GiB、网络下载 12 GiB、RAM 32 GiB、墙钟 8 小时、一张空闲渲染 GPU、显存 8 GiB。
- 不下载/加载 Qwen、不生成历史族、不训练、不做导航效能、不终止或抢占其他进程。
- 用户需确认已有 MP3D 的合法获取及受控本地使用依据；这不替代发布或商业用途许可。
- 用户可指定空闲 GPU，或明确授权执行者只读检查后选择一张空闲卡；不得把允许自动选择理解成允许释放别人的进程。
- 确认后执行会话记录授权与实际 GPU UUID/ordinal，形成节点内 execution config。若前置条件缺失，停止，不能把本报告当已经获批安装。
- G0R 完成回交；G1F 仍需单独批准，G2/训练均未授权。

完整执行交接见 [G0R 交接](../../HANDOFF_G0R_RUNTIME_SETUP_ACCEPTANCE_V1.md)。主 agent 不再要求对同一主线做常规文献复审。

## 技能与方法引用

使用 scientific-critical-thinking 区分静态接收、可执行授权和经验支持，避免既把未知项当失败，又把示例检查当科学 PASS。
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。本轮核对最新 v2 作者记录，未进行新算法文献搜索。
