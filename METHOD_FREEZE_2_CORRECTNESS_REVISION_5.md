# RevealNav Method-Freeze-2 正确性与新颖性修订 5

**修订编号：** `MF2-CR5`  
**日期：** 2026-08-24  
**状态：** Phase-0C 决策事件重建；CR5 全部关卡通过前禁止生成训练特征或启动训练。  
**修订类型：** 可复现正确性错误触发的版本化修订，同时把经过验证的事件构建范式列为
候选 benchmark 贡献；不增加 RevealNav 在线方法模块。  
**替代范围：** 暂停 `MF2-CR1` 至 `MF2-CR4` 中由稀疏参考路径转角直接定义分支、
`B/T`、Reveal 窗口和 35 项审核集合的全部结论。既有文件仅作为失败证据保留，不得作为
标签、审核包或训练输入。  
**保持不变：** `FROZEN_SPEC.md` 的主问题、REE/ECOG/OPP 三模块、U/A/D 语义、
RxR-CE-en/R2R-CE 数据边界、Oracle/Frozen 两种候选协议、实验矩阵和投稿硬门槛。
`FROZEN_SPEC.md` 与 `PHASE0_PROTOCOL.md` 必须继续逐字节不变。

## 1. 触发本修订的可复现错误

旧生成器把每个稀疏参考路径锚点 `reference[j]` 当作决策分支入口 `B`，把
`reference[j+1]` 当作目标 `T`，并主要在水平 `XZ` 平面匹配候选。该假设至少导致：

1. 路线锚点转向、楼梯折返或高度变化被误当成拓扑分支；
2. `B/T` 可能位于不同楼层或指向与当前视觉 Reveal 相反的行进方向；
3. 从 `T` 的反向接近也可能因端点或二维距离接近而被错误匹配；
4. 单走廊、已经完全可见的楼梯或仅有小幅视角变化的片段被强制构造成正事件；
5. MLLM 能正确定位“发现楼梯”等 Reveal 子句，却被错误的几何目标判为不一致。

已确认的代表错误为 `ep41233_turn01`。因此旧 35 项集合、其 `B/T`、`C1-C3`、
`P,D1-D3` 和 CR4 审核板全部标记为 `SEMANTICALLY_SUSPENDED`。不得通过重画俯视图、
放宽阈值或要求审核者迁就旧 `B/T` 来修补。

## 2. 新的事件构建范式

CR5 固定采用 **完整轨迹后见发现—语义多视角提议—三维定向落地—可执行反事实验证—
因果前缀投影** 五阶段范式。英文工作名为
**Hindsight-Verified Semantic Branch Mining (HVSBM)**。名称只是当前工程标识，在完成最新
相关工作审计前不声称首次提出。

### 2.1 完整轨迹的后见分支发现

第一提议器查看完整原始 instruction 与完整参考执行的有序视频帧，在整段轨迹上主动定位
潜在决策区，而不是在预设稀疏转角附近被动判断。视频帧必须保留确定性时间 ID；优先使用
有序独立帧或本地确定性 storyboard，禁止依赖服务端不可复现的隐式抽帧。

为兼顾完整性和上下文长度，固定使用分层输入：

1. 以每次 30 度转向和每约 0.5 m 位移为最密采样上限，生成覆盖整条路线的全局时间轴；
2. 长轨迹按有重叠的连续 chunk 输入，每个 chunk 前后均保留上下文，边界候选必须在相邻
   chunk 中复核；
3. 另给全路线均匀 storyboard，供模型确定当前片段在完整 instruction 中的阶段；
4. 模型输出一个或多个候选时间区间、中心帧、支持帧、reveal/action 子句和不确定性，
   不输出 `B/T` 或坐标；
5. 对同一轨迹的 chunk 结果按稳定 frame ID 合并；只有有时序证据的候选才进入多视角阶段。

这种完整轨迹访问是 **离线 privileged hindsight annotation**：它只确认事件是否存在、参考
执行后来选择了什么，绝不能成为 REE 的在线输入，也不能直接定义 `T_R`。

参考路径转角、冻结候选前端变化、门洞/走廊/楼梯视觉线索和指令动作词只作为独立的召回
补充与构建基线。任何几何 seed 均不预填“存在决策点”、目标分支、`B/T` 或 Reveal 时刻。
原 104 项稀疏转角集合保留作对比和困难负例池，其原标签全部作废。最终候选池取完整轨迹
MLLM 提议与确定性高召回 seed 的并集，并记录来源；不得只保留二者交集造成隐性低召回。

### 2.2 事件级离线多视角语义提议

每个轨迹提议区间或补充 seed 的固定证据包包含：

- 完整原始 instruction，以及按原文字符区间切分、带 SHA-256 的确定性子句；
- 严格按执行顺序排列的 approach/through/departure RGB 帧和动作 ID；
- 候选中心前约 1 m、中心附近、中心后约 1 m 三个实际可导航位姿；
- 每个位姿 12 个等间隔方位角视图，间隔 30 度，视图 ID、相机位姿和原图 SHA-256
  只在项目内记录；
- 三张 12 方位角总览板用于高召回提议，模型选中的原始单视图随后以完整分辨率用于复核。

MLLM 只从 instruction 与图像提出语义出口，不接收 navmesh、最短路、参考坐标、目标
坐标或旧 `B/T`。它必须结构化输出：

- `DECISION`、`NO_DECISION`、`AMBIGUOUS` 或 `INSUFFICIENT_EVIDENCE`；
- 所有可见的语义出口，而不是只描述参考路线；
- 每个出口的稳定视觉描述、方向类别和支持它的 view/frame ID；
- instruction 所要求的目标出口，或明确无法唯一确定；
- `reveal_clause_ids`：说明何种分支/地标开始可识别；
- `action_clause_ids`：说明真正选择/进入哪个分支的动作；
- `already_visible_before_seed`、`no_alternative_exit`、`retrace_only` 等拒绝理由；
- 最早支持判断的帧 ID、置信度和简短理由。

MLLM 输出永远是 `proposal`，不是真值。禁止用 CLIP 距离、像素变化、动作熵或单次
MLLM 置信度直接接纳事件；这些只能作为基线或诊断量。

### 2.3 三维定向落地

确定性 grounder 使用受控的 depth/navmesh/controller 信息，把模型提出的出口落地为
有方向的三维分支对象：

```text
decision_region Q
incoming directed path A -> Q
target directed branch B* -> T*
alternative directed branches Bi -> Ti
```

其中 `B` 是离开共享决策区域后的分支入口，`T` 是沿同一可执行分支继续前进的下游点，
不是稀疏参考路径的下一个锚点。必须满足：

1. 全部距离、匹配和轨迹顺序使用三维坐标与 navmesh geodesic/controller path；二维
   `XZ` 只能用于可视化，不能决定身份；
2. 参考执行必须先通过 `B*` 再通过 `T*`，索引严格递增；候选必须从 `B*` 一侧进入且
   局部运动方向与 `B*->T*` 一致；
3. `T` 固定取离开决策区后沿分支 1.5--2.0 m 的可导航下游点；场景受限时不足 1.5 m
   必须拒绝，不能缩短到端点凑数；
4. 至少存在一个不同于目标出口的可执行备选分支；在离开半径 1.0 m 后，目标与备选的
   初始方向夹角至少 45 度，且在 2.0 m 分支长度内不能重新合并为同一控制轨迹；
5. 任何出边的前 1.0 m 不得只是沿 incoming path 原路返回；单一楼梯折返、路线自交、
   反向靠近 `T`、跨楼层二维别名和端点近邻均为硬拒绝；
6. 楼梯本身不是负例：只有在同一决策区还存在至少一个不同的可执行出口，且目标方向、
   高度演化和执行顺序全部成立时，楼梯才可构成分支。

数值阈值先在 RxR-train 与 scene-disjoint val-seen 工程集上做一次预注册敏感性分析；不得
查看 val-unseen/test。若阈值必须改变，需要新的版本化修订，禁止按人工喜好逐例调整。

### 2.4 可执行反事实验证

正事件必须对目标分支和至少一个备选分支分别执行冻结 controller rollout，并验证：

- 两条轨迹都可执行、有限且无碰撞/导航失败；
- 目标轨迹与参考执行及 action clause 一致；
- 备选轨迹在语义和几何上均不是同一出口的抖动、副本或短暂分叉后重合；
- 从分支入口继续、返回 checkpoint、错过 checkpoint 后再返回的成本均可复现；
- 交换目标/备选或打乱 instruction clause 后，语义验证器应拒绝或改变目标判断。

`NO_DECISION`、目标已在候选中心前稳定可见、只有一个出口、反向/折返、高度别名、多个
同等合理目标和无法执行等样本必须保留为有类型的困难负例。`ep43805_turn02` 与
`ep7619_turn05` 暂作为 `NO_DECISION_OR_ALREADY_VISIBLE` 校准样本，而不是删除。

### 2.5 严格因果回放

只有通过 2.3/2.4 的固定三维分支，才允许在原始 63 度 ego-FOV 上重放 Reveal。这里是把
后见事件标签重新投影成在线可学习的严格因果前缀：

1. 全景多视角只能用于离线标签构建，不能作为 REE 在线输入；
2. 每个 prefix 只使用当时及过去 RGB 和冻结候选前端输出；
3. `T_R` 由目标进入候选集合、与竞争出口可区分、reveal/action 决定性语言证据闭合
   且连续 `K=3` 个 prefix 稳定共同决定；
4. `T_X` 和 cost frontier 只能在分支身份固定后计算；
5. 未来帧遮蔽、时间顺序打乱、指令最小替换和视图移除必须作为因果负向控制。

## 3. 自动标注与质量控制

固定采用三层、相互可追溯的角色：

1. **MLLM proposer：** 高召回枚举分支和语义；
2. **deterministic verifier：** 三维定向落地、执行与因果硬门；
3. **MLLM adversarial judge：** 在看不到 proposer 自由文本理由的独立请求中，检查
   视图打乱、目标互换、已可见和无选择困难负例。

同一模型的多次调用具有相关性，不得称作独立标注者。人类只审核经过确定性验证后的
分层样本和最终 Gold 集；MLLM-only 标签不得称为人工真值。所有请求必须记录精确模型
标识、base URL、prompt/schema SHA、温度、usage、媒体 SHA、原始响应与重试；API 密钥
不得进入任何日志、命令或交付物。

第一轮提示词必须以用户指出的六项校准集做预飞行，但不得把用户判断写成模型答案：

- 预期有判别价值：`ep34121_turn02`、`ep46758_turn03`、`ep56443_turn01`；
- 预期应触发拒绝或重建：`ep41233_turn01`、`ep43805_turn02`、`ep7619_turn05`。

预飞行只检查范式能否区分问题类型。任何为了匹配六项预期而逐例改 prompt 的操作必须
   记录，且最终 prompt 只能整体锁定后一次性运行完整轨迹和候选池。

## 4. 数据构建范式作为论文贡献的边界

HVSBM 可以成为 RevealBench-CE 的独立数据/benchmark 贡献，但不是因为使用了 MLLM。
只有同时满足以下条件才允许在论文摘要或贡献列表中主张该范式：

1. 相比“稀疏转角即分支”、几何-only、图像变化-only、MLLM-only 四类构建基线，HVSBM
   在 scene-disjoint 人工 Gold 子集上显著提高有效决策事件 precision，并报告 recall；
2. 人工审核至少覆盖全部自动接纳事件的分层随机样本、全部失败类型的样本以及冻结 Gold
   test；最终仍满足 `FROZEN_SPEC.md` 的 2,000/600/三人标注门槛；
3. 发布完整 schema、构建代码、拒绝原因、哈希/provenance、确定性 verifier 和允许公开的
   元数据；Matterport/RxR 派生图像受许可约束，不擅自再分发；
4. 提供 proposer、三维 grounder、反事实执行、因果回放和困难负例的独立消融；
5. 证明更干净的数据不仅让标签自洽，而且改善 REE 的校准、False-Ready/Premature
   Commitment 和跨数据集迁移；
6. 完成投稿前的一手文献更新，明确区别于大规模合成 instruction-trajectory 数据、一般
   MLLM-VLN 评测、长时程任务生成和驾驶领域反事实标注。

在这些关卡通过前，允许称其为“候选数据构建贡献”或“工程假设”，不能声称 novel、SOTA
或保证 CVPR 录用。

## 5. CR5 机器契约与版本边界

完整轨迹 locator 的固定 JSON Schema：

`artifacts/phase0/phase0c_cr5_contract/CR5_MLLM_TRAJECTORY_LOCATOR_SCHEMA.json`

完整轨迹 locator 的固定英文提示词：

`artifacts/phase0/phase0c_cr5_contract/CR5_MLLM_TRAJECTORY_LOCATOR_PROMPT_V2.md`

事件级 MLLM 提议的固定 JSON Schema：

`artifacts/phase0/phase0c_cr5_contract/CR5_MLLM_BRANCH_PROPOSAL_SCHEMA.json`

固定英文提示词：

`artifacts/phase0/phase0c_cr5_contract/CR5_MLLM_BRANCH_PROPOSAL_PROMPT_V2.md`

确定性验证器的输出必须另有 schema，并禁止把 navmesh/pose 字段复制回 MLLM 请求。
任何 schema/prompt 修改都要产生新版本号和 SHA，不得覆盖既有原始响应。

## 6. 重新开放训练前的硬关卡

CR5 完成以下全部条件前，Phase-0C 继续 `NO_GO`：

1. 完整轨迹分层时间轴、重叠 chunk、三个位置、每位置 12 方位角的证据包构建器通过路径、
   哈希、时序、chunk 边界召回和 train-only 边界测试；
2. 六项预飞行完成，且不再出现反向 `B/T`、二维跨层别名或单通道伪分支；
3. 完整 RxR-train 轨迹先经固定 hindsight locator 扫描；其提议与独立高召回 seed 的并集
   再经固定 branch proposer 运行，结构化输出和 usage 100% 可追溯；
4. 全部自动接纳项通过三维方向、双分支执行、目标一致、反事实和因果回放硬门；
5. 困难负例完整保留，任何模型歧义或验证失败均 fail closed；
6. 至少 50 个自动接纳正事件进入新版人工 pilot，初步有效率至少 60%，或给出具有统计
   置信区间的证据证明可达到冻结的 300/25% Phase-0 门槛；
7. 新版审核板不再依赖旧 `B/T/C1-C3/P,D1-D3`，并明确区分离线全景证据与在线 ego-FOV；
8. `FROZEN_SPEC.md`、`PHASE0_PROTOCOL.md`、checkpoint、环境、允许数据 payload、
   `toporeveal/` 与 reserve 文件全部回归无漂移。

关卡通过后只能请求生成训练特征；不能据此直接声称方法有效、benchmark 成立或达到
CVPR 投稿标准。

## 7. 当前非结论与相关工作风险

大规模合成 VLN 数据已有 ScaleVLN、A New Path 等工作；NavGen 已将自动数据生成作为
长时程 VLN benchmark 组成；VLN-MME 也显示通用 MLLM 的三维/时序判断不可靠。因此
“使用 MLLM 生成或标注 VLN 数据”本身没有足够新颖性。CR5 的可辩护点只能是：从普通
路线中挖掘 **有方向、可执行、含真实备选、带 Reveal--Expiry 因果时序和困难负例** 的
决策事件，并以确定性 3D/控制器验证约束模型提议。

本修订只固定新的构建假设和拒绝边界，不证明 HVSBM 已工作，不证明任一旧样本有效，不
证明数据范式或 RevealNav 优于基线，也不构成 CVPR 正分或录用保证。
