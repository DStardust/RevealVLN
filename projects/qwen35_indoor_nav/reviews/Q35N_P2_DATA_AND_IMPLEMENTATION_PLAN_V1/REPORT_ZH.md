# Q35N P2 数据与实现规划报告

节点：`Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1`  
日期：2026-09-09  
结论：`READY_FOR_MAIN_AGENT_REVIEW`

这是静态数据/实现规格已经足够具体、可以交主 agent 审查的结论；不是算法实现许可、数据验收、科学 PASS 或导航收益。`scientific_pass=false`、`runtime_verified=false`，本轮新增训练与导航 episode 均为 0。

## 首选决定

唯一首选是 **Habitat-Sim v0.1.7/challenge-2021 + Matterport3D semantic scenes**。普通导航示范来源唯一选 **R2R-CE v1-3 train split**。理由不是沿用旧线偏好，而是官方 VLN-CE v1-3 明确使用该 simulator/data 组合，本地又有相同 tag/commit 的源代码、90 套 MP3D scene 四件套和 phase0 R2R-CE 文件；其 API 同时覆盖真实离散动作 replay、pathfinder、RGB/semantic observation、region/object 语义和 agent state。

本轮没有选择 AI2-THOR/ProcTHOR：冻结的 V3 不要求动态物体，增加第二引擎会扩大接口与许可证面。也没有选图导航作为替代，因为图上拼接不能证明 Habitat 连续场景离散动作中物理成立。

首个机制族固定在旧暴露 house `17DRP5sb8fy`，只用于接口验收，永不作为 clean confirmation。R2R-CE reference path 不是现成 executed action label；未来必须由 follower/pathfinder 产候选、再完整 replay 和自动认证。具体资产、版本、哈希和许可见 [ASSET_AND_SOURCE_MANIFEST.json](ASSET_AND_SOURCE_MANIFEST.json) 与 [SOURCE_EVIDENCE.md](SOURCE_EVIDENCE.md)。

## 最小物理族

[MINIMAL_FAMILY_SPEC.md](MINIMAL_FAMILY_SPEC.md) 固定一个 18-cell crossed family：

- 三条 history：访问餐厅 `H_D`、访问厨房 `H_K`、访问餐厅并做无关客厅绕行 `H_D_L`；三者从规范位姿出发，沿实际动作回到同一 `u`，补齐相同动作数，再执行相同 8-step 公共尾段到 `s`。
- 三条 continuation：直达卧室 `C0`、补访问餐厅后到卧室 `C_D`、补访问厨房后到卧室 `C_K`，共享 160-action residual budget。
- 两个 crossed tasks：先见餐厅再在卧室床边 STOP 的 `g_D`，以及先见厨房再在卧室床边 STOP 的 `g_K`。换 task 必须在同一原始历史上重新计算 instruction-conditioned memory 和全标签。
- 事件必须由 agent region、semantic object-region-category 关系和 aligned semantic mask 自动认证；witness 工程阈值固定为至少 256 pixels、连续 2 帧。缺语义/边界歧义归 `unknown`，不能猜成 fail。
- 纸面 Y 表已经给出，但 18 个 cell 均为 `PAPER_SPEC_UNREPLAYED`。任何实际不符都使下一 gate 失败，不能事后改标签或放宽阈值。

物理回路不靠 teleport。`MOVE_FORWARD` 的候选逆操作由 180° 转向、forward、转回组成，且所有 outbound/inverse/common-tail/continuation 必须从 reset 完整 replay。`set_state()` 只可在证明与 reset+actions 对 frame/RNG/static state/sensors/observations 等价后用于缓存优化。

恢复动作监督只落在满足任务的 continuation：例如 `g_D,H_K` 用 `C_D`，`g_K,H_D/H_D_L` 用 `C_K`。历史动作 mask 全 0，负/unknown cell 的 action mask 全 0；action CE 按唯一 target sequence 去重，交叉矩阵只扩展 continuation BCE。

## 数据隔离和自动检查

[DATA_SCHEMA.json](DATA_SCHEMA.json) 用 draft 2020-12 定义了互斥的 `policy_input`、`supervision_only`、`family_manifest`。示例在 [EXAMPLE_RECORDS.json](EXAMPLE_RECORDS.json)，全部明确 `synthetic_spec_only=true`，outcome 为 unknown，不伪装实测。

policy 顶层严格白名单仅含任务文字、因果截止、因果 RGB 引用、实际已执行动作、reset 与 sensor profile；sample_id 只作 join key并禁止 tokenization。house/scene/family/split、pose/坐标、hidden task program、semantic event、future q、Y/outcome 和文件名侧信道全部留在监督侧。loader 必须先计算 prefix memory，再连接独立 q reader。

同一 house 整体只属于 train/development/confirmation 之一，旧线暴露另设字段；旧暴露与 split 不能混为一谈。采样顺序是 family→task→valid cell 均匀，BCE 先在 family-task 内平均，action target 去重，避免更大 cross matrix 获得隐式权重。非法 replay、漏字段、未来泄漏、semantic 歧义、非确定性、失败样本误当负例都有明确自动失败码。

## Qwen3.5、记忆与训练接口

[IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md) 固定 model `Qwen/Qwen3.5-2B@15852e8...` 与官方 Transformers `v5.15.0` 源码。官方接口依据包括 multimodal processor 输出、vision features、`inputs_embeds`、显式 position ids、hidden states 与 cache 控制。

本项目 wrapper 采用 8×2048 BF16 连续 memory slots、最近 2 帧 RGB、最近 8 个实际动作、4-action head。每步先按原始 multimodal token IDs/types/grid 计算 MRoPE position IDs，再把保留 placeholder embeddings 替成旧 memory，加入 8 个 write queries 和 1 个 action query。Qwen3.5 是 hybrid cache，故每步强制 `use_cache=false`、`past_key_values=None`，并 reset rope delta；不能偷存 KV、linear-attention recurrent/conv state 或完整历史。

训练专用 continuation reader 只在 memory 计算完后接收 `m,I,q`；q 不回写 memory、不回调 prefix、不影响 action path。换 instruction 重新计算 memory。部署时移除 reader，但 memory writer/slots 与算力并不消失。

该 wrapper 不是官方开箱功能，仍为 `UNVERIFIED`。下一模型 gate 必须单独检查 MRoPE/placeholder/grid、PEFT 模块、cache/reset 和短序列梯度。16-step 短序列不 detach memory；continuation BCE 必须对早期关键 RGB 路径和 writer/LoRA 产生 finite nonzero gradient，并在 optimizer step 后看到目标参数变化。长序列 TBPTT 另立协议，在此之前不声称已解决长历史训练。

M1/M2/M4 的可比 manifest 字段已经列全：同 revision/init、数据、采样、token/step、action mask、seed、资源与评估 episodes，仅 auxiliary mechanism 有预注册差异。当前没有方差依据，因此没有擅自冻结确认样本量或科学效应门槛。

## 资源与许可

[RESOURCE_PLAN.json](RESOURCE_PLAN.json) 将现有 `du` 观察与未来估算分开。只读锚点包括 MP3D 21 GiB、现有 simulator tree 7.5 GiB、旧环境 8.1 GiB、旧 cache 8.4 GiB、phase0 61 MiB。未来独立环境估 8–12 GiB、cache 8–16 GiB、官方权重约 4.57–5.5 GiB、G1 数据硬上限 5 GiB。训练 24–40 GiB 与 54–216 aggregate GPU-hours 只是缺少 throughput/batch 测量时的规划范围，不是授权或实测。

MP3D/R2R-CE 涉及 Matterport3D 条款与 CC BY-NC-SA 3.0；在发布衍生 RGB、semantic frame、路线或编译数据前必须完成许可复核。旧 ETP-R1 工作树有修改，只允许读资产/配置；旧可写环境不复用。

## 下一工程节点建议

[NEXT_ENGINEERING_GATE_DRAFT.json](NEXT_ENGINEERING_GATE_DRAFT.json) 提议 `Q35N_G1_MINIMAL_FAMILY_DATA_ACCEPTANCE_V1`，且明确 `executable=false`。主 agent 若批准，其固定范围为一个旧暴露接口 house、18 cells、每 cell 3 次完整 replay、种子 1109/2209/3309，共 54 次；不加载 Qwen、不训练、不衡量导航收益、GPU 数为 0。

关键硬标准：0 collision；history 等长且 ≤512；continuation 同为 160-action residual budget；汇合 position/yaw spread 分别 ≤1e-4 m/1e-4 rad；sensor/static state 相同；公共尾段 RGB/semantic hash 精确相同；witness 256 pixels×2 frames；18-cell 标签与纸面表一致且关键对比无 unknown；policy 白名单/schema/hash/failure audit 全通过。任一失败即停止，不按结果放宽。

G1 后仍需独立 G2 验 Qwen wrapper/gradient，再由主 agent 预注册 G3 的样本量和效应门槛后才可进行 M1/M2/M4 导航试验。数据 gate 通过也只能说明工程契约成立。

## 关键缺口与逐项自查

| 问题 | 状态 | 证据与剩余缺口 |
|---|---|---|
| 数据物理可构造 | `UNVERIFIED` | API/候选 house/路线构造规则明确；未运行 18-cell replay |
| 事件可观察且自动标注 | `STATICALLY_SUPPORTED` | SemanticScene region/object 与 aligned semantic frame 有源码接口；具体 category/instance/witness 未运行确认 |
| 无人工逐例依赖 | `STATICALLY_SUPPORTED` | 确定性模板、证书和 unknown/failure 规则已写；仍待 compiler 实现验收 |
| Qwen 接口有真实依据 | `STATICALLY_SUPPORTED` | 官方 model/config/Transformers v5.15.0；自定义 memory+MRoPE wrapper 为 runtime unverified |
| 未来监督隔离 | `STATICALLY_SUPPORTED` | 分文件 schema、白名单与 compute-memory-before-q contract；待 loader test |
| 关键历史有梯度路径 | `UNVERIFIED` | 无 detach 的短序列图与 G2 判据已定；没有 backward 实测 |
| 强状态监督可匹配 | `UNVERIFIED` | state/sensor/static/observation 判据已定；`set_state` 非完整快照且尚未 replay |
| 资源可追溯 | `STATICALLY_SUPPORTED` | 本地 `du` 与官方权重大小为锚点；显存/吞吐/全数据量未测 |

最关键未决项是：`17DRP5sb8fy` 的 region 名称/实例和路线尚未由 runtime 确认；旧 runtime 不能直接复用；Qwen memory wrapper/LoRA/MRoPE 没有 smoke；MP3D 衍生数据发布许可待审；clean confirmation houses 尚未分配。它们都有边界明确的后续 gate，不需要改题。

## 完整性说明

本节点产物还包括 [LOCAL_STATIC_AUDIT.json](LOCAL_STATIC_AUDIT.json)、保护文件前后哈希和交付物 [SHA256SUMS](SHA256SUMS)。静态 JSON/schema/link/hash 检查只用于交付完整性，不构成数据、运行时、科学或导航验证。
