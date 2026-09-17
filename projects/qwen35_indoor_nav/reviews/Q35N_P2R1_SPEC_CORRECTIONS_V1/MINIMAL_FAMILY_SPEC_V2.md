# 最小历史—续接族 V2

状态：`STATIC_SPEC_ONLY / UNREPLAYED`。本文件只修正 R2、R3、R4、R7；沿用 Habitat-Sim v0.1.7、MP3D `17DRP5sb8fy`、R2R-CE v1-3 与原 3×3×2 结构。没有生成坐标、动作轨迹、证书或标签。

## 1. 版本化任务：语言等于机器谓词

原 V1 的“进入房间”“床边”“固定物体”超出实际证书，全部弃用且不得作为别名。V2 在任何路线搜索前冻结以下文字与谓词：

| task | 确定性语言 V2 | 结构化程序 |
|---|---|---|
| `g_D_v2` | 先连续两帧看见餐厅内的一把椅子，再连续两帧看见卧室内的一张床，然后停止。 | `SEE2(chair,dining_room) ; SEE2(bed,bedroom) ; STOP_NOW` |
| `g_K_v2` | 先连续两帧看见厨房内的水槽，再连续两帧看见卧室内的一张床，然后停止。 | `SEE2(sink,kitchen) ; SEE2(bed,bedroom) ; STOP_NOW` |

V2 不声称机器人进入 region，也不声称靠近床。`room_category` 只修饰被看见对象的 MP3D `object.region.category`；策略从 RGB 与语言学习，不能读取 region/object/instance ID。

对象集合以公开类别存在量化，绝不在语言中暗选单一 instance：

```text
Eligible(object_category, room_category) =
  object.region.category.name == room_category
  AND object.category.name("mpcat40") == object_category
  AND normalized raw category satisfies the frozen filter below
```

- dining anchor：`chair` in `dining room`，raw name 包含完整词 `chair`；
- kitchen anchor：`sink` in `kitchen`，raw name 精确为 `sink`；
- terminal anchor：`bed` in `bedroom`，raw name 精确为 `bed`；
- irrelevant control witness：`tv_monitor` in `living room`，raw name 精确为 `tv`。

静态 `.house` 中四类候选均存在；runtime 若解析后的集合为空、region 关系不一致或任何实例不可见，候选族失败。不得换类别、改语言或降低阈值以适配结果；只能另提版本交主 agent。

## 2. 时间语义和 T/F/U

决策时刻 `t` 的因果输入为 observation `O_t` 及已经执行的 `a_0...a_(t-1)`；策略随后选择 `a_t`。`SEE2(C,R)@t` 使用同一 eligible instance 在 `O_(t-1)` 和 `O_t` 的 aligned semantic mask，两个 frame 各至少 256 pixels。RGB 与 semantic 均保存内容哈希；semantic 只供 compiler。

逐时刻原子值：

- `T`：两帧、相机对应、instance ID、object category、object→region relation 和像素计数均完整，且同一实例两帧都满足阈值。
- `F`：上述证据链完整可判，但没有任何 eligible instance 同时满足两帧阈值。这只表示该时刻没有完成“看见”动作，不表示世界中不存在该物体。
- `U`：任一必要帧/哈希/语义映射/实例稳定性/region 关系缺失或歧义，因而不能判 T 或 F。

完整区间事件 `EVER_SEE2(C,R,[0,t])`：存在原子 T 则为 T；无 T 且区间 completeness certificate 为真、所有原子均 F 则为 F；其余为 U。完整性要求 observation/action step 连续、每帧 RGB/semantic 对齐、资产/配置/hash 一致、无非法 action/碰撞、无缺帧。

`STOP_NOW@t`：action log 完整且 `a_t=STOP` 为 T；完整但不是 STOP 为 F；action 缺失/损坏为 U。终止谓词要求 terminal `SEE2` 在同一个决策时刻 `t_stop` 为 T，然后 `a_t_stop=STOP`；不是“以后任何时候停”。

有序程序用强三值时序求值：

```text
PASS iff exists t_anchor < t_terminal == t_stop:
  SEE2(anchor)@t_anchor=T AND SEE2(bed,bedroom)@t_terminal=T
  AND a_t_stop=STOP AND budget/legal/full-log certificate=T
FAIL iff full-log/evidence certificates=T and the Boolean program is false
UNKNOWN otherwise
```

因此可靠完整日志中始终未发生 anchor、顺序颠倒、bed witness 不足或错误时刻 STOP 都可确定为 fail；坏分割、缺映射、缺帧或不完整日志是 unknown。物理轨迹本身非法时 `sample_status=REJECTED`，不生成 Y，不伪装 fail。

三个不同示例（均为规格示意，不是实测）：

- pass：t=20 椅子两帧 T，t=90 床两帧 T 且 `a_90=STOP`，完整性 T → `g_D_v2=pass`。
- fail：完整日志中椅子每时刻均 F，t=90 床 T 且 STOP → `g_D_v2=fail`；结论只是不满足指令中的看见事件。
- unknown：t=20 缺 semantic frame，此后无椅子 T，床与 STOP 可判 → anchor 区间 U，程序 `unknown`，action mask 全 0。

## 3. history、continuation 与纸面矩阵

- `H_D`：形成 `SEE2(chair,dining_room)`，不形成 sink，真实返回 `u`。
- `H_K`：形成 `SEE2(sink,kitchen)`，不形成 chair，真实返回 `u`。
- `H_D_L`：形成 dining-chair，再形成 irrelevant living-room TV witness，不形成 sink，返回 `u`。
- 三者在 `u` 末端等长，随后执行同一 8-action public tail 到 `s`；所有 task anchor 必须早于最近 8-step window。
- `C0`：从 `s` 形成 bedroom-bed witness 并立即 STOP，且不形成 dining-chair/kitchen-sink。
- `C_D`：从 `s` 依次形成 dining-chair、bedroom-bed 并立即 STOP，不形成 kitchen-sink。
- `C_K`：从 `s` 依次形成 kitchen-sink、bedroom-bed 并立即 STOP，不形成 dining-chair。

纸面矩阵仍为 unreplayed expectation：

| task/history | C0 | C_D | C_K |
|---|---:|---:|---:|
| g_D_v2 / H_D | pass | pass | pass |
| g_D_v2 / H_K | fail | pass | fail |
| g_D_v2 / H_D_L | pass | pass | pass |
| g_K_v2 / H_D | fail | fail | pass |
| g_K_v2 / H_K | pass | pass | pass |
| g_K_v2 / H_D_L | fail | fail | pass |

同一真实 trace 用两个程序从头重算；实际证书不符即 frozen-family certification 失败，不改矩阵。对 `g_D_v2+C0`，`H_D` 与 `H_K` 的 query 内容完全相同而标签相反，是最小历史辨识对。

## 4. continuation query V2

q 是未来真实续接的**内容**，不是 `C_D` 名字。它采用规范化结构序列：agent-relative 离散 movement run、实际顺序的公开 event description、最终 STOP；可附 action/RGB content SHA256 供 loader 核对，但这些哈希不进入 reader tensor。q 不含 house/scene/family/continuation ID、坐标、任务要求、残余状态、相容性答案或 Y。

示意序列：

```text
MOVE(TURN_LEFT, repeat=6)
MOVE(MOVE_FORWARD, repeat=12)
OBSERVE(object_category=bed, room_category=bedroom,
        min_pixels=256, consecutive_frames=2)
ACT(STOP)
```

实际 q 必须逐段描述完整 executed continuation，run-length 展开后与监督侧 trace hash 一致。query encoder 只接受 `query_schema_version/coordinate_frame/sequence` 中的枚举动作、非任务性观察描述和顺序/repeat；所有 integrity refs、sample/storage IDs 在 tensorization 前删除。按 semantic projection 的 canonical JSON key order + UTF-8 + controlled vocabulary 编码。存储 ID 改名不改变 canonical semantic bytes 或 reader tensor。q 只在 prefix memory 已经算完后进入独立训练 reader，从不进入 policy/Qwen prefix/cache。

静态不变性例：

1. 把 supervision 的 `continuation_trace_id=C0` 改名为 `foo`，q object 不含该字段，canonical bytes 必须逐字节不变。
2. 同一 `g_D_v2`、同一 C0 query 在 `H_D` 为 pass、在 `H_K` 为 fail；仅 q 的模型不能同时正确。
3. query 中若出现 `outcome/pass/required/remaining_state/house_id/family_id/continuation_id/position` 任一 key，validator 必须报 `QUERY_FORBIDDEN_FIELD`。

## 5. 固定物理配置

未来 runtime/family gate 需把以下配置的规范化 JSON 计入 hash；本轮未创建配置文件：

- agent：cylinder，height 1.5 m，radius 0.1 m；静态场景 physics disabled；`allow_sliding=false`。
- actions：forward 0.25 m，left/right 15°，STOP；非法扩展动作拒绝。
- RGB 与 semantic：各 224×224、HFOV 90°、pinhole、position `[0,1.25,0]`、orientation `[0,0,0]`，两者内外参相同；depth 不进入 compiler 或 policy。
- scene/navmesh/semantic asset 用原 P2 四个 SHA256；renderer/driver/GPU fingerprint 由 runtime gate 产生，不能预填 CPU。
- `history + public_tail <=512` executed actions，public tail 恰为 8；每个 continuation（含 STOP）≤160。碰撞数必须 0。
- merge certificate 同时核对 agent position/quaternion、sensor poses、collision/body config、scene/static-state、reset/seed/frame provenance 与 RGB/semantic hashes；position/yaw fixed tolerance 沿用 1e-4 m/rad，hash 精确相同。

## 6. 有界 discovery 与冻结 certification

候选搜索不是最终族证书。流程和上限在执行前固定：

1. `u` position pool：按 `(episode_id, path_index)` 排序，取本地 R2R-CE train 中 `17DRP5sb8fy` 的前 32 个唯一 start/reference waypoint；每点 snap 后枚举 24 个 15° yaw。静态已见该 house 有 75 episodes，但尚未生成候选池。
2. witness viewpoints：每个 eligible object 最多 16 个，由其 OBB 水平外接圆上 8 个方位×2 个半径（0.75/1.25 m）生成，snap 后朝向 object center 量化到 15°；按 instance/category/position tuple 排序。不可达、重复或不满足 SEE2 的点记 candidate rejection。
3. public-tail 库固定为八个长度 8 模式，`F/L/R` 分别为 forward/left/right：`FFFFFFFF`、`LFFRFFFF`、`RFFLFFFF`、`LLFFRRFF`、`RRFFLLFF`、`LFRFLFRF`、`RFLFRFLF`、`LRLRLRLR`。按列出顺序检查；tail 不得产生任何 task event。
4. 用 `PathFinder.find_path`/greedy follower 生成 outbound 候选。返回 trace 从逐 primitive 逆序列开始，只做相邻旋转的模 24 合并和相邻相反旋转消除；不得跨 `MOVE_FORWARD` 重排，不得删除平移。每个压缩前后 trace 都记录理论 transform 并真实 replay。
5. naïve 每个 forward 的逆需 25 actions，加 outbound forward 共 26；因此 512 上限至多容纳 19 个此类 forward，且实际还要扣 turn、detour 和 8-step tail。任何候选在生成 action list 后先做精确计数，超限只记 `DISCOVERY_HISTORY_BUDGET_REJECT`，不暗示场景不可构造。
6. 三 history 只用已认证不会触发 task event 的原地中性旋转回环 `LR/RL/L^24/R^24` 补齐。无法在 512 内等长、奇偶/姿态不匹配或 padding 产生 event，均拒绝该候选。
7. 组合按上述稳定顺序最多尝试 256 个 family candidates。每次保留参数/hash/失败码和动作计数，不保留未授权大帧缓存。256 个耗尽则 discovery fail。
8. 第一个同时满足 route legality、event pattern、预算、u/s equality 与 query/schema 预检的候选立即写入 immutable `frozen_candidate_hash`；随后不再搜索。对其进行独立 certification，任一重放/标签/相等性失败即 gate fail，不能回到第二候选。

两类失败分开：discovery rejection 用于报告构造筛选率，不是负标签也不是冻结族失败；冻结 hash 后的任何失败是 `FROZEN_FAMILY_CERTIFICATION_FAIL`。

certification 计数为：9 个唯一 `history×continuation` physical traces ×3 registered seeds = 27 次物理重放；每条 trace 由两个 task checker 求值，共 54 个 label evaluations。它们是确定性/一致性重复，**不是 54 个独立统计样本**。所有 seed 共享 scene/config/frozen trace，仅用于检查隐藏随机性。

## 7. 动作监督与恢复

- `g_D_v2,H_K` 的正确恢复是 C_D；`g_K_v2,H_D/H_D_L` 的正确恢复是 C_K。
- history/padding/public-tail 的 mask=0；fail、unknown、rejected cell 全 mask=0。
- pass cell 仅在其预注册 normal/recovery continuation decision positions 允许 mask=1。
- action 去重按 task/history/decision/context 的 V2 key，只消除同一物理 trace 被 cross cell 复制的权重；详见 `IMPLEMENTATION_SPEC_V2.md`。

本规格没有证明任何路线可见、可达、可逆或满足 512/160；这些均为 FAMILY_REPLAY gate 的运行事实。
