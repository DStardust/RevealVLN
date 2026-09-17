# 最小真实历史—续接族规格

状态：`PAPER_SPEC_UNREPLAYED`。本文件定义可由未来 G1 尝试构造的单一族，不声称路线、标签或语义类别已在模拟器中成立。

## 1. 固定环境与任务

- 引擎：Habitat-Sim v0.1.7/challenge-2021；MP3D house `17DRP5sb8fy`。
- 动作：`STOP, MOVE_FORWARD(0.25 m), TURN_LEFT(15°), TURN_RIGHT(15°)`；策略只看因果 RGB 与已执行动作，不看 semantic/depth/pose。
- 编译器候选语义：`D=region 0_5`、`K=0_9`、`B=0_0`、`L=0_7`。其静态类别码分别为 d/k/b/l；G1 必须用 `SemanticScene.category.name()` 与实例映射确认，否则族生成失败。
- 共同状态：中性连接区域的规范位姿 `u`；三个历史真实返回 `u`，随后执行完全相同的 8 个 action-observation pair 公共尾段抵达 `s`。关键事件必须早于这 8 步窗口。

两个任务共享相同 RGB 历史，换任务必须重新计算记忆：

| task | 结构化程序 | 确定性语言 |
|---|---|---|
| `g_D` | `VISIT(D,witness) -> STOP_AT(B,bed_witness)` | 先进入并看见餐厅区域中的固定物体，再到卧室床边停下。 |
| `g_K` | `VISIT(K,witness) -> STOP_AT(B,bed_witness)` | 先进入并看见厨房区域中的固定物体，再到卧室床边停下。 |

“当前条件”是最终 `STOP_AT(B)`；“过去事件”是先前的 `VISIT(D)` 或 `VISIT(K)`。语言由模板与已验证 category name 确定性生成，不调用 LLM 猜真值。

## 2. 三条真实历史

候选路线只通过 pathfinder/greedy follower 产出，再完整 replay：

- `H_D`：从 `u` 真实执行到 D，形成 D witness，再执行物理逆路线回 `u`。
- `H_K`：从 `u` 到 K，形成 K witness，再物理逆路线回 `u`。
- `H_D_L`：到 D 形成 witness，再完成与任务无关的 L 绕行，回到 `u`；不得进入 K。

`MOVE_FORWARD` 没有直接 backward 动作。对无碰撞 outbound trace 的单步逆操作为：先转 180°（12 次同向 15° turn），`MOVE_FORWARD`，再反向转 180°；turn 的逆为 left/right 对换。每条候选必须 replay，任一碰撞或位姿漂移即拒绝，不能用 teleport 掩盖失败。通过可逆的中性回环把三条历史填充到相同实际动作数，history 上限固定 512；padding 不得进入 D/K/B/L witness 条件。

`set_state()` 只允许在“从完整 reset + actions 重建”已证明等价后用于加速。同一完整 H+C cell 的证书必须来自端到端 replay，不能从 `s` teleport 开始。MP3D 静态场景也要记录 scene/static-state fingerprint；如发现门、对象、时钟或随机态变化，直接失败。

## 3. 三条续接与纸面标签

残余预算固定 160 个动作，三者都必须在 `s` 真实可达并以 STOP 结束：

- `C0`：直接到 B/bed，避开 D 与 K。
- `C_D`：先到 D 形成 witness，再到 B/bed，避开 K。
- `C_K`：先到 K 形成 witness，再到 B/bed，避开 D。

纸面预期 Y（`P/F` 表示 pass/fail，不是实测）：

| task/history | C0 | C_D | C_K |
|---|---:|---:|---:|
| g_D / H_D | P | P | P |
| g_D / H_K | F | P | F |
| g_D / H_D_L | P | P | P |
| g_K / H_D | F | F | P |
| g_K / H_K | P | P | P |
| g_K / H_D_L | F | F | P |

这是 3 histories × 3 continuations × 2 crossed tasks = 18 cells。换成 `g_K` 后重算完整历史与续接，不可沿用 `g_D` 的内部任务状态。实际证书若与表不符，保留事件日志并使 G1 失败，不得强制标签；边界/语义不清则为 `unknown`。

## 4. 自动事件证书

编译器使用与 RGB 同位姿的 semantic sensor，查询 `semantic_scene.objects`、object category 与 `object.region.id`。这些是监督专用特权信息，绝不进入 policy record。

- `VISIT(R,witness)`：agent position 被唯一判入 R 的 AABB，且属于 R 的已登记静态实例在 224×224 semantic frame 中至少占 256 pixels，连续 2 帧；保存 RGB/semantic 内容哈希、instance id、region id、category 和 step。
- `STOP_AT(B,bed_witness)`：当前动作是 STOP，位置唯一属于 B，并满足同样的 bed-instance witness。
- `unknown`：semantic scene/对象关联缺失、category name 不匹配、多 region 边界、instance 像素不足、观测哈希缺失或任一必要状态不可判定。`unknown` 不归入 fail。
- `fail`：所有必要证据可判定，但有序程序在预算内不成立。

256 pixels × 2 帧是预注册工程阈值，不是科学最优阈值。G1 不得逐例人工改标签；人工只能审查聚合失败码和少量预先规定的可视化审计样本，不能覆盖证书。

## 5. 汇合与隔离判据

每个 history 在 `u` 及公共尾段末端 `s` 记录：agent position/quaternion、各 sensor pose、已执行动作数、collision 序列、scene/asset/config/seed、frame/time/RNG/reset provenance、static object/door fingerprint、RGB 与 semantic SHA256。

G1 预注册相等标准：history 间 position spread ≤`1e-4 m`、yaw spread ≤`1e-4 rad`、sensor poses 一致、同点 RGB 与 semantic SHA256 精确一致；3 次重放均满足。任何条件不成立即丢弃该族，不放宽阈值。所有 history 真实动作数相等；续接共享 160-step residual budget，超限/碰撞/非法 action 均失败。

policy loader 白名单仅允许 instruction、因果 RGB、实际已执行动作、reset flag 和 sensor profile。scene/family/split/坐标/program/event/q/Y/文件名均在监督侧。公共尾段相同只能消除最近窗口差异，不能被当成全隐藏状态相同；后者由上述 replay certificate 单独检查。

## 6. 恢复示范与 loss mask

- `g_D,H_K` 的恢复示范使用 `C_D`；`g_K,H_D` 与 `g_K,H_D_L` 使用 `C_K`。
- 历史 prefix 的全部动作 `action_loss_mask=0`，因为它包含用于制造不同过去的动作，不是当前决策模仿目标。
- 满足任务的 recovery/normal continuation 为 `mask=1`；负 continuation cell 仅进入 continuation BCE，动作 mask 全 0。
- 普通 R2R-CE 由未来编译器生成并 replay 的有效示范为 `mask=1`。不能把 reference path 直接称作已执行动作。
- action CE 对每条唯一 target continuation 只计一次，不随交叉矩阵复制；BCE 才使用全部有效 cells。

## 7. 真实 API 映射与接口差距

| 操作 | v0.1.7 源码接口 | 用途/限制 |
|---|---|---|
| 初始化/reset | `Simulator(...)`, `reset()`, `initialize_agent()` | 每次完整 replay 建立同源初态 |
| 执行动作 | `Simulator.step(action)`, `Agent.act()` | 记录 action、collision、observation |
| 状态读写 | `Agent.get_state()`, `set_state()` | 读状态；写状态仅作已验等价后的缓存优化 |
| 可达性 | `PathFinder.find_path()`, snap/island/is_navigable | 产候选路线，不能替代 replay |
| 路径动作 | `make_greedy_follower()`, follower `find_path()` | 无噪声候选；返回终止 `None` 需映射 STOP |
| 事件 | `semantic_scene`, `SemanticScene.regions/objects/categories` | 仅 compiler/supervision 侧 |
| 观测 | `get_sensor_observations()` | 对齐 RGB/semantic 并哈希 |
| 终止 | 本项目 compiler 对 STOP、预算和事件程序判定 | Habitat API 不自动给本任务 Y |

R2R-CE 的 reference path 是 waypoint 级普通导航资料；本族工作在 Habitat 连续场景中的离散动作接口。只有 G1 完整回放后，才能说“连续动作环境中物理成立”。当前不存在这一验证。
