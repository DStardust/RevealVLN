# Q35N P2R1 七项规格修订报告

节点：`Q35N_P2R1_SPEC_CORRECTIONS_V1`  
日期：2026-09-09  
结论：`READY_FOR_MAIN_AGENT_REVIEW`

本轮仅关闭主 agent 指定的 R1–R7 静态缺项。论文主线、Qwen3.5-2B、Habitat-Sim v0.1.7/MP3D 与 R2R-CE v1-3 首选均未改变；原 P2 保留且其 SHA256SUMS 13/13 复核通过。结论只表示修订规格可回交审核，不批准 G0R/G1F，不是 runtime、科学或导航收益通过。

## 逐项关闭矩阵

| 项 | 静态关闭 | 核心修订 | 仍待运行/外部条件 |
|---|---|---|---|
| R1 | `STATICALLY_RESOLVED` | runtime 准备与 family replay 拆成两个 `executable=false` gate；安装/下载/cache、renderer GPU、产物分别预算 | 新环境兼容性、owner 指定空闲 GPU、主 agent 授权 |
| R2 | `STATICALLY_RESOLVED` | 删除“床边/进入房间/固定物体”；V2 语言精确对应 same-instance 256px×2-frame `SEE2`，对象按公开类别存在量化 | 实际对象映射、可见 viewpoint |
| R3 | `STATICALLY_RESOLVED` | 原子事件与完整区间采用 T/F/U；完整证据下未发生为 F，缺证为 U；增加 pass/fail/unknown 样例与 X01–X13 | compiler/validator 运行验收 |
| R4 | `STATICALLY_RESOLVED` | q 改为 agent-relative movement/observable event/STOP 结构内容；无 continuation ID/答案/状态/坐标；同 q 异 history 反标签样例 | serializer/query encoder 的 G2 测试 |
| R5 | `STATICALLY_RESOLVED` | M2 明确定义三类 causal program-progress target、mask、`[B,T,3]` logits 和 masked CE；与 physical merge 分开；action key 保留 task/history/time/context | M2 与 dedup 实现验收 |
| R6 | `STATICALLY_RESOLVED` | full sequence 在 MRoPE 前同时含 old memory、实际 action-status、write/action queries；列全 mask/type/position/embed shapes；冻结视觉 feature probe 与参数梯度分开 | 固定 Qwen 对象、shape、PEFT、backward 的 G2 smoke |
| R7 | `STATICALLY_RESOLVED` | 固定 physical config、有限候选池/尾段/视点/256 attempts、逆动作成本和 freeze-then-certify；27 physical replay 与 54 task evaluations 分记；许可分四层 | bounded replay；owner 本地使用确认；发布/商业法律审核 |

完整逐项证据在 [CORRECTION_MATRIX.json](CORRECTION_MATRIX.json)。七项的“static resolved”均伴随 `DEFERRED_RUNTIME`，未用“尚未跑”掩盖静态缺项，也未把纸面修正写成已通过。

## R1：两阶段资源与权限

[RUNTIME_SETUP_GATE_DRAFT.json](RUNTIME_SETUP_GATE_DRAFT.json) 只准备并指纹化独立 Habitat runtime/renderer。其 proposed 上限为新增 disk 40 GiB、network 12 GiB、host RAM 32 GiB、wall 8 h，以及一张由项目所有者明确指定且空闲的 renderer GPU（VRAM cap 8 GiB）。这是未来权限请求，不是当前 allowed；不加载 Qwen、不训练、不终止或抢占进程。固定 v0.1.7 若与现代 driver/dependency 不兼容，gate 停止并回交，不在 gate 内换版本。

[FAMILY_REPLAY_GATE_DRAFT.json](FAMILY_REPLAY_GATE_DRAFT.json) 依赖已验收的 immutable runtime fingerprint 与 owner 的本地资产使用确认。它不安装、不下载，单独给 replay artifact 5 GiB、RAM 16 GiB、wall 8 h 和同一已分配 renderer GPU 预算。原“0 GPU + RGB/semantic renderer”和“5 GiB 同时容纳新环境”的矛盾已经移除。两 gate 都保持 `executable=false`，且 G0R 成功不自动启动 G1F。

本地源码直接证明配置有 sensor 时创建 renderer，并在 EGL 分支选择 CUDA device；没有核验 CPU 软件渲染替代。详见 [SOURCE_EVIDENCE.md](SOURCE_EVIDENCE.md) 与 [RESOURCE_PLAN_V2.json](RESOURCE_PLAN_V2.json)。

## R2/R3：可观察任务与三值标签

[MINIMAL_FAMILY_SPEC_V2.md](MINIMAL_FAMILY_SPEC_V2.md) 将任务版本化为：

- `g_D_v2`：先连续两帧看见餐厅内的一把椅子，再连续两帧看见卧室内的一张床，然后停止；
- `g_K_v2`：先连续两帧看见厨房内的水槽，再连续两帧看见卧室内的一张床，然后停止。

任务不再主张进入 room 或到床边。room 只来自所见对象的 MP3D object→region category；instance ID 仅用于 same-instance 证书，不进语言或 policy。候选类别/原始类别 filter 在路线搜索前冻结，缺对象或不可见即失败，不运行后改词。

在决策时刻 `t`，同一 eligible instance 在 `O_(t-1),O_t` 每帧至少 256 pixels 才是 `SEE2=T`。证据链完整但未满足是 F；缺帧、错配、实例/region 歧义是 U。完整任务只有在 anchor 早于 terminal witness、且 terminal witness 当时立即 STOP、预算/物理/日志完整时 pass；完整可判但公式不成立为 fail，存在影响结论的 U 为 unknown。这里的“未看见”只否定观察任务，不推断世界里没有对象。

[DATA_SCHEMA_V2.json](DATA_SCHEMA_V2.json) 除严格 schema 外列出 13 项 mandatory cross-field assertions：mask/target 长度、因果 step、event threshold、三值程序、完整 18-cell Cartesian matrix、query/trace、ID 不变性、反标签 pair、M2 causal state、action lineage 与 trace accounting。[EXAMPLE_RECORDS_V2.json](EXAMPLE_RECORDS_V2.json) 分别给出 synthetic pass、complete-evidence fail、missing-evidence unknown；全部显式 `synthetic_spec_only=true`。

## R4/R5：query、M2 与 action 去重

query 采用严格结构序列，编码实际 agent-relative movement runs、可观察 category/room 事件与 STOP；可带的 action/RGB SHA 只供 loader 完整性核对，在 tensorization 前删除。semantic projection 禁止 Y、任务剩余真值、相容答案、house/family/continuation ID、坐标和模板身份。canonical JSON/controlled vocabulary 独立编码，必须先算 prefix memory 再把 q 送训练 reader。样例中同一 `g_D_v2+C0` query 在 `H_D` 为 pass、`H_K` 为 fail；把 storage ID 改名不改变 query bytes/tensor。

M2 不再以 physical shared-state 冒充 strong state supervision。每个 causal decision t 的训练标签是 `WAIT_ANCHOR / WAIT_TERMINAL_WITNESS / READY_TO_STOP`，缺证为 UNKNOWN 且 mask=0。M2 head 读取与 M4 相同的 causal memory/action hidden，不读取 semantic/program truth/q/future；输出 `[B,T,3]` 并做 grouped masked CE，部署删除。physical merge certificate 是另一字段，仅证明 histories 的环境状态匹配。

action CE key 现在包含 `task_hash, history_hash, decision_step, causal_policy_context_hash, target_continuation_lineage_hash, target_action, mask_version`。task/history/time 不同，即使动作字符串相同也保留；只去掉同一动作监督位置因 crossed q/Y matrix 被复制的额外权重。

## R6：Qwen 完整张量流

[IMPLEMENTATION_SPEC_V2.md](IMPLEMENTATION_SPEC_V2.md) 固定每步完整序列：

```text
processor text/image
+ 8 old-memory placeholders
+ <=8 executed-action/collision-status tokens
+ 8 write queries
+ 1 action query
```

先构造完整 `input_ids_full/attention_mask_full/mm_token_type_ids_full [B,L]`，再由固定 `Qwen3_5Model.get_rope_index` 路径对整个 L 计算 `position_ids [3,B,L]`；不能先算 base positions 再追加 17+A 个 token。视觉 features 与 image placeholders 数精确匹配后 scatter；替换 memory/query embeddings 后得到 `[B,L,2048]`，调用 `Qwen3_5TextModel`，cache 始终 `None/false`。所有返回 shape、官方对象路径和 tokenizer resize 留给 G2 实测，不因静态规格写全而升级为 runtime verified。

视觉塔冻结时 parameter `.grad=None` 是预期。G2 另建 early frozen feature leaf probe，检查 downstream BCE sensitivity；同时在真实图检查 writer/LoRA/reader 的 finite nonzero gradients 与参数更新。两者不会混为一项。

## R7：有界构造、计数与许可

V2 固定 agent 1.5 m/0.1 m cylinder、sensor `[0,1.25,0]`、RGB/semantic 224×224/HFOV90、0.25 m/15°、physics/sliding disabled、0 collision、history+tail≤512、continuation≤160。候选池由 17DR 的 R2R-CE waypoint、24 yaw、每 eligible object 16 viewpoints、8 个明确 tail pattern 和最多 256 family attempts 组成。

V1 naive inverse 中每个 outbound forward 的来回成本为 26 actions，512 最多容纳 19 个且尚未计 turn/detour/tail；V2 因此在生成每个候选 action list 后先精确计数。只允许相邻 turn 的群等价约简，不能跨 forward 重排或 teleport。discovery rejection 只计筛选失败；第一个 preflight-valid candidate 立即 freeze hash，其后任一 certification 失败则 gate fail，不回搜下一个。

计数修正为每 seed 9 个真实 history×continuation traces，3 seeds 共 27 physical replays；同一 trace 用两个 task checker 得 54 task evaluations。它们只检查工程一致性，不是 54 个独立统计样本。

许可分为：已有资产获取依据、受控本地研究使用、本地产物生成/保存、对外发布与商业部署。文件在本地不等于授权。执行 G0R/G1F 前，项目所有者至少须确认 MP3D 资产合法获取及当前受控本地研究使用范围；任何 raw/derived RGB、semantic、route 对外发布和未来商业用途保持 blocked，等待适当的书面/法律许可。本轮没有接受条款或发布数据。

## 剩余前置条件与停止点

静态修订后仍真实未知：Habitat v0.1.7 在现有 driver 的运行兼容性、17DR 路线/可见性/汇合与 18-cell labels、Qwen wrapper/PEFT/gradient、以及发布/商业许可。它们分别进入 G0R、G1F、未来 G2 或外部 owner/legal 决定，不要求改变主线。

主 agent 下一步只需审核 R1–R7 是否按 [CORRECTION_MATRIX.json](CORRECTION_MATRIX.json) 关闭。若接受，再**另行**决定是否授权 G0R；不能从本报告自动启动 G0R、G1F、G2 或训练。

完整机器结论见 [result.json](result.json)。保护文件前后内容哈希、原 P2 复核与本交付 [SHA256SUMS](SHA256SUMS) 用于静态完整性，不构成科学验证。本机没有可用的 Draft 2020-12 validator，且本轮禁止安装；因此只执行 JSON parse、schema required/property 与本地 `$ref` 结构检查，以及对示例的 X01–X13 对应手工断言。正式 schema-runtime 验收仍属于未来工程 gate。
