# 主 agent 对 P2 的验收 V1

日期：2026-09-09。被审对象：[P2 原始交付](../Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1/REPORT_ZH.md)。机器技术与科学契约审查，不是独立人工同行评审。

## 裁决

`RETURN_FOR_BOUNDED_CORRECTIONS`。接受 Habitat-Sim/MP3D 和 R2R-CE v1-3 作为首选接口与普通数据路线；保持 V3 论文主线与 Qwen3.5-2B 底座，不重新选题。
**尚不批准原 G1 草案执行。** 原执行者的 READY_FOR_MAIN_AGENT_REVIEW 是合理回交状态，不是获得主 agent 接收或科学通过。

下面区分真实规格缺陷与应由未来运行检验的事项。路线能否回放、实际梯度、泛化收益未知，本身不是拒绝静态规划的理由；但内部矛盾、标签定义和缺失接口必须先修正。

## 本轮实际核验

- 原交付 SHA256SUMS 的 13 个条目全部通过；before/after 相同；16 个保护文件与当前磁盘匹配（主 agent 更新状态之前）。原产物不修改。
- 阅读了报告、结果、数据族、模型规格、schema、样例、资源草案和来源清单；JSON 可解析不等于跨字段语义校验通过。
- Habitat 相关路径 realpath 均在项目内，直接查本地 simulator.py、Simulator.cpp、WindowlessContext.cpp 和 Mp3dSemanticScene.cpp。
- 官方 Qwen 固定 revision 的 config 可核：2048 hidden、24 layers 等与报告一致；官方 Transformers v5.15.0 原始源码可核 get_rope_index、compute_3d_position_ids、TextModel inputs_embeds/cache 接口。不能由这些接口存在推断自定义 wrapper 已可运行。
- 未运行环境、模型、梯度或导航，无新训练和 GPU 操作。没有证据可在本轮确认完整数据许可；条款 PDF 入口可定位，但正文重取失败，不能签发法律许可。

## 必须修正的七项

### R1 高：环境准备与重放预算矛盾

证据：NEXT_ENGINEERING_GATE_DRAFT.json:42 要新建隔离环境，:68–75 禁止任何下载/GPU且总新增磁盘 5 GiB；RESOURCE_PLAN.json 预计环境 8–12 GiB、cache 8–16 GiB。
本地 habitat_sim/simulator.py:86 根据传感器启用 renderer；src/esp/sim/Simulator.cpp:177–184 创建窗口外 GL context；WindowlessContext.cpp:44–45 的 headless EGL 分支选择 CUDA device。
因此“不加载模型”不等于“不用渲染 GPU”。本版本没有经核验的 CPU 软件渲染替代，不能将 gpu_count=0 当作可执行事实。

要求：分开隔离 runtime 准备与单族重放；分别列安装/下载/cache、渲染设备、产物预算与权限。可以提出一张空卡的渲染预算，但本轮不启用、不占卡、不终止进程。首选版本仍是候选，现代驱动/依赖兼容性由工程验收实测。

### R2 高：语言与事件判据不等价

证据：MINIMAL_FAMILY_SPEC.md:15–16 要求“卧室床边停下”，:58–59 仅要求 region AABB 与床像素可见，没有距离或可达停靠区域。
看见床不能证明到床边。room AABB 是包围盒，不自动证明真实房间内点；静态源码只证明装载了 bbox，并未提供该语义等价。
“固定物体”也未说明语言指哪个实例，不能在多候选物体中用隐藏 instance ID 选唯一真值。

要求：在运行前选定一种可观测、机器可验证且语言对应的定义。若任务本来只要求在卧室看见床就停，显式改写模板并版本化，不宣称床边；若保留床边则给可追溯空间判据。区域/目标多实例歧义必须处理。不得运行后改文字去适配成功。

### R3 高：确定不发生与证据缺失未分开

证据：MINIMAL_FAMILY_SPEC.md:60 将 instance 像素不足列为 unknown，:43–50 却需要多项因未发生访问事件而得到的确定 fail。
若所有未达到访问阈值都成为 unknown，则关键负例无法判定。反过来，不能将坏分割或缺失元数据都当 false。

要求：定义原子事件 T/F/U 与完整有序任务结果的推导。可靠完整日志明确不满足操作谓词可为 F；映射/帧缺失或不可判才为 U；部分观测逻辑不能用“没看见”证明全环境对象不存在。提供合法 pass、合法 fail、缺证 unknown 的不同示例。
schema 当前只限制 outcome=unknown 的 mask；须同时约束 fail 无正动作监督，并提供 event 阈值、mask 长度、因果时间、完整交叉矩阵等跨字段检验。schema 解析通过不替代这些断言。

### R4 高：续接查询内容尚未实现到规格

证据：EXAMPLE_RECORDS.json 的 q 只是“执行 C_D 是否通过”，DATA_SCHEMA.json 的 query 是无结构字符串。
这仅用任意后缀 ID 命名问题，没有 V3 规定的真实运动/事件描述，也不支持新后缀的语义处理。不能将此样例直接变成训练接口。

要求：定义 query 的 movements/events/时间顺序/内容引用白名单与编码方式，不含 Y、残余任务真值、任务相容性答案或家屋/模板身份。保持只进独立训练 reader。增加 ID 重命名不变、同 q 在不同历史出现相反标签等静态验收例。

### R5 高：最强对照和动作去重的定义需落地

证据：报告自查把“强状态监督可匹配”解释成物理 shared-state 匹配；result 也只登记 strong_shared_state_match。它与 M2 的任务程序状态监督不是一回事。
模型规格仅提到 M2 program-state head，没有监督张量和采样时点。action CE 的 once_per_unique_target_sequence 也未明确包含 history/task 条件。

要求：给 M2 的执行进度状态标签、unknown mask、时间位置、loss 和同样的因果记忆输入；它是训练标签，不是部署 oracle。明确物理状态匹配和强状态监督各自证据。
动作去重键必须包含任务/历史/时刻与目标动作上下文，不能因为不同历史有相同 action string 就丢掉应学习的样本。只消除交叉复制导致的重复权重。

### R6 中高：Qwen 位置、动作和冻结视觉梯度的张量流缺项

证据：IMPLEMENTATION_SPEC.md 每步伪代码 processor 只接 instruction/RGB，最近动作如何编码未列出；先算 position IDs 再 append 9 个 query，未扩展各字段长度。
冻结视觉塔后，早期视觉特征默认可能不保留梯度；不能把未设置 probe 的 .grad=None 直接判定不存在从 memory 到 LoRA/writer 的可训练路径。

要求：先定义完整序列位置（含 old-memory、动作、write/action queries）、对应 attention/mm types，再按官方接口计算或明确扩展位置索引；列精确对象层级、形状与长度断言。
写明冻结视觉特征的人工梯度 probe 与可训练参数梯度检查的差别。保持未来 q 隔离、无未登记 cache，并将 wrapper 可运行性留给 G2，而非要求本轮运行。

### R7 中高：固定场景的构造和许可前置条件需闭合

证据：逆每一步 MOVE_FORWARD 要 25 个逆动作；连同原动作，一个 forward 来回贡献 26 步，512 总长最多容纳 19 个这样的 forward（尚未计普通转向/绕行/公共尾段）。这不证明场景不可构造，但不能默认其路线符合预算。
原方案任一候选失败就停止，却尚未固定 u、尾段或候选搜索范围。也没有完整 sensor 高度/姿态、agent footprint、navmesh 配置和稳定实例身份规则。

要求：给有界候选构造规则及失败计数，区分候选搜索拒绝和冻结族最终重放失败。可在动作语义等价且真实可重放的前提下合并逆路线中的重复旋转；不得用 teleport 或事后放宽观测匹配伪造汇合。
补齐 physical config。54 次是 18 cell ×3，不是 54 个独立统计样本；两任务共享的物理 trace 与任务检查层分别登记。
分开已有资产的获取/使用依据、本地生成、对外发布、未来商业部署许可。缺许可证明就保留前置条件，不以“文件在本地”视为授权。当前不对外发布。

## 不应错误升级为失败的事项

18-cell 未 replay、Qwen 未 smoke、未见家屋未选、确认样本量无方差依据等，均已诚实记录。不能以尚未运行否定创新主线，也不能在修正规格后自动声称已解决。
G1 若只验证一个旧暴露家屋，即使通过也只是工程契约，仍无泛化或算法收益证明。

## 主 agent 后续决定

下一任务固定为 `Q35N_P2R1_SPEC_CORRECTIONS_V1`：只关闭 R1–R7，保留原 P2，输出新的独立版本和逐项证据，不泛搜、不新增模块、不运行。
修正后主 agent 只核验这些关闭项；真实运行未知项进入对应工程 gate，不继续增加与该 gate 无关的论文级要求。
当前所有实现/安装/仿真/训练/GPU 授权仍为 false；主线创新预审状态不变，scientific_pass=false。

## 引用与技能影响

本次 scientific-critical-thinking 技能用于区分构念不匹配、缺失数据、证据等级与执行可行性，未把机器检查当人工同行评审。
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。本日核对作者及最新 v2 记录。

接口复核：[Qwen 固定 config](https://huggingface.co/Qwen/Qwen3.5-2B/raw/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json)、[Transformers 原始源码](https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。本轮仅核相关接口段，不声称完整库审计。
许可入口：[MP3D 条款](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf)。正文重取失败，本报告未给条款法律解释或使用许可结论。
