V6_READER_REPAIR_AND_STOP_DATA_COMPLETE

已完成一处真实读出器修复、300 次轻量记忆参数更新，以及真实 FIT 轨迹的 STOP 监督接口。修复后单族拟合由 15/18 提高到 16/18，仍低于精确程序状态监督 B2 的 18/18。这个结果支持修复实现缺陷，不支持交叉续接优于 B2，也不是闭环导航收益。本轮没有新导航 episode、Qwen 前向或 Qwen 权重更新，未采用新导航策略。

工作起点是 `codex/q35n-v5-runtime-20260917` 的 `fb8473aa019958952efe99753889a908c8b07e55`，新分支为 `codex/q35n-v6-query-repair-20260918`。按用户“直接进行下一步的修复与设计”和 V5 的有界开发授权执行。实际读取了 CURRENT_STATUS、V5 RESULT/REPORT、完整 100 对日志中的原生 A、pilot 的 model/run/DATA/保存的特征和参数、ordinary 训练索引/划分/原始策略与监督/回放证书、官方 R2R train 目标和所选房屋 navmesh。根与项目 AGENTS 及 private-house-code 技能已读取；历史启动限制不覆盖最新用户授权。已有 V5 结果和旧失败记录未修改；工作区其他人的 `.gitignore`、`AGENTS.md`、`README.md` 修改及无关未跟踪文件未纳入提交。

代码与运行证据在 [query_reader_repair_v2](../../research/continuation_memory_v1/query_reader_repair_v2/)。原读出器用 GRU 最后一个 hidden 编码长度 112、312、422 的续接 query。三个 query 具有长公共尾部；在原已训练参数上，最终表示最大差异仅为 0 和 2.09e-7。对表示平方和求导，长 query 前 20 token 的梯度为 0，112-token query 也只有 4.03e-18。这是训练读出器丢失前部查询信息的具体证据，不是对整个 Qwen 长程能力的结论。

`model.py` 仅将训练 query 的聚合改为全部有效 GRU 输出的时间均值。当前每次输入一个未 padding 的 query；未来批量 padding 必须加 mask，不能直接复用无 mask 均值。参数形状、数量、运行记忆更新和动作前向不变；未来 query 不进入运行记忆或动作输入。在同一已训练参数上的诊断中，均值表示差异恢复到 0.06405、0.03639，前 20 token 梯度范数为 0.09345、0.03244、0.02455。此梯度诊断针对 query 编码，不能冒充新的视觉编码器训练证据。

正式修复复测使用同一初始 state_dict、同一 764 个已缓存真实 Qwen 特征、同一数据、种子 1209、FP32、AdamW 0.001 / weight_decay 0.01。B2、原 Ours、修复 Ours 各 100 次更新；没有搜索学习率、阈值、槽数或选择最好 step。

| 实际运行臂 | 有效结果拟合 | 动作拟合准确率（492 个 owner） | 交叉结果 BCE |
|---|---:|---:|---:|
| B2 精确组合式程序状态监督 | 18/18 | 59.96% | 不作为 B2 的优化指标 |
| Ours_last 原读出器 | 15/18 | 60.57% | 0.32875 |
| Ours_mean 修复读出器 | 16/18 | 60.77% | 0.20671 |

B2 的有效结果由预测的组合式程序状态和续接事件按确定性规则求出，不是把没有受监督的神经 query 分支拿来削弱 B2。保存参数回读确认原 B2 和 Ours_last 的最终 state_dict 与 V5 逐位一致；Ours_mean 实际更新了 writer、recurrent、action 和 query/result 分支。每臂 1–100 步日志完整、所有保存张量有限。检查点在本地保留，GitHub 提交对应 SHA256、参数指纹和逐步日志，不上传原始权重。

同一暴露族上的进一步机制检查如下，均非独立测试：

| 臂 | 正确历史记忆 | 只输入末 8 个观察并重新起记忆 | 同任务、错误历史供体 | 等状态 H_A/H_A_I 替换后预测变化 |
|---|---:|---:|---:|---:|
| B2 | 18/18 | 10/18 | 6/18 | 0/12 |
| Ours_last | 15/18 | 12/18 | 3/18 | 0/12 |
| Ours_mean | 16/18 | 12/18 | 6/18 | 0/12 |

仅按历史与任务、完全忽略 query 的逐行多数标签可达到 15/18；仅按任务与 query、完全忽略历史的查表上限是 14/18。原 Ours 达到 15/18 不能单独证明它读懂了续接，也不能据此说它完全没有记住历史。修复后错误历史替换改变结果与动作 logits，但尚未验证分叉动作是否正确、实际闭环任务是否改善。记忆同时可能编码空间信息；swap/sham 不证明任务状态完全解耦。原 SEE2 含义、规范化拼接、缺失负控制和 `training_admission=false` 全部保留，不能称为“到访房间”或正式泛化数据。

STOP 修复的现有证据与代码在 [ordinary_stop_diagnosis_v6](../../closed_loop_bench/ordinary_stop_diagnosis_v6/)。只读分析 V5 原生 A 的完整 100 条、19,305 次决策，标签严格对应动作执行前的状态，最后动作之后到达但已无后续决策的帧不算停止机会。48 条在决策时曾进入 3m 范围，其中 28 条未成功；55 条在远处 STOP，52 条从未在决策时进入范围。这些集合有重叠。

原 STOP margin 的逐决策 AUC 为 0.5087；按 episode 均衡为 0.6173，轨迹内去重输入后为 0.5696；五屋分别约 0.543、0.621、0.401、0.573、0.673。它不足以支持直接加一个全局 STOP 偏置就能修复问题。本轮没有扫描阈值。已存轨迹的特权“在首次入范围时截断”得到 48% 只是诊断参照，既不是可部署 SR，也不是改变运动或抑制旧 STOP 后所有策略的上限。

`prepare_fit.py` 已实际调用 CPU Habitat PathFinder 和冻结 V3 距离后端，为 6 条现有真实 R2R FIT 路线生成监督。按房屋名、record ID 固定选择前三个 FIT 屋、每屋两条物理路线，不参考模型得分。前两个屋为 head_fit：177 决策；第三个屋为 head_check：157 决策。三屋均参与过基座 FIT，不能称盲测或基座未见场景。

334 决策中 can_stop 正例 94、负例 240、UNKNOWN 0；88 个正例的原教师动作仍为运动。这表明“公共指标允许停”与“教师接下来执行 STOP”是不同监督，不能把这 88 个原标签直接改写为错误。新标签只表达原 geodesic < 3m 成功半径条件，不证明任意指令或程序语义已完成；未知距离用 mask 排除。所有原教师动作只读保留。

`audit_fit.py` 实际解码并校验了全部 334 个 RGB 文件，检查每步最近两帧、前八个实际动作、位置与原监督对应，确认两个分区的物理路线不交叉。`split_inputs.py` 输出 `POLICY_INPUTS.jsonl` 与 `OFFLINE_SUPERVISION.jsonl`；模型输入提取器只返回 instruction / rgb_paths / executed_actions，拒绝额外真值字段与错位标签。房屋、位置、目标、距离、ID 留在离线审核或标签文件，不能作为 token。原来源、navmesh、导出数据、源码和 RGB 哈希审计均已落盘。没有创建 renderer，也没有新增环境动作。

下一步实现设计已据这些真实接口收紧：先固定 best4k 编码器，从上述因果输入提取执行位特征，训练一个独立线性 can_stop 读出头；原四类动作标签继续保留。使用 head_fit 的类别统计固定 BCE 权重，初始诊断阈值固定 0.5，不在这 100 条 INTERNAL_DEV 轨迹上搜索阈值。先报告 head_check 的逐屋 AUC、precision/recall、校准和错停/漏停，明确只有两条检查路线的局限。六路线接口样例不够支撑导航效能结论；扩展时继续按完整物理路线和指令族隔离。只有实际特征前向、反向、保存回读与独立于拟合的检查均完成后，才把停止头接入新版本配对运行器。首个接入设计保留原生 STOP，只测试新增提前 STOP，因而不能宣称解决原有远处 STOP；公共成功定义、运动尺度与 500 决策预算不变。本轮交付数据接口和设计，STOP 头尚未训练或部署。

研究线的下一比较仍优先 B2 与修复后的 Ours。现有目录还包含其他跨屋候选导出，不能声称只有一个真实族；抽查到的 manifest 仍未获得 training admission。下一步需要逐族审核缺失负控制、规范化处理和物理证书，并统一跨族语义词表：当前 family-local 类别/房间整数不能直接跨族当同一语义。保持整屋/历史/续接/改写族隔离，再固定小型比较，不反复调当前 18 个 cell 追平 B2。现阶段结论是“尚无超过精确状态监督的证据”，研究优越性状态为 UNTESTED。

CPU 验收：读出器 3 项回归测试通过；STOP/数据当前 4 项测试通过（其中包含最初 2 项，不能加算成 6 项），另完成保存参数回读和真实数据审计。预算按 launcher 占用时间计 161.266 秒，即 0.044796 GPU 小时；加 V5 pilot 累计研究用量 0.120724 GPU 小时，未包含旧工程评测的 1.363966 GPU 小时。设备为 GPU1 / RTX 5090 / `GPU-734a5268-31fe-6452-105b-36cd08c3d9c8`，单卡、无重试，全部自有 GPU 进程已退出，未向外部任务发信号。资源日志保留共享设备状态，不能当成部署性能测试。

实际使用的命令（仓库根目录）：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/query_reader_repair_v2/launch.py
CUDA_VISIBLE_DEVICES='' projects/qwen35_indoor_nav/.envs/q35n_qwen_g2_v1/bin/python3 -I -B projects/qwen35_indoor_nav/research/continuation_memory_v1/query_reader_repair_v2/verify_saved.py
CUDA_VISIBLE_DEVICES='' projects/qwen35_indoor_nav/.envs/q35n_habitat_v017_g0r/bin/python3 -I -B projects/qwen35_indoor_nav/closed_loop_bench/ordinary_stop_diagnosis_v6/prepare_fit.py
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/closed_loop_bench/ordinary_stop_diagnosis_v6/split_inputs.py
CUDA_VISIBLE_DEVICES='' projects/qwen35_indoor_nav/.envs/q35n_qwen_g2_v1/bin/python3 -I -B projects/qwen35_indoor_nav/closed_loop_bench/ordinary_stop_diagnosis_v6/audit_fit.py
```

上述运行产物使用排他创建，原目录重执行会拒绝覆盖；它们是执行记录，不是继续追加训练的命令。当前停在已完成的实现修复、实测和数据接口节点，没有资源或数值阻断。V5 的 100 对 A/B SR 均为 20%、无成功率收益、不采用控制器的结论保持原样。历史 41800 全 1839 条 SR22.02%、best4k 旧开发100条 SR21%、小集拟合通过但检查 CE 升高、独立入口16输入动作验收仍分别成立；SR40、研究泛化与真机部署都未完成。
