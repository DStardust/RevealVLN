# Q35N 项目完整接管说明与新会话 Prompt

维护日期：2026-09-10，Asia/Shanghai。本文汇总研究决定、数据资产、当前生产、失败记录、授权和下一步；不搬迁、复制或删除原项目文件。运行状态会随后台进程变化，新会话必须实读复核。

**第 1 节和最终交接快照是本次移交的执行状态入口；旧 README、STATUS.json、AGENTS 中的“最新”段落有不同时间的历史状态，不能据旧文字重复启动。后台任务在交接时继续运行，不需要重启。**

## 1. 最快读懂当前状态

2026-09-10 10:16 CST 主 agent 实读快照（数量继续增长）：

| 项目 | 已经发生的事实 | 尚未成立的结论 |
| --- | --- | --- |
| 当前研究路线 | Qwen3.5-2B + 交叉续接监督的执行记忆，V3 主线已冻结 | 没有科学收益、泛化、SOTA、CVPR 录用保证 |
| 普通数据 | 已闭合严格合格累计 5849 条路线、17548 条官方指令、1394744 条指令条件动作 | 尚未达到 300 万；不是同数的独立物理决策 |
| 特殊旧自动队列 | batch100/101/102/200/201 共 14 个强审核合格族 | 不是全项目历史族总数，不是 14 个独立场景 |
| 特殊当前运行 | GPU1/2 已强审 batch220/221/240 各3族、241为2族，本波新增11族；已自动进入222/242 | 不是全项目历史总量，也不是11个独立场景 |
| 特殊新增四卡 | 114批、342新候选已逐批准备/审核/主签；GPU3/4/5/7实际开始300/301/302/303并写出轨迹 | 342不是合格族；新四卡首批尚未闭合，链式恢复尚待实际首轮验收 |
| 普通新来源 | EnvDrop首20000路线队列正在GPU6运行；固定sentinel真实强审3/3通过；shard1已完成460/1000，滚动审核449 | 滚动计数不等于最终合并；尚未产出全部2万 |
| 新特殊采集来源 | 43 FIT屋/172新位置来源、独立scout runtime均已CPU准备 | 未启动新scout；不是172个实际可达/可视hub或完整族 |

最终状态以本文末尾的“最终交接快照”和磁盘最新receipt为准。已启动不代表每批最终通过；达到完整训练量级仍需后续生产波次。

## 2. 用户目标、授权与严格范围

- 用户要具有学术诚信、贡献充分、故事清楚、能争取 CVPR 竞争力的通用室内 VLN；近期优先获得可学习的大批量合格数据和真实收益，不再无休止写方案/换小补丁。
- 长期服务居家医疗养老机器人多 agent 系统：本线是导航底座，不是医疗决策、身份识别、关节控制或安全认证。低延迟、小模型、轮足式机器人后续兼容是工程需求，不伪装成论文已测贡献。
- 用户没有人工审核人员。采用可核验的自动标签、反例测试、独立代码/证据复核；不能称机器审核为人类同行评审。
- 唯一授权项目根：`/mnt/data_nas/deeprobotics/daiyang/vla`，以下简称 ROOT。本线为 `ROOT/projects/qwen35_indoor_nav`，以下简称 LINE。所有项目数据、环境、模型、下载缓存和新产物限 ROOT 内；禁止沿旧 `/mnt/daiyang/vla` 链接借用其他项目。
- 用户确认 MP3D 合法取得、有授权文件并批准预算。仅本地合法研究使用，不自动推断衍生数据可以公开再分发。
- GPU0 真实外部任务不可碰。GPU1/2 外部 context 不可停止。GPU3–7 只有经 PID/starttime/argv/cwd/UUID/tmux 二次核实的占位可借用，用后恢复。100% 利用率可能只是占位，不是造数。
- 用户允许连接失败时尝试 `proxyon`。不得绕过 TLS 校验或把代理凭据写日志；下载限项目内并保存官方来源/字节数/hash/许可。
- 当前执行任务是普通和特殊数据生产、扩源与交接，**不是新训练授权**。过去 SFT 已结束；不要看到可用 GPU 就启动长训。
- 已存在的变更属于用户及并行会话；不做 git reset/checkout、清空目录、删失败、改锁文件、移动预留空间文件。

## 3. 论文故事与算法：不要重新从旧 A/B/C 小补丁开始

权威主线：[MAINLINE_FREEZE_V3.md](MAINLINE_FREEZE_V3.md)。完整工作流：[06_END_TO_END_METHOD_AND_VALIDATION_V1.md](06_END_TO_END_METHOD_AND_VALIDATION_V1.md)。近邻与贡献裁决：[P1 报告](reviews/Q35N_P1_PAPER_CORE_ADJUDICATION_V2/REPORT_ZH.md)。

一句话：**让导航模型记住会改变后续任务完成方式的执行历史，而不只是记住看过的画面或模仿下一动作。**

例子：同一句“先经过餐厅，再到卧室”，两段历史到达同一走廊，一段已经经过餐厅、一段没有。相同的“直接去卧室”续接会得到不同任务完成结果。这些真实交叉续接结果约束导航实际使用的记忆。

三个核心模块：

1. **真实数据编译器**：匹配物理汇合状态的历史 × 真实合法续接 × 任务；确定性程序检查器产生通过/不通过/未知，另记物理失败与资源截断。
2. **运行期导航模型**：通用预训练 Qwen3.5-2B + 有限连续执行记忆槽 + RGB/指令/实际已执行动作 → 更新记忆并输出导航原语/STOP。不是随机初始化，也不加载成熟 VLN 策略后打补丁。
3. **训练期续接读出器**：从同一运行记忆、指令和离线续接查询预测交叉结果，和动作监督联合训练；部署去掉读出器，保留执行记忆。

```text
因果历史 + 指令 → Qwen/执行记忆 → 下一导航动作
                         └→ 训练期续接读出器 ← 离线续接查询
                                      ↓
                            自动核验的交叉结果监督
```

未来续接/语义真值/世界坐标/任务检查器隐状态不得进入导航前缀、动作提示、KV cache 或持久状态。修改指令须重算相应任务条件记忆，不能复用别的指令下的记忆冒充新状态。

贡献主张是**具体可执行数据关系与运行记忆联合训练算子**，不是首次有记忆、BCE、自动机、进度、PSR 或主动感知。有限续接集合等价不证明全局最小充分状态，也不主张 UAD 数学闭环。

UAD 仅保留历史/证据随时间演化的启发，不继承旧 U/A/D 标签或旧 PASS。EgoCoT-Bench、VisualThink-VLA 是用户/同组相关工作，复用须登记与归功，不把测试答案转成训练标签，不完全照搬概念。旧 A/B/C/B-VLFM/UAD 是隔离历史，不恢复为本线底座；需要历史证据再从 ROOT 的研究档案有界读取。

主线冻结后不例行重开全量文献搜索；若出现真正同构直接先例、核心方法变更或科学漏洞才有界重审。`algorithm_novelty=PASS_FOR_SPECIFIED_CLAIM` 是限定主张的研究准入，不是永久“无撞车/贡献充分”证书。

## 4. 模型、环境与已完成/失败的训练事实

- Qwen：`Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。
- 本地模型：`LINE/runtime/models/Qwen3.5-2B_15852e8`；环境：`LINE/.envs/q35n_qwen_g2_v1`，Transformers 5.15.0。
- G2 最小接口：真实 RGB 两步，D=2048、8 记忆槽、冻结视觉塔、rank8 LoRA、官方 MRoPE、记忆/reader/LoRA 梯度和一次诊断更新已验证；约 6.57GiB 峰值 allocated。入口：[G2](reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/REPORT_ZH.md)。这不是长历史机制训练或完整导航模型通过。
- Habitat：项目内独立 `.envs/q35n_habitat_v017_g0r`，Habitat-Sim v0.1.7/challenge-2021 + MP3D semantic scenes。不要升级现有环境、复用其他项目 cache、假定未登记环境兼容。
- 原 SFT：加载/训练/重载接口通过，但训练后 10 条闭环不 STOP。后续 D1 原真实更新等价门槛失败，保留其“0 正式更新”的记录，不能说那次是训练拟合失败。
- 后来的两卡 200 更新训练确实完成，约 29.8min、8.94s/update，accuracy 62.48%、STOP recall 0；轻加权后 62.65%，STOP 仍 0，CE 更差，不采用为已验证修复。
- 缓存未证实明显提速；三角算子重写数值失败未采用。不能宣称多卡线性加速或基础 VLN 已完成。
- 训练证据入口：`sft_acceptance/`、`reviews/Q35N_SFT_REPAIR_PLANNING_V1/`、`reviews/Q35N_D1_MAIN_REVIEW_V1/`、`reviews/Q35N_EFFICIENCY_D1_MAIN_REVIEW_V1/`、`reviews/Q35N_ACTION_BALANCE_MAIN_REVIEW_V1/`。

后续应有普通 SFT、同架构同增量数据仅动作监督、强任务状态辅助监督、常规未来事件预测、完整交叉续接及关系删减的匹配对照。先看可学习性，再独立场景 SR/SPL、nDTW/SDTW、恢复、STOP、正常任务退化和实测成本。读出准确率不替代闭环导航收益。

## 5. 训练数据目标与合法计数

权威量级文件：[DUAL_PRODUCTION_TRAINING_SCALE_V2.md](data_pipeline/DUAL_PRODUCTION_TRAINING_SCALE_V2.md)。

| 池 | 正式首档生产目标 | 快速检查点 |
| --- | --- | --- |
| 普通 | ≥3000000 有效指令条件化单步动作监督 | 现有约139万可供下一轮基座训练规划，不是本轮立即训练授权 |
| 特殊 | 10000 去重完整族；180000 交叉结果；另有≥1000000 合法动作监督 | 300、1000族做产率/捷径/学习曲线检查，不把它们当正式终量 |

特殊 1 万族是本项目工程目标，没有同行定理证明它必然足够；混合权重需后续固定协议验证，未决定最佳比例。正式路线目标还要求特殊尽量覆盖 43 适用 FIT 屋、至少 30 屋；现有十屋/22hub不足。

Uni-NaVid 报告 240 万 VLN 导航样本、全部任务 360 万，但“动作序列样本”不等于本项目单步记录或独立轨迹；仅作百万级信息量参照。[原论文](https://arxiv.org/html/2412.06224v2)。其他参考见 [数据规模调研](reviews/Q35N_VLN_DATA_SCALE_SURVEY_V1/REPORT_ZH.md)。

- 普通：source + instruction + 实际 rollout + 因果 prefix + 接口配置去重；分别报告物理路线、物理动作、指令条件动作、房屋与长度分布。
- 特殊：完整族、18格、真实认证重放、合法动作监督、不同hub/房屋分别计。旋转seed、同任务改名、同轨迹复验、重复epoch不增独立量。
- `PROGRESS.physical_families` 是 producer 声明，整批强审核完成前不计正式合格。
- 已合法但任务失败的续接不能全部当动作正示范；unknown、物理错误、服务故障、预算截断不能统一当负标签。
- 只使用冻结 FIT；INTERNAL_DEV、CONFIRM、官方val/test不用于补数量。相关家屋、历史/续接/措辞一起分组留出，不用“新目录”代表未暴露。

## 6. 普通数据资产、完成量与新扩源

### 6.1 已关闭普通生产池

下列均在 `LINE/data_pipeline` 下。累计不含旧99路线pilot、不含未强审的历史截断尾项。

| 池目录 | 严格路线 | 指令条件动作 |
| --- | ---: | ---: |
| `ordinary_scale_v1/recovery_v4` | 770 | 167870 |
| `ordinary_parallel_v1/merge_receipt_v2` | 780 | 167127 |
| `ordinary_fullscale_source_v1/salvage_rescue_v1/run_v1` | 814 | 235410 |
| `ordinary_fullscale_source_v1/runtime_v5/merge` | 496 | 101121 |
| `ordinary_fullscale_source_v1/runtime_v6/merge` | 660 | 192951 |
| `ordinary_fullscale_source_v1/runtime_v4/merge` | 1209 | 290570 |
| `ordinary_fullscale_source_v1/auto_generation_v1/salvage_old_v1/run_v1` | 284 | 68345 |
| `ordinary_fullscale_source_v1/auto_generation_v4/merge` | 836 | 171350 |
| 合计 | **5849** | **1394744** |

最后一个 merge：2509 条指令、57091 独特路线决策、0物理/alias重复；索引SHA `d4d3fd3fdc83988ae49b664da3ef4fdb720165b8305d477c88bd2ea24f3d1746`。合计官方指令17548。

旧已登记 R2R/RxR 英语guide来源8742条已全部尝试：8737终态+5永久partial，未尝试0。核验文件 `ordinary_expansion_v1/COVERAGE_V2.json`；初版 `COVERAGE.json` 漏历史pilot recovery，不能拿它误报的83条重新生成。

### 6.2 当前优先扩源：官方连续环境 EnvDrop

目录 `data_pipeline/ordinary_expansion_v1/envdrop_source_v1`。

- 官方146304条，FIT118913，排除旧GT/重复189，得到 **118724条新来源、50屋**。
- 首波冻结 **20000条、20000条官方合成英语指令**，21片：3条sentinel + 19×1000 + 997。
- 官方GT动作总参考1228720，中位59、四分位47/73、范围8–266。与现有1394744相加也只有参考2623464；实际严格产量还会减少，因此首波**不能承诺达到300万**。另98724来源已在本地，无需重下。
- 选择是“房屋轮转，屋内物理hash排序”，不是长度分层；不要伪称已按长度分层，也不要实验后换成功sentinel。
- sentinel固定3屋：17DRP5sb8fy、1LXtFkjw3qL、1pXnuDYAj8r；3/3真实双回放+独立strict通过才允许自动后续。
- 保留官方原指令/start/goals/reference_path/GT。标签为官方模型生成，不能标成真人描述。
- 源 `JOBS.json` SHA `e1d79e741c6ba63cc7b85f828dfc1d8d30b4d6dcf9dc35064ab75076cf4b118b`；`INPUT_LOCK.json` SHA `3b1f3a2d38ec1acf58c029bce1934c9a959a19862ce4c2a90599b6722c8d9adf`；`PLAN.json` SHA `c8106f1cc63d1b9f13d2f89a6635948d47e0ecc9a4805993fbe013283d46e6e8`。41/41来源复核通过。
- 排除表是 `PHYSICAL_EXCLUSION.json`；另一个 `EXCLUSION_MANIFEST.json` 的physical_route_keys为空，不能误用为空排除表。
- 官方出处：[VLN-CE数据页](https://jacobkrantz.github.io/vlnce/data)，CC BY-NC-SA 3.0 US及MP3D条款；仅本地研究准入不等于可公开发布全部衍生内容。

已实现并启动：`ordinary_expansion_v1/runtime_v1` 复用已审普通autoV4；生产数据在其兄弟 `envdrop_production_v1/shard_0000..0020`，不可改放runtime内部。GPU6，lane82800s，运输83400s，audit1800s，job85200s，queue86400s；sentinel600s、其余单片4200s，RAM12GiB/ownGPU<4096MiB/总新增512GiB（21×24GiB+metadata8GiB）。尾片预算不足不启动，留账，不假称全部已生成。

运行接收：32项CPU测试通过；12文件封存；595个INPUT项（含200场景资产）；主/实际Habitat解释器审批probe均PASS，worker import验0/1/20三片。`runtime_v1/INPUT_LOCK.json` SHA `e9bdb5a75972ddac68fec6b04a0728f892a27a1f26684c5d887ac61b2df309ac`。源码封存，不能原地修复。

实际入口为 `AUTO/ordinary_envdrop_gpu6_v1/PLAN.json`，tmux `q35n_envdrop_gpu6_v1`，queue PID3584930、supervisor3584932（仅启动快照）。生产授权 `authorizations/ORDINARY_ENVDROP_GPU6_PRODUCTION_V1.json`，精确审批 `runtime_v1/MAIN_AGENT_APPROVAL_GPU6.json`。整lane只借还一次GPU6；尚在使用，不应提前恢复占位。

真实sentinel凭据：`runtime_v1/lanes/gpu_6/attempt_000/SENTINEL_GATE.json`，固定3/3严格通过、3指令/167路线决策，已自动进入shard1。不能把官方合成指令的物理/接口验收说成完整自然语言语义认证。余下分片继续生产、终态后自动CPU合并；先前5849/1394744合计不含这些新数据。

### 6.3 已下载但不能直接投产的另外两条来源

- `follower_manifest_v2`：英语FIT16495记录，严格原语义/起点/目标/路径覆盖筛选后97候选。多数follower goals与guide不一致，不擅自重绑指令。不是百万量扩源主路。
- `marky_manifest_v1`：官方Marky-Matterport共有1001331三语行；英语333777，FIT内283461不同path+heading、50屋。指令是官方模型生成，CE坐标/heading转换尚未验收，不硬减固定相机高度。仅作后续扩源。
- 官方connectivity固定commit `589d091b111333f9e9f9d6cfd021b2eb68435925`；51FIT图成功，额外R2R JSON404，因此其总RESULT仍false，不能涂改PASS。
- 下载收据：`ordinary_expansion_v1/acquisition_v1`（TLS失败0bytes）、`acquisition_v2`（407失败0bytes）、`acquisition_v3`（follower成功）、`marky_acquisition_v1`、`connectivity_acquisition_v1`、`envdrop_acquisition_v1`。累计554821653 bytes，低于2GiB，不含读取val/test成员正文。

## 7. 特殊数据：六卡当前运行和新hub供给

以下简写 `WF = LINE/data_pipeline/mechanism_runtime_v1/witness_first_v1`；`BE = WF/batch_execution_v1`；`AUTO = LINE/data_pipeline/auto_production_v1`。

### 7.1 已实际验证自动强审/接续的GPU1/2队列

| GPU | tmux | AUTO计划 | 首批/后续 | queue PID（仅快照，必须复核） |
| --- | --- | --- | --- | --- |
| 1 | `q35n_scale_special_gpu1` | `special_gpu1_scale_v2/PLAN.json` | batch220…228，9批/27候选 | 3555171 |
| 2 | `q35n_scale_special_gpu2` | `special_gpu2_scale_v2/PLAN.json` | batch240…247，8批/24候选 | 3559124 |

10:16快照：batch220/221/240均强审3/3；batch241强审2/3，另一候选缺完整族证据，照实拒绝、不补标签。自动进入batch222（supervisor3586538）/242（3582118）。新波11强审族；此前指定旧队列14族另记（不是全项目历史总量）。两卡非独占，无借用holder，不恢复不存在的占位；外部context不动。

17批均CPU prepare/逐批深核通过，最大1173锁文件；仅迁移旧GPU1 batch203…211、旧GPU2 batch104…111的**从未启动预约**，旧batch202/103的失败/partial不重试，旧预约也绝不能再次启动。

封存运行模块：`WF/special_scale_transport_v1`，8文件封存/17CPU测试通过，`SHA256SUMS`文件SHA `82a965f3133d840372063a901f8eed585a88b059bf6eafce8dc78969d6110d39`。**禁止修改该目录源码，也禁止新增.py**；动态dependency glob会把新文件加入必需锁而导致活跃队列失败。

强审核输出：`WF/quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/special_scale_transport_v1/batch_NNN/result.json`。合格grade明确为 `QUALITY_VERIFIED_FIT_AMENDED_ACTIVE_ACCOUNTING_NOT_MODEL_GAIN`，不替换旧cohort。

### 7.2 已挂GPU3/4/5/7的342新候选

源：`WF/auto_scaleout_v1/queue_v1/QUEUE.json`及`GPU_3_QUEUE.json`、`GPU_4_QUEUE.json`、`GPU_5_QUEUE.json`、`GPU_7_QUEUE.json`。

- 四个真实银行442候选，排除历史/所有已冻结预约99，343未预约中342能组成三不同hub批，1个余项留账。绝不凑重复。
- 114批，GPU3/4各29批、GPU5/7各28批，batch300开始按四卡轮转；每批3候选。来源仍仅10屋22hub，不是新增场景。
- 每lane12h，单批原supervisor3900s/factory3600s/60000动作/7GiB，运输4200+audit1200=5400预留。剩余总时间不够预留不启动尾批。**注册114批不等于12h内必完成全部**。
- 主agent复跑15CPU测试、核114快照来源并集2042哈希全部通过；每批来源最高1647文件/3093236815bytes，不删来源来过审。
- `WF/special_scale_holder_v1` 已封存7文件，36项CPU测试主复跑PASS；独立复核另20项四GPU原XML资源反例和7项源码封存PASS。`SHA256SUMS`文件SHA `ea91ec3493ed6cbc2a74e5a61a301642797b779e6831a0f850ddde6d7ed50b0a`。114批均实际 `check_inputs(worker=True)` 通过，最大1695锁项<2048。
- 首holder来自主冻结身份；后续只能从**该GPU立即前批**成功RESTORATION取新PID/starttime，并绑定前批锁/授权/cleanup，同命令/cwd/pane/env；不能按进程名重新发现或跨失败批跳接。

10:14:48–49 CST，主agent逐计划验证全部job输入hash及精确审批后，已启动四条独立队列：

| GPU | tmux | AUTO目录 | 批次范围（步长4） | queue / 首批supervisor PID |
| --- | --- | --- | --- | --- |
| 3 | `q35n_scale_special_gpu3` | `special_holder_gpu3_v1` | 300…412，29批 | 3593571 / 3593572 |
| 4 | `q35n_scale_special_gpu4` | `special_holder_gpu4_v1` | 301…413，29批 | 3593574 / 3593577 |
| 5 | `q35n_scale_special_gpu5` | `special_holder_gpu5_v1` | 302…410，28批 | 3593579 / 3593581 |
| 7 | `q35n_scale_special_gpu7` | `special_holder_gpu7_v1` | 303…411，28批 | 3593588 / 3593620 |

所有单批主审批在 `BE/batch_N/MAIN_AGENT_SCALE_APPROVAL.json`；队列审批与PLAN同目录。四卡均已实际写入PROGRESS/真实trace，不只是tmux存在。CPU首/后继身份链测试通过不代表实际恢复全已发生：首批未闭合前，实际链式还卡接续与GPU4恢复环境读回仍是待验事项。当前占位被原lease精确借用，结束由各批finally恢复，新会话不要手动占回仍在造数的卡。

### 7.3 下一阶段真正扩大特殊多样性的供给

`WF/new_hub_scale_v1/snapshot_v1/SCOUT_QUEUE.json`：43个适用FIT屋，31屋此前未scout，172个官方R2R路径新3D位置（每屋4），1631个来源引用；到旧hub/同屋本批位置最小2.1487m，原阈值仍1m。12CPU测试、来源/产物seal已通过。

独立适配 `WF/new_hub_scale_runtime_v1` 已完成CPU准备：25项测试、9源码seal和95来源锁项通过；`SOURCE_LOCK.json` SHA `a1a0ced74aec25cd29c0fb2ed6c76e368d3a10344c14168168f5edae223b6284`。它仍是**待主agent运行审查/精确授权/逐job准备**，没有GPU scout实际产物。

下一步使用原每屋40000动作/2700worker/3000supervisor→自然closed/内容与四个位置终态核验→CPU银行→新完整族认证。新执行配置明确记录实际3000s/7GiB/4位置，不改旧source里过时4500s/8GiB/2位置。单job运输3300+audit1800=5100s；GPU7与cert不能并发，使用 `special_scale_holder_v1/locks/gpu_7.lock` 同锁，须显式排期与新的真实RESTORATION身份接续。

主agent已明确允许**新版本银行**纳入实际成功1–3/4 hub子集，但逐位置须有成功/拒绝终态，无补采替换、无资源截断冒充自然完成；每hub组件和最终族质量阈值不变。旧4/4协议/结果不改，不把覆盖数量修订当科学质量放宽。

## 8. 质量门槛与必须保留的失败

特殊：3历史×3续接×2任务=18格，27次真实认证重放/54求值；数值共同状态重建须显式有界，原汇合容差≤1e-5，可见事件256pixels/2frames等依冻结原checker。F/L/R计数匹配、强M2重算、查询与未来信息隔离、来源/phase绑定、自然闭合后的内容哈希核验都保留。当前控制类型包含已完成子目标revisit的位置对照，不能夸称一切无事件无关绕行不变性已证实。

历史记录关键点：

- 原早期单族仅interface_only，长度/步数与强M2 oracle可解释标签，不拿拟合当创新收益。
- 原四成员cohort含batch06r1资源FAIL，后来的06r2/07/08/100…不能替换成员刷PASS。旧批有恢复grade和正常强审核grade，分开计，不合并为泛化。
- 旧自动特殊：100/101/102各3PASS，200=3PASS、201=2PASS，共14；202/103因MEMORY_ACCOUNTING停止且已清理。中断批保存的producer族不直接计合格。
- 普通旧queue_v1/v2因精确审批dict不匹配停止；v3/runtimeautoV2发生嵌套数据扫描FileNotFoundError，60终态+1partial；v4/runtimeautoV3因显存计量不一致停止，0终态+1partial；旧记录不变，未审核60终态不进入正式合计。
- 普通autoV4/queue_v5已正常完成两片，1362终态、836严格合格，GPU6恢复；它解决的是工程中断，不是导航算法正收益。

### 显存计量修复的精确含义

原失败是设备总量偶尔小于进程项之和。普通实际失败样本device733MiB/process sum737MiB，未超4096MiB。原始证据在普通autoV3的`GPU_SNAPSHOTS.jsonl`与`SUPERVISOR_EXCEPTION.json`。特殊旧202/103触发值未先持久化，不能拿前次样本编造其数值。

新版只在活动本worker存在、原guard唯一错误为计量一致性时，使用`max(device_total, process_total)`保守计费，且不一致时gross<4096、external每进程≤768/合计≤2048。idle/final/restore仍用原guard；不同错误继续拒绝。

特殊新版先fsync原XML→原parser→单独PENDING/ACCEPTED决策，`ACTIVE_ACCOUNTING.jsonl`与`RESOURCE_SAMPLES.jsonl`逐索引/数值/时间绑定；强审核独立重算。日志失败即停止，不修改raw或把新规则接受谎称旧规则PASS。

## 9. GPU身份、预算、锁和停止方式

主授权：[DATA_PRODUCTION_SCALEOUT_V1.json](authorizations/DATA_PRODUCTION_SCALEOUT_V1.json)，特殊6卡各12h，最多131批/393候选/1024GiB工程上限。393=342新+51迁移，不含任何已尝试重跑。

冻结占位身份：[DATA_SCALEOUT_HOLDER_IDENTITIES_V1.json](authorizations/DATA_SCALEOUT_HOLDER_IDENTITIES_V1.json)。以下是本波借用前快照；当前3–7正被生成任务使用，**不能把这些旧PID作为后续操作对象**：

| GPU | PID | starttime_ticks | tmux pane / target |
| --- | ---: | ---: | --- |
| 3 | 3330799 | 132486942 | %153 / vla_idle_occupancy_20260904:0.0 |
| 4 | 3376327 | 133119357 | %149 / vla_idle_occupancy_20260904:1.0 |
| 5 | 3344021 | 132602274 | %235 / vla_idle_occupancy_20260904:2.0 |
| 6 | 3529991 | 135434836 | %151 / vla_idle_occupancy_20260904:3.0 |
| 7 | 3354835 | 132743404 | %152 / vla_idle_occupancy_20260904:4.0 |

GPU4有原项目cache环境须完整恢复；GPU3占位argv的max-initial-used-mib为1536，其他为1024，不能统一重写。原脚本`ROOT/scripts/occupy_idle_gpu.py` SHA `1d2d613dfd63e31c60db9176f1203124e0f257f83e5a3bb99e6dbdf3f19f7090`。

普通独立身份：[ORDINARY_ENVDROP_GPU6_IDENTITY_V1.json](authorizations/ORDINARY_ENVDROP_GPU6_IDENTITY_V1.json)，避免后续修改综合身份文件污染普通输入锁。EnvDrop独立AUTH/最终启动审批已建立并实际通过，见6.2。

通用AUTO `queue.py` SHA `e606eab59bb654bb31583333cfa2367780f908be70356594f97c9eeb27422683`；主19测试+独立8安全测试已通过。perGPU flock由child继承；同GPU互斥；SIGTERM/INT给queue只请求排空当前有界child再停，不立即杀GPU。不要杀tmux整会话代替受控清理，也不要对未知PID/进程名pkill。

总队列wall≤86400、单PLAN≤64jobs；114批须分成每GPU独立PLAN。每batch完整input lock+精确主审批+生产receipt+CPU审核；失败停止本lane、无自动重试。活跃时不要运行会争抢同GPU flock的queue测试。

**禁止把来源文件修改后重新算hash覆盖旧lock。** 任何修复新版本/新namespace；旧失败、partial、原cohort、下载失败都保留。普通生产数据与runtime metadata必须sibling，避免原子PNG改名/重复disk计费故障再现。

## 10. 新session第一轮只读接管命令

先阅读ROOT/AGENTS.md、ROOT/research/EXECUTION_RULES.md、LINE/AGENTS.md与本文件；实际操作再读相应冻结协议。不要全盘递归扫描庞大的content目录。

```bash
cd /mnt/data_nas/deeprobotics/daiyang/vla
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/data_pipeline/auto_production_v1/status.py
tmux list-sessions -F '#{session_name}'
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_id} #{pane_pid} #{pane_dead} #{pane_current_command}'
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
```

运行接口：stdlib用上述ROOT本地python；Habitat实际回放用`LINE/.envs/q35n_habitat_v017_g0r/bin/python3 -I -B`。不要给Habitat加`-S`屏蔽所需包。ROOT框架检查命令`bash scripts/research.sh check`不替代局部实际接口/资源/质量验收。

先查活跃PLAN/IDENTITY/EVENTS、producer PROGRESS年龄、worker.log、SUPERVISOR/LAUNCH/RESTORATION；queue活着不代表仍在生成，audit阶段不占GPU，旧100%可能是恢复占位。不要仅凭旧PID或shell进程存在推断状态。

目前 `status.py` 只识别普通旧 `output_roots`，EnvDrop配置使用 `output_namespace`，所以它的普通 `production_progress: []` **不是没生成**。必须另读 `ordinary_expansion_v1/envdrop_production_v1/shard_*/PROGRESS.json`（仅这一层glob，不全盘递归）；sentinel与最终强审/合并receipt另核。它还可能按多个completion重复显示同一个特殊PROGRESS路径，必须按path去重，不累计三遍。不要为修显示修改封存runtime。

普通实时只读补充命令（已在本轮实际使用）：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B - <<'PY'
import json
from pathlib import Path
base = Path('projects/qwen35_indoor_nav/data_pipeline/ordinary_expansion_v1/envdrop_production_v1')
for path in sorted(base.glob('shard_*/PROGRESS.json')):
    d = json.loads(path.read_text())
    print(path.parent.name, {k: d.get(k) for k in ('completed', 'target', 'audited_routes', 'audited_instruction_conditioned_decisions', 'failure_counts')})
PY
```

新启动必须先完成真实主解释器及实际worker解释器的审批函数调用。普通旧common.approved曾要求精确五字段dict，多一个说明字段也失败；别只测parent import后认为worker准入已通。

## 11. 明确优先级与下一步验收

1. 保持七条实际队列，GPU1/2新raw→decision→RESOURCE强审核及自动接续已在多批实证通过，不重跑旧批。
2. 观察新增GPU3/4/5/7首批自然关闭，核强审、实际RESTORATION及下一批精确身份接续；GPU4缓存环境用白名单字段实际读回核对。四lane114批已全准备/主签/启动，不再重复安装或重建。若某lane失败，只读定位、保存结果，新版本再准入，不跳失败前驱。
3. EnvDrop首2万路线已经运行且固定sentinel3/3通过；保持GPU6自动分片与末端合并，核真实合格量/磁盘预算/尾片是否获足额时间，不能把整2万来源数填成完成数。
4. 新hub采集与银行流水线补齐，跨43屋扩大物理多样性；GPU7和cert显式错峰，不能共用同卡双producer。来源不足时扩真实覆盖，不靠semantic rename凑万族。
5. 周期报告“已预约/实际尝试/完整物理认证/强审合格/不同hub房屋/合法动作监督/失败原因与成本”，不是只报GPU利用率。按实际合格率规划后续EnvDrop来源，补到300万。
6. 数据量达检查点后另冻结训练协议：普通大数据基座、STOP/动作类表现、长历史可学习性、相同数据强M2等对照、独立场景闭环导航；当前造数不自动授权训练。

验收目标是**大队列实际持续写出可审计数据、自动质量审核/接续、资源受控并可恢复**。CPU计划、下载完成、producer族声明、单屋成功都不是量产终点。

## 12. 可直接复制给新Codex的完整Prompt

```text
你现在是此项目新的唯一主agent，接管既有项目，不从零重开方向。

唯一项目根：/mnt/data_nas/deeprobotics/daiyang/vla。
当前独立路线：/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav。
先完整阅读以下交接文件和其中要求的AGENTS/执行规则：
/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/PROJECT_TRANSFER_20260910_ZH.md

先只读复核实时tmux、queue EVENTS/RESULT、producer PROGRESS、GPU与占位精确身份，明确当前在跑什么、哪些已关、哪些只是草稿。交接文档的最终快照优先于旧README/STATUS里过时的“最新”段落。不要重复启动现有队列，不要修改活跃或冻结源码/输入锁，也不要新增.py到使用动态依赖glob的封存目录。

我的目标：基于Qwen3.5-2B自建通用室内VLN，研究主线是交叉续接数据编译与实际运行执行记忆联合学习，服务未来轮足式居家医疗养老导航底座。要求学术诚信、明确贡献、好故事和最终有竞争力的闭环结果。MAINLINE_FREEZE_V3保持不变，不回到A/B/C/VLFM的小组件路线，不无理由重开全面文献检索，不声称CVPR或全局创新已被保证。

立即工作重点是把普通/特殊数据自动量产扩起来，不是再写一轮长方案：
普通至少300万有效指令条件动作；特殊工程目标1万去重完整族、18万交叉结果，另统计至少100万合法动作，尽量覆盖43个适用FIT屋/至少30屋。300/1000族是检查点，不是终点。重复epoch、重复轨迹、改ID/换说法不增加独立数据。

沿交接的真实完成状态执行：GPU1/2特殊队列已多批强审/自动接续；GPU3/4/5/7的114批342候选已封存准备、逐批主签并开始实际生产；GPU6官方EnvDrop首2万路线已启动且固定3条sentinel真实强审3/3通过。维持七条队列，不重建，不手工重启。先核新增四卡首批最终强审/真实占位恢复及后继身份链，再接手已CPU实现但尚未GPU准入的新hub采集→CPU银行→完整族审核，使供给不只限十屋22hub；GPU7与现有cert必须同锁错峰。根据实际合格产量安排本地余下98724 EnvDrop来源的后续有界波次补到300万。允许独立CPU开发/审核分给子agent，主agent独占GPU启动/恢复/共享状态，文件范围分清。

MP3D有合法授权；其他内容只在项目根内。GPU0和所有真实外部任务不碰；后部卡只有精确核实为占位才借用，用完恢复同argv/cwd/env。已明确用户允许利用这些卡扩量、可尝试proxyon；按现存有界授权和每个新运行的具体预算/输入锁执行，不无限下载、不自动启动训练。

保留所有历史失败和partial，不放宽数据质量/未来信息隔离/18格27重放54求值来凑数。明确区分原始观测、显存协议修订接受、严格数据质量和模型收益。新数据源若坐标/语义/许可未验收不得硬上；优先已完成CE迁移的官方EnvDrop，不把Marky离散路径直接当可回放轨迹。

第一条回复简要说明核实后的实际状态和下一项具体执行，然后在安全、已授权范围内直接做。每个阶段报告真实产出、合格量、失败与待办，不只报告准备完成。不要为了迁移会话停止健康的后台生成任务。
```

## 13. 最终交接快照

实读时间：**2026-09-10 10:18:56 CST**。这是有时间戳的启动后快照，不是生产结束汇总；进程随后继续运行。

| GPU | 正在执行 | 已观测实际进度 | 接管时首要复核 |
| --- | --- | --- | --- |
| 1 | batch222 | 64完整trace、20738动作、0碰撞；producer声明1族 | 批终态后强审，不把暂存声明计正式量 |
| 2 | batch242 | 113完整trace、21740动作、0碰撞；producer声明2族 | 同上，随后应自动243 |
| 3 | batch300 | 40完整trace、5672动作、0碰撞 | 首次真实还卡、强审、自动304 |
| 4 | batch301 | 27完整trace、6316动作、0碰撞 | 首次真实还卡及原cache环境、强审、自动305 |
| 5 | batch302 | 26完整trace、6272动作、0碰撞 | 首次真实还卡、强审、自动306 |
| 7 | batch303 | 24完整trace、3351动作、0碰撞 | 首次真实还卡、强审、自动307；不同时开scout |
| 6 | EnvDrop shard1/21片 | sentinel3/3终审通过；shard1已尝试593/1000、滚动审核543路线/26225动作 | 本片终审与自动shard2，最后CPU合并及整lane还卡 |

- 七条queue均按实际 `/proc` 的queue命令核实存活，当前进度持续更新；新增四卡启动后多次读到动作/trace增长。四个 `CHAIN_IDENTITY_BINDING.json` 与 `LEASE_ACTIVE.json` 证实借用的是冻结精确占位，`holder_exited=true`，未按名称泛杀。
- 普通新波尝试总596（sentinel3+shard1的593）；目前滚动审核合计546路线/26392动作，**尚未进入最终merge累计**。shard1出现2条REFERENCE_CORRIDOR_DEVIATION、5条COLLISION、1条INVALID_OBSERVATION，保留逐路线失败；未重绑标签或把失败改PASS。
- 已闭合旧普通合计仍5849严格路线/17548指令/1394744指令条件动作。不要把新波来源20000条或实时滚动统计直接填入这个闭合合计。
- 特殊本波四批强审：220=3、221=3、240=3、241=2，共11个不同attempt_id、5屋/8hub；旧指定自动队列14族分开登记。其他历史族的全项目最终去重汇总仍待做，不将25误称完整历史总数。
- 本波特殊总预约131批/393候选，包含17批51个未尝试迁移候选及114批342新候选。它们已经挂为6条有界队列，但12h内是否全部完成取决于实际时长和5400s尾批预留，不允许擅自无限延长或重跑失败。
- 普通源第一波2万不足以保证总300万；后续仍有98724条本地未选EnvDrop来源。特殊新供给仍须在现有cert之后安排43屋/172位置scout，源和runtime仅CPU准备，不能报成已新增43屋的完整族。
- 四卡holder封存最后复核7/7通过；文档9个本地Markdown引用逐个检查存在。EnvDrop的12个.py使用 `INPUT_LOCK.json` 封存，并没有单独 `runtime_v1/SHA256SUMS`，不要因找不到该文件误判没有锁。
- 本轮CPU准备/审核子agent已交付、无遗留准备进程；它们未操作GPU。健康tmux生产任务保留给新主agent，无需旧会话继续在线。当前会话不另行并行争抢主控制权。

**完成边界：本轮已完成自动生产扩容的准备、逐队列准入与七卡真实启动，以及单文档完整移交；未完成百万级/万族终量，也未通过模型收益验证。新主agent从观察七条现有队列和新增四卡首个完整借还周期开始，随后接续扩源，不从零搭建、不再启动同一批。**
