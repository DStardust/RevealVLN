# 给 OpenCode：闭环导航评测、特殊数据融合与双路造数接续

日期：2026-09-11。状态：**只读核实后的执行指导，尚未实施本文件中的新评测、融合训练或造数重启。**

用户最新分工：OpenCode 继续负责训练和代码执行；本会话负责核实、判断和交接，不修改活跃训练源码，不操作其进程。本文件承接 9 月 10 日交接，但运行状态以本次快照及执行前复核为准。

ROOT = `/mnt/data_nas/deeprobotics/daiyang/vla`；LINE = `ROOT/projects/qwen35_indoor_nav`。下文相对路径均相对 LINE，标 ROOT 的除外。

## 1. 先说结论

下一步不是继续只盯 STOP 召回率，而是回答三个实际问题：

1. **现在的模型究竟能不能自己走完导航任务？** 接 R2R-CE 闭环，先检查动作接口，再测到达、停止、路线跟随和碰撞。
2. **特殊数据能否帮助模型处理“之前已经做过什么”？** 先做正式数据准入与普通动作融合；长期历史敏感任务另接实际运行记忆，保留同数据、同架构的动作监督对照。
3. **数据是否真的还在增加？** 当前造数已经停下。普通队列需要处理中断片与未执行部分；特殊队列需要接新 hub 供给，不能只是反复使用已有房屋和位置。

这三件事可以分工推进，不必等普通数据达到 300 万或特殊数据达到 1 万族才开始有界评测。反过来，训练曲线变好也不代表已通过导航 bench 或论文机制验证。

## 2. 当前到底做到哪里了

只读实测窗口 09:29–09:48 CST；下面训练数字取 **09:48:23** 快照。完整数值、检查点 hash、逐审核回执及导出 manifest hash 见 [只读证据快照](reports/opencode_navbench_fusion_20260911/LIVE_READONLY_SNAPSHOT.json)。这是回执盘点，不是本轮重新执行全部图像、物理和强审核。

| 工作 | 实际状态 | 还不能说什么 |
| --- | --- | --- |
| 普通基座训练 | `ordinary_baseline_v3/formal/run_0001`，GPU3/4/5 继续运行，14,285 次更新；最近落盘 checkpoint 为 14,200 | 尚无本轮闭环成功率，不是模型已经训练完 |
| 训练吞吐 | 最近差分约 141.5 decisions/s | 原始 throughput 字段约 221.5，分子带恢复前累计量，不能直接用它算加速倍数 |
| STOP 学习 | 最新变化样本窗 7/19，召回 36.8%；预测 STOP 23 次，精确率 30.4% | 样本少且窗口变化；09:41 曾为 12/23，52.2%，不能把单窗上升当稳定收益 |
| 现用普通训练池 | 5,849 路线、17,548 指令、1,394,744 指令条件决策/epoch，51 FIT 屋 | 不含新增 EnvDrop，也不含特殊族；不是同数的独立物理轨迹 |
| EnvDrop 首波 | 来源计划 20,000 路线；滚动终态计数 10,338，滚动审核 10,098 路线、634,414 动作；第 11 片中断 | 无整 lane 最终 merge，不能把滚动量直接纳入正式训练池 |
| 特殊四组审核回执 | 67 批、201 个登记候选，178 个不同合格 attempt_id，11 屋、24 个精确 pose hub，3,204 格交叉结果 | 不是全历史最终物理去重总数；24 是精确位置键计数，近邻 hub 仍需空间去重 |
| 特殊可用监督候选 | 178 个 export_v4 manifest 均存在；族内 CE owner 数加总 161,936 | 尚非跨族去重后的合法训练动作数；导出仍 `training_admission=false` |
| 新 hub 扩屋采集 | 43 FIT 屋、172 位置有来源和 CPU 实现；`jobs_prepared=0`、新物理 hub=0 | 尚未进行这波 GPU scout，不可报告成新增 43 屋特殊数据 |

当前训练累计计费约 1,049,144 个决策，但经历恢复和并行配置变化，不能把该计数直接当成唯一样本覆盖率。覆盖需按样本键与实际访问记录另核。

特殊合格数的回执范围明确为：

| 审核组，均在 `WF/quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/` | 批数 | 合格 ID |
| --- | ---: | ---: |
| `auto_generation_gpu1_v1` | 2 | 5 |
| `auto_generation_v1` | 3 | 9 |
| `special_scale_holder_v1` | 45 | 117 |
| `special_scale_transport_v1` | 17 | 47 |
| 本表 ID 并集 | 67 | 178 |

其中 14 个为原强审核 grade，164 个为显存计量协议修订后的强审核 grade，分别留账；全部都不是模型收益。WF 下文表示 `data_pipeline/mechanism_runtime_v1/witness_first_v1`。

### 两个需要 OpenCode 优先处理的工程问题

**P0：epoch 边界可能空转。** 本次实读 `sft_acceptance/ordinary_baseline_v3/train.py`，`while cursor['epoch'] < protocol['epochs']` 内只推进 `position/updates/decisions`，一个 shard 迭代完后没有 `epoch += 1; position = 0`。据此推断：到首轮末尾可能不断创建空 loader，且预算检查位于 batch 内，空转时不会执行。当前仍有更新，不是已经发生空转的结论。

OpenCode 应在独立版本中增加“空 shard、epoch 切换、最后一个 batch、边界恢复、不同 world size、终止同步”的 CPU 测试，核实累计预算和学习率步数。安排自己管理的训练在安全 checkpoint 边界受控换版；不要直接热改正在运行或已封存文件，也不要重启清零累计成本。不能单改日志让 epoch 看起来推进。

**P0：旧面板绑定不一致。** `ordinary_baseline_v3/PANEL_EVAL.json` 绑定的 model.py hash 与当前 model.py 不一致；旧 eval.py 会拒绝。保留旧锁，在独立评测版本中重新登记实际代码/检查点/数据绑定。不能覆盖旧 hash 冒充原评测协议通过。

当前 model.py SHA：`e539c9ec1c268fbb4ce5e88acaef18df5489ec67bfaeb12547029d8aea390aa8`。

当前 train.py SHA：`52d39af3c28b7b5473208e097543595b8a75a92f8adea637ff0af75371b7e07c`。

## 3. 第一条线：接实际导航 bench

### 3.1 先选 R2R-CE，随后再扩 RxR-CE

理由是本项目已经有相应普通训练来源、MP3D 合法资产和 Habitat-Sim 0.1.7 环境；先把同任务的闭环链路接通，避免同时更换任务、传感器和动作空间。官方仓库支持 R2R 与 RxR，提供评测入口及任务配置。[VLN-CE 官方仓库](https://github.com/jacobkrantz/VLN-CE)

近期不用 ObjectNav 替代路线导航验收。语言目标寻找仍是后续统一模型验证项；R2R 成功也不能直接声称会找任意物体、老人或病人。RxR 英语扩展与正式多语 challenge 分开，不能用英语子集冒充完整挑战赛。

### 3.2 现有闭环脚本只可作接口参考

参考 `sft_acceptance/v1/recovery_r1/sim_server.py`，**不要重跑或改写这个旧节点**。建议新目录 `closed_loop_bench/r2r_ce_v1/`。

| 项目 | 旧项目脚本 | 新接入要求 |
| --- | --- | --- |
| SR/SPL | `official_sr=None`、`official_spl=None` | 使用固定版本官方 evaluator；否则逐项等价测试并明确标自定义评测 |
| nDTW | geodesic 距离 + episode reference_path，自写 DTW | 官方代码使用 GT locations、欧氏距离、去除连续重复位置，且有 DTW/FastDTW 配置；不得混报 |
| 动作预算 | 512 个 motion 后才处理终止 | 锁定官方任务的 500-step 口径，含 STOP 的计步/截断行为按 evaluator 测试 |
| Sliding | `allow_sliding=False` | 官方该版 R2R 配置为 True；公开协议结果按官方设置，训练同配置 False 的检查另报 |
| 观测/动作 | 224 RGB，0.25m、15° | 固定相机高度、FOV、预处理和动作；保持 RGB-only 策略，不暗加深度、目标向量或 oracle action |

上述官方配置不是所有后续论文的统一设定；执行前保存 commit、配置 hash、版本和差异表，再决定哪些数字可直接比较。[官方 R2R 任务配置](https://raw.githubusercontent.com/jacobkrantz/VLN-CE/master/habitat_extensions/config/vlnce_task.yaml)、[官方路径指标实现](https://raw.githubusercontent.com/jacobkrantz/VLN-CE/master/habitat_extensions/measures.py)

部署动作映射应显式写出并测试：**模型输出 `[F,L,R,STOP]` 的 index 0/1/2/3，不是 Habitat `[STOP,F,L,R]` 的 0/1/2/3**。按动作名字映射，不把 logits.argmax 直接交给模拟器。

### 3.3 实现清单：先接口、后分数

1. Qwen 推理沿用当前 v3 的模型、processor、token、action head、FLA 路径与检查点，不误载旧 v6 的记忆结构。只读完整 checkpoint，校验其 receipt/hash；不要追着不断变化的 latest 路径跑。
2. 将 Habitat executor 与 Qwen worker 分进程，分别使用项目内 Habitat/Qwen 环境。复用经过核验的进程通信思路，不在共享环境升级包；新依赖确需获取时独立目录登记官方来源、许可和预算。
3. 策略只接收原指令、截止当前的 RGB 和实际已执行动作。每个 episode 重置短窗/缓存；未来 RGB、GT action、scene/episode ID、地图、目标坐标和评测距离均不进入模型。评测器可以读真值，但不能回流到动作决策。
4. 单步贪心四动作先跑通；不临时加目标距离自动 STOP、到终点 teleport、oracle 回退或偷偷规划。失败轨迹原样记录。
5. CPU/接口反例至少覆盖：四动作映射、四元数顺序、初始 pose/RGB、碰撞后的实际动作历史、STOP 后无后续动作、500-step 边界、跨 episode reset、未来/ID 字段扰动不影响当前策略、checkpoint 完整性。
6. 数值指标回归至少覆盖：到目标但不 STOP、远处 STOP、成功 STOP、零位移、重复位置、不可达目标、服务错误。对同一轨迹比较官方与封装输出；无法比较的指标设 null，不用另一算法顶替。

官方 Success 要求调用 STOP 且目标距离低于阈值；SPL 还惩罚实际路程。R2R 配置阈值是 3m。因此训练轨迹中“最后一帧的 STOP 召回”不是闭环成功率。[Habitat 0.1.7 Success/SPL 实现](https://raw.githubusercontent.com/facebookresearch/habitat-lab/v0.1.7/habitat/tasks/nav/nav.py)

### 3.4 建议的第一波评测预算

以下是建议执行批次，需由 OpenCode 在看新分数前冻结完整 episode 清单、代码/资产 hash、随机种子和资源协议；不是本会话已启动任务。

| 阶段 | 数据与检查点 | 上限与交付 |
| --- | --- | --- |
| E0 接口闭环 | 固定 10 条 FIT 物理路线，各 1 条指令，checkpoint 12,000 | ≤5,000 环境动作、≤1,800s；逐 episode 轨迹/终态，纯工程检查 |
| E1 内部开发 | 现有 INTERNAL_DEV 的 5 屋，各预选 10 条不同物理路线；checkpoint 6,000 与 12,000 配对 | ≤100 episode、≤50,000 动作、≤7,200s；配对 SR/SPL/STOP/路径表现 |
| E2 公开验证 pilot | 官方 R2R-CE val_unseen，事前固定 200 条指令 episode；在 E1 后预先选定一个 checkpoint | ≤100,000 动作、≤14,400s；只能称 val_unseen 子集，不称全量榜单结果 |

优先使用已经存在且本次实算 hash 一致的 checkpoint 12,000：

`sft_acceptance/ordinary_baseline_v3/formal/run_0001/checkpoint_000012000.pt`

SHA：`933e55704e64f54903ac04bd0c522d7bd24c991111d31d24ce912f36fafbf3df`。

先固定这个基线，不等“最漂亮的 STOP 窗口”。新的后期 checkpoint 可在另一次预注册批次比较，不对每 200 步都跑完整 bench。512 样本 FIT 面板可作为辅助曲线，不设成开始闭环的高拟合率门槛。

E0/E1 分数不好本身不要求停止所有评测或重新发明模型；先查失败原因。接口不合法、泄漏、资源或服务错误则停止对应批次。预算到期保存未完成状态，不因凑足 episode 无限延长，也不把超时运行的部分样本当完整 E1/E2。

资产现状：ROOT 下 `data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz` 与 `val_seen/val_seen.json.gz` 存在；同一入口未找到 `val_unseen/val_unseen.json.gz`。这不是对全项目其他来源的“不存在”证明。先核已有登记资产与许可；缺少的官方 val_unseen/GT 按有界下载进入 LINE 新资产目录，不借其他项目。GT 文件不齐时不宣称官方 nDTW 已可用。

E1 沿用 `sft_acceptance/ordinary_baseline_v2/snapshot_v1/SPLIT.json`，不是旧 pilot 的 reserved_unassigned。严禁为补样本移动 FIT、INTERNAL_DEV、INTERNAL_CONFIRM 的成员。当前普通基座已经见过 51 个 FIT 屋，未知旧线暴露不能改成“全球从未见过”。官方 val/test 和 INTERNAL_CONFIRM 不进入造数训练池，测试服务器提交不在首轮任务内。

### 3.5 第一张闭环结果表应长什么样

每个 checkpoint 报告：episode 总数/已完成/服务失败/资源截断，SR、SPL、NE、OSR、nDTW、SDTW、碰撞、终止原因、推理 p50/p95 和完整 episode 耗时。

额外把失败拆成四类：从未进过目标范围；进过但没成功停止；在范围外停止；动作预算耗尽。各类定义明确是否互斥，服务失败另记。OSR−SR 仅帮助定位停止/离开目标问题，不证明只改 STOP 就能提高 SR。

按 house 与物理 route 分组报告不确定性；同路线多指令不能当完全独立样本。先汇报计数，不以 10/50 条小样本宣称显著泛化。验证集可以用于声明过的开发选择，不能反复调到高分后当独立确认。

## 4. 第二条线：特殊数据怎样开始融合

### 4.1 先承认当前模型能看到什么

当前 v3 是普通短窗导航基座，不含旧 v6 的长期 writer/8-slot 记忆。特殊族的关键事件会被设计在公共短尾之前；如果两个样本的合法短窗输入相同、正确续接却因历史而不同，单靠这个模型无法辨别。

所以“把特殊样本塞进普通 dataloader”最多是动作数据增强，不能自动实现论文主线。更不能把未来 query 或“已经看过 A”真值文本加进提示，伪造模型拥有记忆。

### 4.2 特殊数据正式准入：独立新 snapshot

建议 `data_snapshots/special_fit_v1/`，只读复用 V4 导出，不向活跃普通 snapshot 追加。

1. 以本次 178 个 ID 的回执清单为明确起点，再另列历史恢复/interface_only 等队列，不默认并入。逐条复核源码/来源、自然关闭、18 格/27 重放/54 求值、query 隔离、强 M2 和资源 grade；producer 自报不作准入。
2. 构造跨批次、重试、语义改名的物理/轨迹/任务条件去重键。数据计数分为：完整物理族、任务语言条件、真实动作、成功 CE owner、交叉 Y、屋/hub；不要用 178×18 或重复 seed 顶替独立数据规模。
3. 当前导出为 `candidate_fit_pool` 且 `training_admission=false`。新写准入 manifest/外部白名单，说明本轮准入依据，不改旧 manifest 或 audit 的 false 字段。
4. `SUPERVISION_ONLY.jsonl` 的有效 pass/fail 才用于 reader BCE；unknown、物理失败、资源中断 mask 掉。动作 CE 仅取成功且满足当前任务的 owner，不训练“任务未完成却 STOP”的失败流。
5. 复用 `data_pipeline/mechanism_runtime_v1/loader.py` 的真实 V4 schema，并参考 `parallel_readiness/v2/loader/README.md` 的因果流和动作去重要求。旧单族 CPU loader 通过不代表新多程序、多屋词表已经兼容。
6. 明确 house/族整体分组，历史、续接、措辞不可拆开。现有特殊数据都来自 FIT；若从这 11 屋留出“特殊监督未训练屋”，普通基座仍见过这些屋，必须这样标注，不能称完整模型 scene-unseen。
7. 真正特殊机制的独立场景评测应在原 INTERNAL_DEV/后续冻结确认组单独生成 **evaluation-only** 族，与 FIT 生产计划、缓存和训练 loader 分开；不把 43 屋 FIT scout 当成未见场景测试。

交付至少包括 `ADMISSION.json`、`SPLIT_AND_EXPOSURE.json`、`FAMILY_INDEX.jsonl`、`ACTION_INDEX.jsonl`、`COUNTS.json`、拒绝/重复清单及 loader 反例测试。先完成这些 CPU 工作，不要求先产生新 GPU 数据。

### 4.3 分两步融合，避免一次改变所有东西

**第一步：小比例动作融合，作为工程/数据对照。** 从同一普通 checkpoint 派生两个新的短续训 run：普通-only 与普通+特殊合法动作。初始建议按有效 CE 决策 90%/10% 抽样，每个 run ≤2,000 更新、≤150,000 有效动作、≤7,200s，取先到者停止；这是起始工程配比，不是已验证最优值。

两组沿用同架构、优化配方、有效动作预算，并记录真实图像 token/重复访问量。特殊按 house→family→owner 控制采样，不能让少数长轨迹占满。先查“同合法短窗输入、互斥语义动作”的观测混叠；这类依赖旧历史的决策留给记忆版，不用删一个标签/改任务来假装冲突解决。记录排除比例，因此短窗试验不是完整机制对照。

只在当前普通训练的安全分段之后由 OpenCode 排程，不抢其 GPU3/4/5。比较 E1 上的正常导航退化、动作偏置、特殊成功续接及代价。小样本结果不确定就如实记录；不要在未核原因前提高特殊比例到 50% 或覆盖普通基座。

**第二步：接入真实运行记忆，检验论文机制。** 新目录建议 `sft_acceptance/execution_memory_v4/`。从当前普通基座初始化可兼容参数，新增有限记忆槽和训练期 reader；这是结构扩展，不声称原优化器对新参数完全续训。

必须实现以下真正的通路：

```text
完整因果历史 + 当前指令 → 运行记忆 → 下一导航动作
                               └→ 训练 reader + 离线续接 query → Y
部署：保留运行记忆，移除 reader；query 永不进入动作通路
```

不要把 G2 synthetic `Embedding(8)+mean` 直接接到真实 query。真实 V4 query 有动作、次数、类别、房间和可见事件等 typed 字段；次数是数值而不是类别 ID，需要保序编码并检查当前多程序类别覆盖。

完整前缀从头运行；每个 task/history 重置记忆，换指令重算。TBPTT/缓存不能截断唯一的“早期可见事件→汇合记忆→损失”学习路径。先选固定少量多屋合格族做实际长前缀梯度、跨样本隔离、记忆置零/交换和未来扰动反例；只有读取完整历史但所有关键步骤 no_grad，不足以证明记忆 writer 学得会。

匹配对照按下表顺序规划，不要求第一天全启动：

| 对照 | 数据与结构 | 回答的问题 |
| --- | --- | --- |
| A：普通 v3 | 原普通数据、短窗 | 当前导航底座参照 |
| B：记忆 + 动作监督 | 与完整方法相同记忆结构、相同普通/特殊合法动作 | 是不是只靠多数据或加记忆就能获得收益？ |
| C：记忆 + 强 M2 辅助监督 | 同数据/预算，已重算的任务状态标签只作训练监督 | 简单而强的任务状态监督是否已经够用？ |
| D：记忆 + 交叉续接 | 同数据/预算，grouped valid BCE 作用于同一动作记忆 | 新的监督关系是否带来独立增益？ |
| 后续必要对照 | 常规未来事件预测、去跨历史、去跨任务 | 区分一般预测收益和具体交叉关系 |

主线保持 `MAINLINE_FREEZE_V3.md` 不变；先在小规模 FIT 注册动作/reader 损失归一化、权重、前缀长度、训练计算预算和停止条件，再运行。BCE 按 family/task/valid cell 分组，动作 CE 按 owner 独立遍历，不能把重复 BCE 抽样次数乘到 CE 上。当前 161,936 只是族内 owner 加总，不可据此宣称已经达到 100 万合法动作目标。

论文判断仍看自然闭环导航、特殊任务严格满足/恢复及成本；reader 准确率仅为诊断。普通方法可借鉴和引用，其本身不需要包装为创新。UAD 仅是时间与证据意识的历史启发；EgoCoT-Bench 不是本导航 bench，也不是训练答案来源。

## 5. 第三条线：普通数据生成恢复与扩源

### 5.1 本次确认的停因和可保留产物

`data_pipeline/ordinary_expansion_v1/runtime_v1/lanes/gpu_6/attempt_000/RESULT.json` 明确为 `AssertionError('SHARD_WALL_BUDGET')`。shard_0011 在 supervisor 发 SIGTERM 后退出；worker log 的 `SUPERVISOR_STOP:15` 是受控中断证据，不应误判为 PNG 写入本身随机坏了。GPU6 已有成功恢复记录。

| 范围 | 滚动终态 | 滚动审核路线 | 滚动审核指令条件动作 | 后续 |
| --- | ---: | ---: | ---: | --- |
| shard_0000…0010 | 10,003 | 9,809 | 619,328 | 有 GENERATION_COMPLETE；逐片独立强审/合并，不重放 |
| shard_0011 | 335 | 289 | 15,086 | 无 GENERATION_COMPLETE；核终态账本、审核尾部与 partial，单独恢复分级 |
| shard_0012…0020 | 本轮未见生产进度 | 未计 | 未计 | 核“确实未尝试”后迁移到新波 |

shard_0011 的 `replay_certified_routes=324` 与 audited=289 也不同。差额不能自动补成 PASS；必须根据质量文件和持久化账本处理。中断片已落盘终态、未审核尾项、正在写的 partial、未开始的路线分别登记，不能仅用 1,000−335 推出全部可重跑数。

### 5.2 OpenCode 的恢复步骤

1. CPU 先建独立 `ordinary_expansion_v1/recovery_inventory_20260911_v1/`：读各片 JOBS/LEDGER/quality/完成标记，不全盘扫描 content。确认无活跃 producer；保留所有原始失败和 partial。
2. 对完整片新建最终 merge receipt；中断片用已有严格恢复审计思想，只承认完整可核验终态，原失败仍存在。不得把整个失败 lane 改成成功。
3. 新 `runtime_v2` / `envdrop_production_v2` 采用 sibling 布局，封存新源码、输入排除表和每片预算。优先只调工程分片粒度：建议每片 200–250 个未尝试物理路线，保留双回放/路线/碰撞/观测阈值，不放宽质量来赶时间。
4. 从原 JOBS 和跨波物理排除表精确生成待办，不改原 20,000 清单。已失败需要重试的任务必须是明确的新 attempt，关联旧失败；不得默认无限自动重试。
5. 首新片有界验收通过后自动下一片；每片结束后强审、累计资源账与唯一 producer 检查，预算不足留下 PENDING，不把尾片强行塞进去。
6. 新数据先形成可复现 snapshot，再安排后续普通训练吸收，当前 snapshot/input lock 完全不变。

普通目标仍是 ≥3,000,000 有效指令条件动作。即便本次全部滚动量最后获准，与旧池加总也只是 2,029,158，仍差 970,842；这只是条件计算，不是已获准总量。首波余项之后按实际严格产率从已落地的 98,724 条余下 EnvDrop 来源安排下一有界波次，不重下载整源。

EnvDrop 指令是官方模型生成，和 R2R/RxR 原人工指令分账、分采样、分别看自然指令验证表现。Marky 的 CE 坐标/heading 接入尚未验收，当前不拿它替代现成 EnvDrop 恢复。

## 6. 特殊数据生成：接续现有待办，同时真正扩屋

### 6.1 不要再启动已经补过的任务

`auto_production_v1/manual_failure_recovery_gpu1_20260910_v1/lane_gpu_1/RESULT.json` 已 CLOSED：batch500…507 全部进入队列审核终态，逐批合格数为 **3、2、2、2、3、3、3、3，共 21**。这些 21 已包含在前述 178 内，不能另加一次。

迁移关系：225→500、244→501、226→502、245→503、227→504、246→505、228→506、247→507。旧 GPU1/2 queue 里的 PENDING 仍保留，但不再是可执行待办；先查迁移账再排程。

### 6.2 四卡旧队列还有 69 批未执行预约

| 原 lane | 已审核批 | PENDING 批数 | 首个待办及序列 |
| --- | ---: | ---: | --- |
| GPU3 | 7 | 22 | 328…412，步长 4 |
| GPU4 | 7 | 22 | 329…413，步长 4 |
| GPU5 | 8 | 20 | 334…410，步长 4 |
| GPU7 | 23 | 5 | 395…411，步长 4 |

共 69 批、207 个预约候选，不是 207 个已合格族。GPU3/4/5 已让给当前训练，不为了复原“七队列”再抢回。正常排空与错误停止分开。

若需要迁移上述未尝试预约到 GPU7 或空闲 GPU1，新建迁移计划/输出 ID/授权，检查旧 batch 从未实际启动、全局排除表未被后来波次使用，保持候选内容不变。不能直接用旧首 holder PID 或跳过失败的身份链；执行前重新验证当前 RESTORATION 与实际身份。

### 6.3 新 hub 供给优先于在同一批旧位置反复造族

现有入口：

- `WF/new_hub_scale_v1/snapshot_v1/SCOUT_QUEUE.json`：43 屋、172 固定位置；
- `WF/new_hub_scale_runtime_v1/SPEC_ZH.md`、`result.json`：25 CPU tests，95 source bindings，但未 GPU 准入、未 prepare jobs。

执行顺序：**scout 实采 → 自然闭合/每位置终态 → CPU 新银行 → 冻结不同 hub 候选 → 真实完整族认证 → 强审核 → 新训练 snapshot**。

优先按既定冻结队列前缀启动首 3 屋有界实采做工程验收，不按采集成功与否替换房屋。沿用每屋 4 点、40,000 动作、2,700 worker 秒/3,000 supervisor 秒、7GiB 总盘的已登记上限；含运输/CPU audit 每 job ≤5,100s。首波最多 3 job，lane ≤15,300s；若原封存 prepare/API 不支持这个受限前缀，独立新适配版本，不能修改旧 SOURCE_LOCK 强行通过。

每屋允许按已登记的新银行规则接收实际成功的 1–4 个 hub，但四位置都须有明确终态；资源截断不能算自然完成，不补换第五个点刷通过。单屋 0 hub、部分不可达如实记原因。

GPU7 scout 与认证必须使用 `WF/special_scale_holder_v1/locks/gpu_7.lock` 同锁串行：首 3 屋采集后先验一批真实新 hub 的完整族，再按产率决定下一有界波次；旧 69 批待办可在其后交错安排，不能双 producer 共用一张卡。

保留 300/1,000 族检查点和 10,000 族、≥1,000,000 合法动作、尽量覆盖 43 FIT 屋/至少 30 屋的工程目标。当前 11 屋覆盖不足，增加同轨迹命名/语言变体/训练 epoch 不算补齐。当前合格控制类型包含“已完成子目标 revisit 位置对照”，不能夸称已经验证所有无事件无关绕行不变性。

## 7. 资源安排、监控与收口

建议资源分工，**不是跳过执行前身份核验的永久授权**：

| GPU | 本次实测用途 | 下一步建议 |
| --- | --- | --- |
| 0、2 | 真实外部任务 | 不碰，不停止、不重配 |
| 3、4、5 | OpenCode 普通训练 | 当前训练保持；版本修复/分段只由 OpenCode 自己受控安排 |
| 1 | 当时约 12MiB，无模型占用 | 首选独立闭环；单环境起步，模型+renderer 总显存建议 ≤28GiB，启动前核 CUDA/EGL 真实设备映射 |
| 6 | 恢复后的占位 | 精确借还后用于普通数据新波 |
| 7 | 恢复后的占位 | 精确借还后用于 scout→cert 串行 |

GPU 空闲快照会变化。启动前二次核查 UUID、PID/starttime、argv/cwd、必要 env/tmux 与锁，检查计算和图形 context；未知占用就暂不使用该卡，不发信号。所有 own worker 使用有界 supervisor/finally；不用 pkill 或杀 tmux 替代精确清理。新缓存/日志/模型下载只在 LINE，新旧生产数据与 runtime metadata 保持 sibling。

现有监控入口为 `http://127.0.0.1:18766/`，API `/api/status`。远程通过 SSH 本地转发使用，不为了方便直接暴露无认证的 0.0.0.0 服务。OpenCode 后续可在独立监控版本汇总以下只读 JSON；不要将监控和 GPU 控制 API 混成一个无鉴权服务：

- 训练：真实差分吞吐、最近样本窗与固定面板分别画，checkpoint/epoch 游标、数据 snapshot 和累计预算；旧 RESULT 与新活跃进程冲突时明确显示恢复段。
- 闭环：每 checkpoint 的 SR/OSR/SPL、停止失败分类、碰撞和 p50/p95，附 episode 数与未完成量。
- 普通生成：计划/已尝试/终态/强审/最终 merge 各级累计，剩余有效动作缺口与失败原因。
- 特殊生成：预约候选/实际 hub/完整族/强审/训练准入、屋数、物理与 CE owner 去重、每小时合格产量。
- 每个 job 的最后有效产物年龄、阶段、预算剩余、退出原因和恢复状态。进程存在或 GPU 100% 都不能单独显示“正常造数”。

## 8. 推荐的交付顺序

1. **立即**：核 P0 epoch/面板绑定问题；当前训练继续的同时，CPU 完成 bench adapter 与资产/划分清单、普通恢复盘点、特殊准入盘点。
2. **第一批实际结果**：固定 checkpoint 的 10 条闭环接口轨迹；50 条内部 DEV 的配对表；普通完整片的独立最终合并量；特殊可训练动作/族清单。
3. **数据接续**：GPU6 新小片恢复；GPU7 首 3 屋真实 scout 与新 hub 首个强审族；不重跑已完成批。
4. **基础融合**：当前训练安全分段后，普通-only 与 90/10 合法动作短续训配对；完整记忆版先做长历史和泄漏/梯度验收。
5. **科学验证**：固定公开 R2R-CE 验证 pilot→后续全量；记忆动作-only/强 M2/交叉续接匹配试验；再扩自然 RxR 与语言目标任务。

每项交付写明“已完成/失败/待执行”，未测指标为 null。没有闭环结果就不说普通模型已会导航，没有匹配机制收益就不说论文创新已经证明。

## 9. 可直接粘贴给 OpenCode 的执行指令

```text
请完整阅读：
/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/OPENCODE_NAVBENCH_FUSION_20260911_ZH.md

你继续负责当前训练与后续代码执行。先只读复核实时状态，优先检查 epoch 边界空转风险和旧 PANEL_EVAL 模型 hash 不匹配；需要修复时新版本、CPU 测试、checkpoint 边界受控换版，不覆盖冻结源码/输入锁，也不清零累计预算。

下一步三线推进：
1. 独立接 R2R-CE 实际闭环，先固定 checkpoint_000012000 做 10 条 FIT 接口检查，再与 checkpoint_000006000 做固定 50 条 INTERNAL_DEV 配对；资产/官方指标等价性通过后做预先固定的 val_unseen pilot。保留 RGB-only，禁止 oracle 动作/距离自动停止。不是继续只看训练 STOP recall。
2. 将特殊强审核回执转成正式准入的新 snapshot；先比较普通-only 与 90/10 合法动作融合。当前短窗 v3 不具备长期记忆，历史敏感部分另开 execution_memory_v4，并保留同数据同架构动作-only 和强 M2 对照，future query 只进入训练 reader。
3. 跟进普通与特殊造数：EnvDrop 因 shard wall budget 停在 0011，完整片先严格合并，中断片分级恢复，只迁移未尝试项到新小片；batch500…507 已完成不再补挂，四卡旧 69 批待办先全局核重。新 hub 的 43 屋/172 位置尚未实采，优先准备首 3 屋有界 scout→银行→真实族认证与强审，GPU7 同锁串行。

GPU3/4/5 当前训练不被其他新任务打断；0/2 外部任务不碰。GPU1/6/7 只是建议分工，启动前核实时身份、占用、锁和有限预算，精确借还占位。所有新增产物在 LINE 独立 namespace，旧失败和 partial 留存。

先给出你实际接手的 run/目录和任务清单，再交付闭环轨迹/指标、数据准入统计以及新波生产凭据；不要仅重新写一份长方案，不把 CPU 准备或 producer 自报当已完成。
```

本会话未调用 OpenCode 会话发送接口、未在其 tmux 输入指令、未启动新 GPU 工作；用户需将上述指令交给正在使用的 OpenCode。原始研究主线和历史交接不被本文件覆盖重写。
