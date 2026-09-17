# OpenCode / kimi-k3 执行交接：先训成普通导航基座，再做机制

交接时间：2026-09-10 12:57–12:59 CST。本轮仅只读核实运行状态、新增本文，没有启停训练或采集。本文是最新任务增量，不覆盖冻结规范、输入锁或历史 FAIL。用户要求快速整理后交给 OpenCode 中的 kimi-k3；交接方不再启动新的实验或第二执行者。

## 1. 目标与决策

当前目标是**训练出能正常转弯、停止并实际导航的普通基座，同时把训练端到端吞吐提高至少 10 倍**。不是证明新记忆机制，不以论文创新门槛阻止复用成熟基础训练方法。

采用方向：保留 Qwen 项目路线，优先借鉴成熟 VLN/VLA 的普通监督训练配方；将未验证的自定义记忆机制与普通基座分开。默认准备“Qwen + 因果 RGB/指令/历史 → 四动作”的独立普通基座版本，不直接改旧模型。需要改变结构、监督、采样或初始化时，先写明确版本修订和有限预算；用户已要求普通训练/加速，但这不是对任何新架构、任意下载或无限 GPU 时间的概括授权。显著换成其他模型或增加传感器时再请求用户选择。

复用已有方法应注明来源、许可证和预训练数据；不能把复用部分申报为新贡献。重点排除测试泄漏、额外传感器/真值输入和不公平协议。不要声称“绝无学术风险”，也不要继续为了普通基座反复做泛化 novelty 搜索。

## 2. 路径与最低限度阅读

唯一项目根 `ROOT=/mnt/data_nas/deeprobotics/daiyang/vla`。
本路线 `LINE=ROOT/projects/qwen35_indoor_nav`。
以下路径除 ROOT 外均相对 LINE；不要把这些说明性缩写直接当 shell 环境变量使用。

1. ROOT/AGENTS.md、LINE/AGENTS.md；涉及实现时遵守其中所指规范。
2. `PROJECT_TRANSFER_20260910_ZH.md` 完整阅读，特别第 12/13 节。其快照不是实时状态，本交接是后续用户指令增量。
3. `ORDINARY_TRAINING_CURRENT_20260910_ZH.md`、`MONITOR_AND_RECOVERY_20260910_ZH.md`（里面旧监控版本已被本文 v2 状态替代）。
4. `sft_acceptance/ordinary_speedup_10x_v1/AUTHORIZATION.json`、`BASELINE_EVIDENCE.json`、`PLAN_ZH.md`。最后一份部分“待授权”描述早于 AUTHORIZATION，以授权精确范围为准。
5. 实现前读当前 `ordinary_execution_v6` 的协议、代码锁及其引用；不要全盘扫描数百万数据文件或重跑已闭合准备脚本。

所有新源码/环境/缓存/依赖/输出只写 LINE 下新版本目录。共享资产登记后只读。禁止使用其他项目或旧 `/mnt/daiyang/vla` 链接；禁止修改 ROOT/FROZEN_SPEC、ROOT/NEXT_EXPERIMENT、旧封存和输入锁；保留历史 FAIL。

## 3. 实读运行快照：接手后先再次只读确认

| GPU | 12:57 CST 实读状态 | 后续操作边界 |
|---|---|---|
| 0 | 不在授权范围 | 不操作 |
| 1 | 恢复队列 PID 3746526，当前 batch_500 | 保持；不能重复挂载 |
| 2 | v6 supervisor 3710058 / worker 3710394 正常训练 | 候选验收前不停止；切换需 checkpoint 与可恢复交接 |
| 3 | 原队列已退出，batch_324 已审核，占位恢复 | 可按已批准借卡流程探测，不能直接按显存/PID猜测杀占位 |
| 4 | 原队列已退出，batch_325 已审核，占位恢复 | 同上 |
| 5 | 原队列已退出，batch_330 已审核，占位恢复 | 同上 |
| 6 | 普通 EnvDrop 扩源队列 PID 3584930 | 保持运行 |
| 7 | 特殊采集队列 PID 3593588，当前 batch_327 | 保持运行 |

原始“七条生产队列保持”已被后续用户训练/四卡加速授权局部修订：GPU2 已用于训练；GPU3/4/5 已按授权完成当前批次、清理和强审后停止接续。**不要为了恢复七队列重新启动它们，也不要把有待办的正常 drain 当失败。**

GPU3/4/5 精确证据：

- `data_pipeline/auto_production_v1/special_holder_gpuN_v1/lane_gpu_N/RESULT.json`：error=null，末批状态 `AUDITED_SEE_PER_ITEM_QUALITY`，后续 PENDING。
- `data_pipeline/mechanism_runtime_v1/witness_first_v1/batch_execution_v1/batch_{324,325,330}/run_v1/LEASE_RESULT.json`：execute_returned=true、error=null、holder_restored=true、external_processes_stopped=0。
- `sft_acceptance/ordinary_speedup_10x_v1/DRAIN_BEFORE.json` 及 `DRAIN_REQUEST_GPU{3,4,5}.json` 保留精确身份/信号依据。仅旧 queue parent 收到停止接续信号，没有杀其 worker。
- 本交接核实了上述队列和 lease 收据，未重新逐项裁决全部强审样本。AUDITED 不等于每个候选都通过。

借卡前仍须核对末批 SUPERVISOR_RESULT、RESTORATION、LAUNCH_RESULT 和实际进程。占位恢复后的 PID 会变化，不得用早期 PID 发信号。核实 PID/starttime/cwd/argv/UUID/锁，沿用 finally 恢复流程。精确借卡实现参考 `data_pipeline/mechanism_runtime_v1/witness_first_v1/special_scale_holder_v1/transport.py` 与 `batch_execution_v1/gpu5_transport_v2/transport.py`；不要直接复制历史 GPU7 launcher 中的特定 PID 或 GPU。

## 4. 训练现状及已知问题

运行目录 `sft_acceptance/ordinary_execution_v6`；tmux `q35n_ordinary_train_v6`。
解释器 `LINE/.envs/q35n_qwen_g2_v1/bin/python3`，Torch 2.8.0 / Transformers 5.15.0 / Triton 3.4.0；该环境没有 pip，不要就地安装。

冻结数据 `sft_acceptance/ordinary_baseline_v2/snapshot_v1`：51 个 FIT 屋、5849 物理路线、17548 条官方人工指令、464877 去重路线决策、1394744 指令条件化训练决策/epoch。后者不是 139 万条独立物理轨迹。当前训练不含 EnvDrop、特殊族、DEV/CONFIRM。新增生产数据先审计并做新 snapshot，不向正在训练的锁内追加。

当前实现：通用 Qwen3.5-2B，q_proj/v_proj rank-8 LoRA，冻结基模/视觉；自定义随机初始化 8-slot 记忆、writer 和四动作头。至多两张近期 224 RGB、8 个已执行动作；F/L/R/STOP；无权普通 CE，4 个顺序决策 TBPTT、累积 8 chunks，AdamW lr=1e-4、clip=1。**梯度累积不是实际并行 batch，这不是已验证的公开导航配方。**

12:57:19 实读：206 次更新、6449 个训练决策，吞吐 1.7914 decisions/s，GPU2 使用约 26156/32607 MiB、利用率 24%；最新 checkpoint `ordinary_execution_v6/run_0001/checkpoint_000000200.pt`。只见约 0.46% epoch，尚无 DEV/闭环评估，不能宣布整个 Qwen 路线失败。

但最近 993 决策有明确偏置：accuracy=67.57%，一直前进基线=66.97%，仅高 0.60 个百分点；预测 980 F / 11 L / 2 R / 0 STOP，L recall=4.05%、R=0、STOP=0。不能以总 accuracy 或下降 CE 宣布导航成功。近期窗随路线组成变化，不是固定评测集。

当前单段 wall=86100 秒、总计费决策上限 321000（含 probe）、更新上限 10000；三 epoch 是配置，当前速度根本不能在预算内跑完。重启/换卡不能清零累计资源账。

必须保留的失败结论：

- `action_balance_v1`：温和类别加权 10 路线/200 更新仍 STOP=0，FAIL；不能把“加权就能解决”当事实，也不再拿 10 路线 95% 阻塞正常扩量。
- `efficiency_run_v1`：双卡约 3.51 decisions/s；缓存收益不足 5%，未采用；小集可学性失败。
- `triangular_gpu_v1`：局部约 1.94× 但 logits/梯度数值失败，禁止采用或放宽门槛改 PASS。
- v1–v5 历史启动、优化器恢复或资源失败保留；v6 的真实接口与精确恢复通过不等于导航效果通过。

## 5. 执行顺序：少讨论，按证据推进

### A. 快速落定一个普通基座训练配方

先只检查下列三份官方实现的关键训练代码，不扩展成长期文献调研。产出一页 `RECIPE_DECISION.md`：选谁的哪些部分、动作/传感器差异、初始化、训练目标、batch、LR/调度、数据来源、许可证/commit、评估协议。官方链接是已发现入口，不代表已完成代码/许可证兼容审核。

| 参考 | 可借鉴部分 | 不能直接等同 |
|---|---|---|
| [VLN-CE](https://github.com/jacobkrantz/VLN-CE) | teacher forcing/行为克隆、离线轨迹训练、动作转折点加权、后续 DAgger | 原模型包含 depth 等输入；inflection weighting 不是简单按类别加权 |
| [NaVILA](https://github.com/AnjieCheng/NaVILA) | 预训练 VLA 的导航 SFT、图像历史与动作输出组织、真实批训练 | 模型/动作控制接口不同，不能把其权重直接加载 Qwen 并声称复现 |
| [ETP-R1](https://github.com/Cepillar/ETP-R1) | 导航预训练→在线 SFT→RFT 的阶段划分 | 全景/depth/拓扑航点接口不同，不适合作为当前四动作无缝替换 |

VLN-CE 重点读 `vlnce_baselines/config/default.py`、实际 DAgger/数据 loader；NaVILA 重点读 `scripts/train/sft_8frames.sh` 及数据预处理；ETP-R1 本地登记只读副本在 ROOT/third_party/ETP-R1。不要运行其未经迁移审计的旧环境。

推荐首个实现只做普通监督导航：保留合法因果输入和 F/L/R/STOP，明确图像-动作时间对齐、STOP 终点监督、token mask、路线边界重置；真实 batch 来自多个独立轨迹窗口，绝不能混用别的路线记忆。新普通基座可以不带研究性记忆模块，但这是结构修订，要单列版本，不是对 v6 的同构续训；不能伪称沿用其优化器状态。先不加 GRPO、不开展无限 DAgger、不混入特殊机制族。

### B. 加速与正确性：目标明确，不承诺未测的 10×

登记基线 1.8224896976 decisions/s，10× 目标 **≥18.224896976 decisions/s**；见 `BASELINE_EVIDENCE.json`。计完整数据读取、预处理、forward、backward、optimizer 和正常日志，报告 warmup/编译时间及稳定段；不要用局部 kernel speedup、减少样本/图像或增加梯度累积冒充十倍。

1. 先短 profile 定位。现有 Transformers Qwen3.5 Gated DeltaNet 确实走纯 Torch fallback，其中 chunk 内 Python 行循环是候选瓶颈；尚未测其端到端耗时比例。
2. 官方 FLA 优先；新目录依赖已下载解包，未完成 import/GPU 数值验收，不能宣称可用：
   - `ordinary_speedup_10x_v1/official_fla_0_5_2/deps`，wheel 819225 bytes，SHA256 `5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761`。
   - `ordinary_speedup_10x_v1/official_einops_0_8_1/deps`，wheel 64359 bytes。
   - 两目录各有 PROVENANCE；不重复下载、不向原环境 pip install。所有临时/cache目录在 LINE 内。
3. 先 CUDA_VISIBLE_DEVICES 为空做隔离 import；正式探测前冻结源码/输入/数值阈值。比较 fallback 与官方核的输出、梯度、优化器更新和恢复。注意原始 q/k 在转 FP32 前已做 l2norm，不能忽略 BF16 舍入差异；不要偷偷因提前导入 fla 把 reference 也换成候选。
4. 首个 GPU 核探测已授权：一张完成 drain 的卡、≤1200 秒、自己的显存 ≤28 GiB、≤512 forward decisions、≤8 个丢弃的诊断 optimizer updates；0 正式训练、0 导航 episode。原占位必须精确借还，持有对应 GPU 锁，失败清理只限本次子进程。
5. 普通基座实现真实多样本 batch / 长度分桶 / 有限预取，再考虑 GPU2/3/4/5 四卡 DDP。不能把不同路线沿时间维拼起来共享状态。有效 batch、loss reduction、更新频率改变需登记；不同结构的提速另报，不混成同构优化。
6. 四卡正式训练前冻结自己的有限协议，做匹配数值/吞吐/断点恢复验收。原 kernel probe 的 8 更新许可不能扩写成普通基座训练预算。预算与已有准入不一致时一次性请求所需变更，不无限追加小实验。

若官方核在既定预算内不兼容，保留 FAIL 后评估真实 batching；不要再投入长时间手写未经验证的 GDN 数学重写。10× 未达到就报实测和剩余瓶颈，不伪报完成。

### C. 效果验收、正式切换及交付

冻结一个覆盖 F/L/R/STOP、多屋多路线的固定 FIT 诊断面板和独立 DEV 评测方案；每个 checkpoint 用同一面板观察学习趋势，保持 DEV/CONFIRM 不参与训练。小面板检查接口/学习趋势，不取代全量数据，也不是任意苛刻的小集门槛。

至少记录：CE、每类 recall、macro recall、目标/预测分布、一直前进基线差值、端到端 decisions/s、样本覆盖/epoch、累计预算。实际闭环报告 SR/SPL/终点距离/停止行为；不要把 teacher-forcing accuracy 当 SR。闭环执行须有明确有限 episode/资源协议，不能从 kernel probe 自动推定已授权。

候选尚未通过时 v6 保持原预算运行。候选可接管时：先确保旧训练最新 checkpoint 完整、记录恢复路径/累计资源，核实精确进程身份后正常关闭自有旧训练并清理，再单次启动新版。只兼容时续载 optimizer；新结构明确作为新 run，保留旧 checkpoint。启动后验证心跳、实际更新、checkpoint 恢复、监控指向，并写 START_RESULT，防止 tmux 已存在但 worker 未启动。

最终交付：选定配方及代码版本、冻结数据来源、实测加速表、训练/评估结果、可恢复 checkpoint、远程监控入口、生产队列状态。未完成项和失败明确列出，不只写“已准备”。

## 6. 监控已上线，不需重建

当前 `sft_acceptance/monitor_charts_v2/server.py`，PID 3758671，tmux `q35n_monitor_charts_v2`；监听仅 `127.0.0.1:18766`。GET `/`、`/api/status`、`/healthz`，无 HTTP 控制接口。

```bash
ssh -N -L 18766:127.0.0.1:18766 <用户名>@hpda-jushendaohang-005
```

浏览器打开 `http://127.0.0.1:18766/`。已有 CE/吞吐/GPU/预算/队列图，v2 增加最近约 1000 决策的基线差值、四类召回和预测偏置。v2 部署凭据 `DEPLOYMENT_RESULT.json`；质量统计 5 项 CPU 测试及 Node DOM 绘图检查通过，不声称浏览器布局已验收。

`ordinary_speedup_10x_v1/quality_diagnostics.py` 已被 monitor v2 CODE_SEAL 绑定，不可再就地修改。若新版训练字段变化，做 monitor v3 并测试，不覆盖封存 v1/v2。禁止开放无认证公网写接口。

## 7. 失败任务已挂好，不要再挂一次

`data_pipeline/auto_production_v1/manual_failure_recovery_gpu1_20260910_v1/PLAN.json` 已运行，GPU1 单队列串行：225→500、244→501、226→502、245→503、227→504、246→505、228→506、247→507。前两项失败新尝试，其余是未执行预约迁移；旧失败不覆盖。

准备证据 `data_pipeline/mechanism_runtime_v1/witness_first_v1/manual_recovery_20260910_v2`。总窗口至约次日 00:31:48，最多 8 批，尾批可能预算不足不启动；仍需原强审后接续，非无限自动失败重试。GPU6 普通扩源与 GPU7 跨屋特殊采集继续；不要另起生产者写同一批次。

## 8. 给执行者的工作方式

先用只读状态核对本文，不照抄旧 PID 发信号。第一轮直接交付配方选择和新版本最小实现/测试进度，随后在已授权边界内执行；不要反复问是否继续，不继续扩大文献调研，也不要把所有历史失败从头重跑。遇到真正的权限/预算/架构选择缺口，集中一次说明；不能拿本文作为超出用户授权的凭据。

交接时尚未完成：FLA runtime/import/GPU验收、普通配方选择与新实现、四卡启动、10×吞吐、DEV/闭环导航收益。**没有从本会话启动 OpenCode，也没有证据表明 kimi-k3 已接收或运行；用户需在自己的 OpenCode 会话中发送交接指令。**
