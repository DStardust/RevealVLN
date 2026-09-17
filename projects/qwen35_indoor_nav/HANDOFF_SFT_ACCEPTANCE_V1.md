# 交给另一个Codex会话：普通SFT验收

用户已明确分工：你负责SFT验收，主agent负责真实机制族。仅在项目根 `/mnt/data_nas/deeprobotics/daiyang/vla` 内工作，新增内容仅写 `projects/qwen35_indoor_nav/sft_acceptance/v1/`。不修改主agent的STATUS/README、封存reviews/data_pipeline、环境、模型原权重或机制数据节点；结果交主agent合并。当前交接允许有界普通SFT及工程闭环验收，不授权完整论文实验。

## 开始前必读

根/本线AGENTS、MAINLINE_FREEZE_V3.md、03_DATA_CONTRACT.md、04_VALIDATION_PLAN.md、05_ISOLATION_AND_HANDOFF.md；`data_pipeline/ordinary_pilot_v1/REPORT_ZH.md`、SPLIT_FREEZE、TRAINING_INDEX；`reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/REPORT_ZH.md`、probe.py、GRADIENT_PROBE、ENVIRONMENT_ACCEPTANCE及TOKENIZER_ADDITION。
核验这些封存目录SHA256SUMS；不得重跑封存的probe/生成器。所有下面的路径相对本线。

## 已有可直接复用的资产

- `runtime/models/Qwen3.5-2B_15852e8`：固定revision15852e8c16360a2fea060d615a32b45270f8a8fc，官方权重与LFS哈希通过。
- `.envs/q35n_qwen_g2_v1`：torch2.8.0+cu128、transformers5.15.0、PEFT0.18.0；只读复用，不pip install、不改包。
- `.envs/q35n_habitat_v017_g0r`：独立Habitat0.1.7，真实回放通过；同样不改。
- `data_pipeline/ordinary_pilot_v1/TRAINING_INDEX.jsonl`：99不同路线、5房屋、298原指令。先读取索引项的rgb_reference_root再解析policy中的RGB路径，不能统一按根拼接。
- G2只证明两步核心接口。probe中的合成query/BCE及SGD(.01)是诊断夹具，不作为SFT训练目标、优化器或真实机制标签。

## 任务与边界

建立同Qwen+8槽记忆+4动作的普通action-only SFT基线。无续接reader损失、无M2状态辅助、无伪造机制Y、不加载VLFM/ETP导航策略。可以在新目录提取/整理G2已验证的Policy代码，但记录原始源码哈希和所有差异；解决实际发现的训练接口问题要版本化，不能反复换算法找正号。

在看任何训练结果前生成不可变EXPERIMENT_SPEC.json：

1. 按house内数值source episode排序、同物理路线全部原指令同组，每屋前15条不同路线训练（共75），剩余24条作seen-house route-dev。它们不是未见房屋，不能声称泛化；不得重划已冻结FIT_PILOT房屋、读取官方val/test或使用主agent的机制结果筛样本。
2. 冻结seed1109、原动作/相机、两帧窗口、动作历史≤8、K8、rank8 LoRA、视觉冻结、TBPTT与memory reset。建议TBPTT=4、batch1、梯度累积固定、AdamW；学习率/梯度裁剪/调度/token预算必须在运行前写定。不要照抄诊断SGD超参。
3. 总计最多500个optimizer updates（包括任何10路线过拟合smoke），至多一个预注册主配置；不启动超参搜索，不按dev最佳checkpoint筛选报告。若需要工程重启，保持已发生更新/成本/失败记录，不隐瞒重跑。
4. 先通过单batch真实action CE、相邻步记忆梯度、白名单输入、保存/重载logits一致性。基础loss下降仅是学习接口证据。
5. 更新前后在同一冻结24-route dev列表测teacher-forced action CE/accuracy、STOP混淆，不能把动作准确率当导航SR。
6. 若闭环接口可实现，在训练前固定每屋2条route-dev（共10）做before/after配对闭环，最多20个episode，每episode≤512运动动作+STOP。策略不能读pose/navmesh/goal坐标/参考路径/semantic；这些只能在仿真执行器或离线metrics中使用。明确报告停止时目标距离/到达与停止、碰撞、路径一致性、超限/服务失败；未接入并核对官方Habitat-Lab任务指标时，不声称official SR/SPL复现。若闭环接口未通过，仅交离线SFT结果，不伪造收益。
7. 制定训练技术PASS门槛（有限loss、梯度/参数有效、可重载、数据无泄漏）与efficacy指标分开。微型同房屋结果不可记scientific_pass=true、SOTA或创新贡献成立。

## 并行资源约定

主agent机制族使用GPU3及其占位pane `vla_idle_occupancy_20260904:0.0`，你不得操作GPU3/pane0。优先使用空闲卡；如需GPU4/5/6/7，只按用户已有许可释放核实的对应占位，保存PID/完整命令/cwd/pane与恢复方法，在成功/失败/中断后恢复。不得停止真实任务或不明进程。不要照抄GPU3 lease硬编码来控制别的卡。
单卡、GPU≤28GiB、worker RAM≤48GiB、全部GPU阶段累计≤6小时、新输出≤20GiB、0新下载。出现OOM先停并报告；重构checkpoint/TBPTT等工程修复需在新修订登记，不把batch/模型/数据改动伪装成原配置。
所有可写cache、TMPDIR、HF_HOME、日志与checkpoint都在sft_acceptance/v1，原权重/环境只读；不向第三方上传本地数据。

## 回交

REPORT_ZH.md、result.json、冻结协议/划分/来源锁、代码差异、训练曲线、完整step与episode账本、before/after逐项配对、checkpoints与保存重载检查、成本、GPU恢复记录、SHA256SUMS。未执行指标填null，不填预期值。
result分别给data_integrity_pass、training_interface_pass、offline_learning_signal、closed_loop_interface_pass、paired_navigation_change及scientific_pass=false。完成或达到上限后停止，不更新根状态或自行进入下一方法实验。
