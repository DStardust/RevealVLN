# 执行 Codex：只修正 P2 的七项验收问题

你继续担任执行 Codex，原会话的主 agent 保持论文主线和验收职责。

## 目标与范围

执行 `Q35N_P2R1_SPEC_CORRECTIONS_V1`。不是重做创新审查，不更换主线，不启动 G1。
先完整读取根/本线 AGENTS、README/STATUS、MAINLINE_FREEZE_V3、06_END_TO_END_METHOD_AND_VALIDATION_V1，原 P2 全部规格，以及：

- [主 agent 审核报告](reviews/Q35N_P2_MAIN_AGENT_REVIEW_V1/REPORT_ZH.md)
- [主 agent 机器裁决](reviews/Q35N_P2_MAIN_AGENT_REVIEW_V1/result.json)

项目根仅 `/mnt/data_nas/deeprobotics/daiyang/vla`；本轮唯一写目录为其内 `projects/qwen35_indoor_nav/reviews/Q35N_P2R1_SPEC_CORRECTIONS_V1/`。
原 P2、主 agent 审核、主线、STATUS 和其他项目文件只读。不覆盖旧交付，不改执行授权。
继承原 P2 交接的只读源码/网页和文档权限；禁止安装、下载软件/模型/数据、仿真、模型 import、训练、GPU/tmux 操作及算法实现。schema 和声明式草案可写，允许静态 JSON/路径/哈希检查。

## 逐项关闭 R1–R7

1. R1：分开独立 runtime 准备与单族 replay 草案，解决预算、下载和渲染 GPU 矛盾。提出必要资源，不实际使用；不能假定 headless 等于 CPU 渲染。
2. R2：让可观测语言、房间/目标实例定义、到达与停止谓词一致。静态改动须版本化、事前说明，不以后验成功改任务。
3. R3：明确原子事件 T/F/U、有序程序求值与完整证据要求；补 pass/fail/unknown 示例与 mask/时间/矩阵等跨字段校验契约。
4. R4：query 改成可解释的真实动作/事件续接内容，不是 C_D 代号。提供独立编码接口和 ID 不变性、同查询异标签示例；禁止未来进入策略。
5. R5：写实 M2 任务状态监督，而非物理状态匹配；明确其张量、时间、mask、loss。动作去重保留 task/history/time 条件，只去交叉复制。
6. R6：补动作输入及完整 token/位置/类型/attention 张量流；精确到官方接口对象。冻结视觉特征的梯度 probe 单独定义，真实 backward 留给 G2。
7. R7：补有界候选搜索、物理配置与失败分层，说明逆路线成本和重复 trace 计数。分开本地使用与公开发布许可，记录可查依据、缺失证明及需要用户确认的最小事项。

允许保留“待运行验证”，不能把未重放当阻碍静态修订的理由；也不能将纸面修正写成已运行通过。
本轮不要求选择确认集、做效能实验、完成现代强模型复现或重新检索论文。原方案未测数值全部保持 null/UNVERIFIED。
若某个关闭项确需改变核心算法或新权限，记录精确问题与最便宜的后续动作回交，不自行扩展。

## 交付

在本轮唯一目录输出：

- REPORT_ZH.md 与 CORRECTION_MATRIX.json：每项 R1–R7 的旧问题、修订位置、证据、STATICALLY_RESOLVED/UNRESOLVED/DEFERRED_RUNTIME；后者不得掩盖静态缺项。
- 自洽的 MINIMAL_FAMILY_SPEC_V2.md、DATA_SCHEMA_V2.json、EXAMPLE_RECORDS_V2.json、IMPLEMENTATION_SPEC_V2.md、RESOURCE_PLAN_V2.json。
- RUNTIME_SETUP_GATE_DRAFT.json 与 FAMILY_REPLAY_GATE_DRAFT.json，两者 executable=false，明确依赖和权限。未授权未来权限必须标 proposed 而非 allowed。
- SOURCE_EVIDENCE.md：直接源码/条款依据及访问失败范围。法律许可不由模型凭经验签发；不得接受条款或对外发布。
- result.json、SHA256SUMS；记录原 P2 SHA256SUMS 复核，原产物与受保护文件前后变化。

result 使用 READY_FOR_MAIN_AGENT_REVIEW 或 BLOCKED_WITH_SPECIFIC_GAP；scientific_pass=false、implementation_allowed=false、training_allowed=false、runtime_verified=false、new_training_runs=0、new_navigation_episodes=0、measured_navigation_gain=null。

完成后回交七项关闭矩阵与剩余前置条件，停止等待主 agent。不要自动执行任何新草案。
