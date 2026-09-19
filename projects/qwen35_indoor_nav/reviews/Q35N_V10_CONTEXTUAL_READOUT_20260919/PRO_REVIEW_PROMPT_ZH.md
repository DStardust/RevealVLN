# Q35N 实测回交：裁决下一项研究，不继续追逐同一 CHECK

请访问 DStardust/RevealVLN 的 `codex/q35n-research-evidence-20260919` 分支，先固定实际读取的最新提交 SHA。项目根为 `projects/qwen35_indoor_nav/`。这是已有代码和实测结果的方向审查，不是请求包装正向论文。GitHub包含代码、协议、逐步日志和报告；权重、图像及缓存仍在服务器，不能宣称你已加载模型或重放仿真。

请先读下列真实文件，并区分读到的证据与无法访问项：

1. `CURRENT_STATUS.json` 和 `reviews/Q35N_V10_CONTEXTUAL_READOUT_20260919/REPORT_ZH.md`。
2. `closed_loop_bench/ordinary_memory_transfer_v10/{PROTOCOL.json,RESULT.json,DIAGNOSIS.json,CROSS_SESSION_NATIVE_AUDIT.json,evaluate.py,review.py,finalize.py}`；选择报告引用的逐组 `GROUP.json`、`POLICY_STEPS.jsonl` 和运行身份核查。
3. `research/continuation_memory_v1/contextual_readout_v10/{model.py,losses.py,REVIEW.json,MEMORY_DIAGNOSIS.json}`。
4. `research/continuation_memory_v1/cost_teacher_v13/{teacher.py,losses.py,REVIEW.json,REPORT_ZH.md}`。
5. `research/continuation_memory_v1/fork_balanced_v14/{PROTOCOL.json,losses.py,REVIEW.json,STRONG_STATE_PROTOCOL.json,STRONG_STATE_REVIEW.json,MEMORY_SWAP_TRANSITIONS.json,REPORT_ZH.md}`。
6. `research/continuation_memory_v1/semantic_transfer_v12/{REPORT_ZH.md,ACTION_TARGET_AUDIT.json,SCREEN_REVIEW.json}`，及 `query_semantics_v11/PAPER_PLAN_ZH.md` 中的近邻方法和原始数据来源。

不可丢掉的结论：

- V10三个固定种子×100组×五臂，1500次episode、270475次决策全部完成并独立重算。原生SR均为21%；Ours为10/11/12%，B2为14/12/17%，B1为18/12/14%，N0为10/14/11%。全部同组干预前输入/动作/logits一致，参数前后不变。本版没有方法收益，不采用。
- 跨模型进程的原生数值仍有差异：相对1209，1210和1211分别32、24条相同输入发生原生动作翻转。不能冒称所有运行逐位一致，也不能用跨进程差异抹掉同组有效负结果。
- V14修复了稀少关键分叉动作被普通动作损失稀释的问题。Ours FIT成对正确数从5/78升至30/78，错误记忆交换后降至1，同状态sham为27，这是实际记忆使用的调试正向证据。但B1为36/78；CHECK上Ours与B1均为5/72。近似容量匹配的精确状态MLP对照也已完成，没有稳定的交叉监督增量。72是24对×3种子的重复测量，非72个独立样本。
- V14已有真实前向、跨步梯度、9000次CPU更新和15份权重读回；没有V14闭环。V10与V14是不同固定实现，不能混用结果。
- 数据仍是SEE2，不是房间到访；原始`training_admission=false`、物理状态规范化及负控制缺项仍在。V12的209族/11屋仅审计和有限视觉证据筛查，不能冒称全部正式训练或独立泛化数据。
- V5循环规则完整100对SR20%对20%；旧41800完整1839 SR22.02%不是best4k结果；SR40和真机部署均未完成。已暴露数据不能改名盲测。

请给出具体裁决：当前是否还有值得保留的“交叉续接优于精确状态监督”的模型主张，还是应撤回该主张、仅保留记忆可学习性与数据/运行资产。精确检查器状态已足以决定续接结果，不能宣称Y信息更多；不能把有限记忆、辅助预测或移除训练读出器本身当新颖性。

如果建议继续，只给一个最小、可证伪的下一项实验：指出真实文件/函数的唯一修改、使用或补齐哪项真实资产、独立评测单元、必须匹配的B1/B2、主要终点、预算依据以及正/负结果分别导致什么决策。优先解决当前证据指出的缺口，不再在同一已暴露CHECK上轮换损失权重、挑种子、弱化B2或换名字。不要将首种子优势或FIT提升写成论文成立。

请明确区分三个可能问题：视觉事件证据是否可迁移、监督是否真的训练动作所用记忆、该记忆是否改善真实闭环。某一层通过不代替其他层。若证据不足以选择一个有希望的下一实验，直接说不足，并指出最小缺失事实；不要以宏大新方向或新的准入文档替代判断。

相关工作只在需要裁定具体同构先例时检索一手来源，分别记录论文和可运行代码；重点比较已有未来查询/进度状态/预测性状态方法，不能假设目前命名有新颖性。本轮审查不授权新训练、全1839、方向B或部署。
