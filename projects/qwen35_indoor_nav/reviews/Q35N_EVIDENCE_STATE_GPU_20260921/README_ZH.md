# Q35N：证据状态策略 GPU 开发实验

最新快照：[20260921T100431Z](snapshots/20260921T100431Z/INDEX.json)。状态：**RUNNING**。尚无完整结果时，方法收益为 UNKNOWN。

入口：[协议](../../research/continuation_memory_v1/evidence_state_policy_v1/gpu_runtime_r1/PROTOCOL.json)、[运行说明](../../research/continuation_memory_v1/evidence_state_policy_v1/gpu_runtime_r1/README.md)、[模型](../../research/continuation_memory_v1/evidence_state_policy_v1/model.py)、[损失](../../research/continuation_memory_v1/evidence_state_policy_v1/objective.py)、[Jev 只读调研](../../research/continuation_memory_v1/evidence_state_policy_v1/JEV_RESEARCH_NOTE_ZH.md)。

旧 B2Fix 实验按用户要求停止：1143/1152 续接、127/128 完整组。历史任务 B2Fix 40/192、B2 61/192，两臂各3条未完成；不重写为完整或有效修复。见[停止证据](snapshots/20260921T100431Z/repair_001/USER_STOP_REQUEST.json)和[部分结果](snapshots/20260921T100431Z/repair_001/REPORT_ZH.md)。

新实验共享现有 CPU 验收数据、初始化与监督，对照直接预测状态 DIRECT、单调累积 MONOTONIC、可修订历史 REVISE。每种三种子，共九个全新轻量模型、各1200次更新；冻结 Qwen best4k，原始全量保持 KL 不变。计划128条件×九模型=1152次真实续接。主任务和历史无关控制每臂各192，使用七卡1–7并行完整配对组。

当前数据为已暴露的 FIT四屋/32父族、DEV一屋/8父族，不是独立TEST。新模块的贡献是假设，不宣称普通VLN收益或论文新颖性已经通过。现有B2不是本轮原样匹配臂，不把新增事件监督、动作状态输入的共同变化归于REVISE。

本次运行器修复CONTENT_PATH错误，复用gpu_v1已训练完成的九份权重（新增训练更新0）；模型、损失、动作和评测均未变，旧失败日志保留。

Jev暂按TypeSafe于2026-09-15发布的项目理解。仅增加真实自主输出的只读Brier/ECE统计；不调用其API、不改变动作、不选择阈值，不把概率输出本身作为新发明。

独立systemd流水线完成后复核、恢复GPU2–7占位、发布最终权重及日志，不需要Codex在线。监控为服务器127.0.0.1:18770。当前快照不代替最终封存报告。
