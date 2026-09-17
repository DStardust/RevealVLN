# 当前全历史审核与初步门槛 V2

batch02r2 的两个实际族通过原版常规审核：

- [5LpN3gDmAk7](current_all_batches_v1/batch_5/WF_WIND_d42a6f952947c0851cf870ac_B02_LR1/FAMILY_REPORT.json)
- [29hnd4uzFmX](current_all_batches_v1/batch_5/WF_WIND_756e47e60519cae5e7c93f31_B02_LR1/FAMILY_REPORT.json)

均为 `QUALITY_VERIFIED_CANDIDATE_FIT_NOT_MODEL_GAIN`，各核验 18 格、27 真实认证重放、54 评估。首个 2n8k 候选原中性旋转条件拒绝，记录未删除。整批 92 traces、22658 确认动作、0 碰撞，supervisor rc0/cleanup=true。

[GPU5 借卡恢复审计](current_all_batches_v1/GPU5_LEASE_RESTORE_AUDIT.json) 9/9 通过：renderer 已退出，占位进程按原命令/cwd/UID 恢复为新 PID3242643，tmux 原 off 状态恢复。原借卡没有 pidfd 的双重身份检查残余竞态如实保留；本审核未操作 GPU 或进程。

[当前全历史报告](current_all_batches_v1/REPORT.json) 曝光 6 个已声明批目录，含资源中断和未启动失败。batch02r2 满足单批 3 个独立 budget-start 尝试 hub、2 个常规合格 hub；其余批不满足完整运行资格。这里尝试沿用旧 gate 的预算 bundle 已启动定义，不要求已经确认运动动作；例如 batch01r1 第三候选启动了预算阶段但没有确认动作，且没有终态结果，仍不能通过。

现有这些批的数据库存为 2 个常规完整运行族 + 主 agent 接纳的 3 个恢复族，共 5 个不同 hub、4 个 FIT 屋。未纳入旧 V3；它和 batch00 首族同 hub，不能增加独立空间支持。恢复族不补常规批次门槛。

总体初步稳定门槛 **未通过**：只有一个合格完整批，且没有合法事前冻结的多批评价 cohort，常规独立 hub 数也不足 6。不追认当前已知结果组成 cohort。

新 gate 允许至少 2 批，不再限制恰好 2 批。未来可按 [SPEC_ZH.md](SPEC_ZH.md) 事前冻结新机制/运输版本和完整批 ID 的 cohort；全历史失败继续曝光，但不永久否决这样的新 cohort。同 cohort 失败不得事后删除，恢复数据和常规完整批仍分开。

8 项 CPU 测试通过；旧 validator、旧 SPEC、旧运行输入及冻结结果未改。统计稳定、训练授权、模型泛化及科研收益均不由本工程裁决成立。
