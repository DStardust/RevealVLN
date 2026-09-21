# Q35N：B2 保持损失的可控修复与交接

最新证据快照：[20260921T083259Z](snapshots/20260921T083259Z/INDEX.json)。当前实验状态：**RUNNING**。没有完成结果时，修复收益为 UNKNOWN。

审阅先读：

1. [本轮修复协议](../../research/continuation_memory_v1/grounded_state_transfer_v16/b2_kl_repair_v1/PROTOCOL.json)
2. [实现和运行命令](../../research/continuation_memory_v1/grounded_state_transfer_v16/b2_kl_repair_v1/README.md)
3. [上一轮完整1152次结果](snapshots/20260921T083259Z/runs/matched_001/REPORT_ZH.md)
4. [冻结特征定位结果](snapshots/20260921T083259Z/b2_localization_v1/run_002/REPORT_ZH.md)
5. [梯度幅度补充](snapshots/20260921T083259Z/b2_localization_v1/run_002/GRADIENT_SCALE_ADDENDUM.json)

上一轮主任务：B1 59/192，B2 61/192，Terminal 41/192；控制任务：114/192、74/192、80/192。没有采用B2。当前事件特征探针FIT约97–100%，DEV约62–71%；B2记忆的历史状态线性探针FIT约89–90%，DEV约56%。这不支持单纯扩大记忆容量。

本轮唯一训练变化：B2Fix 在有合法教师监督、原生argmax与该条教师动作不同的位置取消KL分子项，原分母不变。不是删除全部KL，不改变数据、动作标签、普通动作CE、底模、8×64架构或STOP定义。用相同初始化、schedule训练三个B2Fix final1200；复用原B1/B2六个封存权重，不复用旧导航轨迹。

三臂×三种子，同条件九模型在一个底模进程比较；128条件、1152次新续接，每臂主任务192、控制任务192，八卡分组。主要差值B2Fix−B2，B1为参考。仍是单个已暴露DEV房屋的受控任务，不是普通VLN、盲测、论文创新或部署结论。24/24局部梯度反向不证明KL为根因；其幅度中位数约为动作梯度的24%。

代码、报告、逐步日志包、最终轻量权重及SHA可在此分支复核。原始场景、底模、特征缓存和优化器二进制未随GitHub发布，见INDEX边界；不要声称网页审阅等于模型复现。

最终结果和最新快照由独立运行器完成后自动提交推送，无需Codex保持在线。资源/正确性失败会保留所有attempt；不会按分数重试。
