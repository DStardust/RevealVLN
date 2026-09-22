# 冻结动作路径的事件修复 V1

本轮状态以 runs/preserve_001/STATUS.json 和 RESULT.json 为准。当前代码/CPU检查不代表导航收益。

此前完整 EVENT 修复比 ORIGINAL 少10个主任务成功。首分歧定位显示22/23个退化配对的主要不利动作margin变化来自动作路径；这只提供修复方向，不保证本轮成功。

本轮从三个种子已训练的 ORIGINAL checkpoint 分别出发，冻结全部 core（递归记忆、动作读出、归一化）、initial_belief、state_action 和 revision，只训练 events MLP 的4个张量，共262530参数。底模和推理架构不变。固定每个种子200步、3个模型共600步；ORIGINAL本轮0更新，原权重重新跑完整对照。

保留完整原动作CE、cutoff CE、精确状态BCE、preservation KL及ordinary CE，沿用各seed原schedule前200项和相同普通轨迹。事件BCE沿用上一版本已登记的输入去重、屋/角色等权、单元内实际正负比例；每步256标签。本轮不重复LOHO，不按DEV挑步数或参数、不训练失败后扩预算。AdamW lr0.001、weight_decay0.01，优化器只持有event参数，每100步保存模型/优化器/RNG。

冻结权重只保证在相同因果输入下原记忆/动作路径的函数不变；事件状态和最终动作仍可能变化，实际分叉后输入也会变化。验收必须重新运行同一暴露DEV的128条件×6模型=768次续接，主任务每臂192、task_T每臂192。每条件六模型成组封存，原始/processed/原生动作前缀按原协议检查。明确报告原有成功保留、丢失与新增，不能只看平均事件校准。不是用旧66/192替代本轮控制。

这是“有界事件修复 vs 不修复”的工程比较，不是同更新预算的研究监督对照；不声称单独估计冻结、初始化和训练步数各自的因果效果。相对旧EVENT的分数也不能单独作为冻结策略的独立因果效应。没有新场景/新数据/普通VLN-CE结果，也不自动采用。

资源：训练3张GPU并行，评测8张GPU分配完整条件组；只借核实的用户占位、结束恢复。累计GPU会话上限8小时，单worker3小时，产物40GiB；无按分数重试。正确性/资源错误保存已封存证据后停止。5项CPU测试含真实更新、冻结路径、RNG恢复和动作选择契约。

独立启动（首次不加--resume；已准备或故障恢复需--resume）：

```bash
STD=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
export V16_STANDALONE_PYTHON="$STD"
"$STD" -I -S -B standalone.py start preserve-event-20260922-01 -- "$STD" -I -B "$PWD/pipeline.py" --config "$PWD/PROTOCOL.json" --run-id preserve_001 --resume
```

原网站端口18770显示三模型事件训练步数、冻结审查与完整配对计数。训练/评测/复核/上传由独立systemd服务推进，不依赖Codex会话。原始场景/数组与特征缓存留本地，代码、协议、轻量权重与可重算文本日志自动上传GitHub当前分支。旧结果和他人修改只读保留。
