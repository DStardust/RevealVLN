# EXPANDED 100,000步长训

本任务由用户要求单独追加。仅种子1209、EXPANDED数据；从scale_003中1200步完整优化器/RNG断点继续到总100000步，新增98800次轻量更新。旧OLD/EXPANDED1200步×3种子对照继续原样运行。

- 冻结Qwen best4k、137020窗口特征缓存、MONOTONIC架构、动作/状态/事件损失、普通动作池及学习率0.001。Qwen基座不训练。
- 调度延长为100000步，前1200条与已执行记录相同；继续按848父族均衡遍历1550变体。普通动作索引沿用原1200步周期。
- 5000、10000…100000：保存完整模型/优化器/RNG断点和可推理head，共20份，全部测试。另滚动保留两个200步恢复断点。
- GPU0训练，GPU1–7评测。使用同用户占位控制器的并发lease，不停止既有真实评测进程。训练与评测不依赖聊天会话，测试分数不反馈调度、学习率或提前停止。
- 每checkpoint256计划条件：主task_A128、task_T128；248条可执行，8条物理数据缺失保留分母。共5120计划槽位，4960条可执行续接。500决策、真实历史回放、主动STOP和无碰撞SEE2检查器保持原样。
- 单种子20次重复评测只作为四个已暴露开发屋上的学习曲线；不宣称新盲测、同预算OLD对照或普通VLN-CE改善。

## 独立运行与恢复

`pipeline.py --config PROTOCOL.json --run-id long_001`，已有目录使用`--resume`。用本目录standalone.py启动同UID/GID systemd服务；完整命令记录在standalone_jobs下。不会重跑已经封存完整的checkpoint条件，故障半条件保留后重跑；正确性失败停止并记录。

监控沿用127.0.0.1:18770，合并原对照与新长训。`runs/long_001/train/PROGRESS.json`为真实优化器进度；`checkpoints/STEP_XXXXXX/COMMIT.json`出现后才准入评测。每checkpoint的`RESULT.json`由真实轨迹与数组重算产生。`LEARNING_CURVE.jsonl`按固定顺序登记全部checkpoint。

所有运行日志、完整断点和场景数组保留本地；GitHub发布代码、协议、摘要、日志包及轻量head，基座、缓存和许可场景不上传。
