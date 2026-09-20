# V16 rqfALeAoiTq 修复运行

- 来源旧 run：`/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16/runs/v16_formal_001`（只读）
- 修复版本：`v16_rqf_full_neutral_yaw_search_v1`
- 已复用 24 个已认证族；只重新搜索缺失 TEST 屋的 2 个族。
- 不读取模型分数，不改变 SEE2、碰撞、主动 STOP、500 决策或信息隔离门槛。
- 修复方式：穷举冻结中性朝向；仍不足时按确定性 pathfinder 流追加 hub。
