# 从扩大验证转为实际优化

用户要求扩大收益。本次已实现并启动新训练：只监督部署 FIRST 真正干预的动作位置，完整真实观测历史仍参与递归更新和反向。CURRENT/DELTA 各三个种子、每模型3000步，恢复与保护轨迹、初始化、schedule、类权重、损失系数和架构保持原配置。已有六模型全部开始 GPU 更新，不把旧结果写成新成果。

[代码与方法说明](../../research/continuation_memory_v1/strong_backbone_recovery_v1/execution_adaptation_v1/first_alignment_v1/README_ZH.md)。同目录 CPU_TEST_RESULT.json 是真实 FIT 特征的5项测试，含物理历史梯度与断点一致性。新训练记录在 launch_snapshot/，只是提交时快照，不是训练完成或 SR 提升。

训练完成后流水线自动登记六个新模型，等待当前大模型评测释放资源，再跑固定 unseen80 的原生、旧FIRST与新FIRST共13臂1040次执行。新旧模型在同进程内重新配对。属于已暴露开发比较，不是盲测或完整1839。共同训练修订不单独归为DELTA架构贡献。

独立服务 q35n-strong-first-alignment-20260926-01.service；原监控 http://127.0.0.1:18770/ 新增 /api/first_alignment，当前重点是新训练。代码会自行衔接评测，无需 Codex 在线；旧复验未停止。

EVIDENCE_MANIFEST.json 保存文件SHA。权重、特征缓存、场景和原始图像留在服务器；已保存检查点可由同一 run 加 --resume 恢复。当前收益未知，训练损失或参数更新不能充当导航正向结果。
