# 普通 action-only SFT 有界验收

结论：`COMPLETED_BOUNDED_ACCEPTANCE`。科学结论固定为 `scientific_pass=false`。

本次只使用 Qwen3.5-2B 固定 revision、8槽记忆、rank8 LoRA 与4动作 CE，无续接 reader、机制Y或状态辅助。75条训练路线/226条原指令，24条 seen-house route-dev/72条原指令；5个房屋都属于已暴露 FIT_PILOT。不可解释为未见房屋泛化。

## 实测结论

- 数据完整性：True；训练技术接口：True。
- 离线 CE 下降信号：True；动作准确率不是导航 SR。
- 闭环接口：True。官方 Habitat-Lab SR/SPL 未接入，均为 null。
- 保存重载检查：True，详见 RELOAD_CHECKS.jsonl；仅保存可训练参数及终点优化器状态，原模型保持只读。
- GPU4占位恢复：True；停止真实任务数0，未操作GPU3及其pane。

## 协议与结果

EXPERIMENT_SPEC.json 在任何新模型结果之前冻结，SHA256 `87a6840a62dab085478d18a0088ed1e200a59ae19139b53da0203321c9968a6a`。AdamW lr=1e-4、TBPTT=4、batch1、每8片段累积、恒定学习率、梯度裁剪1.0，固定400更新上限。不存在dev最佳checkpoint筛选或超参搜索。

工程修订见 recovery_r1/AMENDMENT.json：原运行更新1已发生，但随后的日志函数重复unix参数而退出，未保存该更新参数。因此从初始checkpoint重启399更新（初始logits再次核验），累计计算400更新、最终checkpoint有效399更新，不能称为原计划无中断的400步模型。没有改变算法或数据；原before离线与10个episode直接复用，不消耗额外闭环名额。原失败、账本、GPU恢复记录均保留。

更新前离线结果：`{"decisions": 3864, "decision_micro_CE": 3.3743679157318933, "decision_micro_accuracy": 0.11283643892339544, "route_macro_CE": 3.4063290898369813, "route_macro_accuracy": 0.11440879620506227, "confusion_target_rows_prediction_columns": [[78, 101, 1135, 1074], [32, 31, 291, 378], [27, 24, 300, 321], [3, 4, 38, 27]], "STOP_TP": 27, "STOP_FP": 1773, "STOP_FN": 45, "STOP_TN": 2019}`

更新后离线结果：`{"decisions": 3864, "decision_micro_CE": 1.0650498660794203, "decision_micro_accuracy": 0.6229296066252588, "route_macro_CE": 1.0690608956444765, "route_macro_accuracy": 0.6241617964658229, "confusion_target_rows_prediction_columns": [[2382, 0, 6, 0], [704, 0, 28, 0], [647, 0, 25, 0], [72, 0, 0, 0]], "STOP_TP": 0, "STOP_FP": 0, "STOP_FN": 72, "STOP_TN": 3792}`

10条固定闭环路线的配对平均变化（after-before）：`{"stopped_within_3m": 0.0, "ever_within_3m": 0.1, "collisions": 504.1, "motion_actions": 510.9, "geodesic_ndtw": -0.30753963077055196, "terminal_goal_distance_geodesic_m": -0.487241268157959}`。停止半径采用3m，分别记录曾到达与到达后停止；nDTW采用仿真geodesic距离及原参考路点，未声称官方指标复现。每episode最多512运动动作，STOP单独记账，服务错误与超限保留。

## 成本、失败与限制

实际优化器更新 400 次；GPU阶段 2.3626 小时；峰值全卡显存 14317.0 MiB；worker树峰值RSS 4140535808 字节；闭环启动 20 个、完成 20 个；新增下载0。

原执行失败：`{"error": "RuntimeError('Worker exited 1; no automatic retry')", "traceback": "Traceback (most recent call last):\n  File \"/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/sft_acceptance/v1/launch.py\", line 99, in main\n    if proc.returncode:raise RuntimeError(f'Worker exited {proc.returncode}; no automatic retry')\nRuntimeError: Worker exited 1; no automatic retry\n"}`。两次执行的完整退出与成本记录：`[{"returncode": 1, "error": "RuntimeError('Worker exited 1; no automatic retry')", "gpu_stage_wall_seconds": 638.7873804569244, "real_tasks_stopped": 0}, {"returncode": 0, "error": null, "gpu_stage_wall_seconds": 7866.414057016373, "real_tasks_stopped": 0}]`。CPU准备首次误用了系统Python3.6，因标准库参数不兼容退出，未写数据、未使用GPU；随后使用已授权项目Python3.10。没有隐藏旧1次更新成本。原进程未保存完整forward-token总计，精确累计值保持null；恢复过程总计与原阶段保守上界分开报告。

这是微型同房屋工程验收，有限loss下降只能支持学习接口；导航变化是固定小样本描述，不确立机制收益、创新性、SOTA或完整论文贡献。未新增未见房屋测试，不读取官方val/test，不使用主agent机制结果筛样本。

## 证据入口

冻结协议与来源：EXPERIMENT_SPEC.json、PREREGISTRATION_LOCK.json、SPLIT.json、SOURCE_LOCK.json、SEALED_SOURCE_CHECKS.json、FINAL_SOURCE_CHECKS.json。

代码来源和全部差异：CODE_PROVENANCE.json、CODE_DIFF.patch、CODE_LOCK.json。训练接口：TRAINING_PREFLIGHT.json、TRAINABLE_PARAMETERS.json、PARAMETER_CHANGES.json、RELOAD_CHECKS.jsonl、checkpoints/。

完整账本：根与recovery_r1各自的MODEL_STEP_LEDGER.jsonl、UPDATE_LEDGER.jsonl、TRAIN_ORDER.jsonl、TRAIN_EPISODE_LEDGER.jsonl、EPISODE_LEDGER.jsonl、SIM_STEP_LEDGER.jsonl；汇总TRAIN_EPISODE_COVERAGE.json、STEP_ACCOUNTING.json。配对结果：OFFLINE_PAIRED.json、NAVIGATION_PAIRED.json；曲线：TRAINING_CURVE.csv（有更新时另有SVG）。资源与恢复：COST.json、两个阶段的RESOURCE_SAMPLES.jsonl、EXECUTION_RESULT.json、LEASE_BEFORE.json、LEASE_ACTIVE.json、LEASE_RESTORED.json。终点checkpoint及更新后评估在recovery_r1中。

结果只写 sft_acceptance/v1，交主agent审核合并；未修改根或本线STATUS/README，不自行进入下一方法实验。未生成的文件或指标以实际目录及result.json为准，不将本证据清单当作已执行证明。
