# 冻结 MONOTONIC 的新屋验证

原开发屋 MONOTONIC 66/192、DIRECT 55/192，差值 +5.73pp，三种子方向一致。这是选择本轮候选的开发证据，原实验主比较 REVISE 的弱结果仍保留。

本轮只复用六份 final1200 权重，不训练。四个新记忆评测屋，各四父族、每族终点可见/不可见两变体，计划256条件×六模型=1536续接。每臂主任务384、task_T控制384，分别报告。仍是原地 SEE2 停止与转向任务，不是普通VLN-CE。

先执行冻结的物理采样规则，保留全部尝试，再审核原始RGB/语义数组并冻结完整槽位清单。采集缺项保留为 NOT_COLLECTED；不因模型分数换样本。同条件六模型在一个底模进程中完成、核对输入与原生动作前缀后整组封存。中断恢复不拼半组，不覆盖已完成组。

`pipeline.py` 自动完成 prepare → collect → audit_data → evaluate_continuations → review → publish。共享GPU2–7通过已有占位租借接口释放，并在结束/失败时恢复；没有全机独占声明。0–7卡在资源检查后使用。最多20 GPU会话小时，每工作进程最多3小时；按完整组主动切段可继续，总成本保留。

```bash
ROOT=/mnt/data_nas/deeprobotics/daiyang/RevealVLN_q35n_v16_fitdev_training_unblock_v1
V="$ROOT/projects/qwen35_indoor_nav/research/continuation_memory_v1/evidence_state_policy_v1/monotonic_holdout_v1"
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
V16_STANDALONE_PYTHON="$PY" "$PY" -I -S -B "$V/standalone.py" start monotonic-holdout-20260921-01 -- "$PY" -I -B "$V/pipeline.py" --config "$V/PROTOCOL.json" --run-id holdout_001
```

恢复时使用新服务名、同 run-id，并追加 `--resume`；配置/源码/资产/模型SHA须相同。只对与得分无关的基础设施错误恢复，不重跑已完成低分组。

进度：服务器 http://127.0.0.1:18770/ 。本轮无需Codex在线；服务不承诺跨断电或管理员停止。完成后自动复核、发布GitHub；上传失败与实验完成分开记录。

```bash
systemctl status q35n-monotonic-holdout-20260921-01.service
tail -n 60 "$V/standalone_jobs/monotonic-holdout-20260921-01/job.log"
cat "$V/runs/holdout_001/STATUS.json"
```

正负结果都报告，禁止更换已看成绩的测试屋。支持有限新屋信号需要主终点识别界下界为正，至少三屋、两种子方向为正且计划完整；辅助指标的权衡照实报告，不自动采用。四屋和模板任务不足以宣称广泛泛化、普通导航收益或CVPR录用。
