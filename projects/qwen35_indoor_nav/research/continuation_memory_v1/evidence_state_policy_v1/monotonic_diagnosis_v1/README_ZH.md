# MONOTONIC 新屋测试后的只读诊断

实际结论与证据：[REPORT_ZH.md](runs/diagnosis_001/REPORT_ZH.md)。

已复核1488条完整续接、98,792次自主决策；另用六份冻结模型在原59个FIT变体上进行2832次接管前向，权重不变。无新Qwen前向、无优化器更新、无仿真、无GPU消耗。旧48个缺失槽位及所有结果原样保留。

这一步定位状态到动作、特别是STOP的读出问题，并区分事件识别和闭环支持不足；不是一次新的方法效能实验。具体待检验修复见 [NEXT_REPAIR.json](runs/diagnosis_001/NEXT_REPAIR.json)，未自动启动新训练。已查看的四个测试屋标为暴露，后续调试不能再称新盲测。

复算（新的输出目录，不覆盖已完成证据）：

```bash
ROOT=/mnt/data_nas/deeprobotics/daiyang/RevealVLN_q35n_v16_fitdev_training_unblock_v1
D="$ROOT/projects/qwen35_indoor_nav/research/continuation_memory_v1/evidence_state_policy_v1/monotonic_diagnosis_v1"
PY=/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/.envs/q35n_qwen_g2_v1/bin/python3
CUDA_VISIBLE_DEVICES='' "$PY" -I -B "$D/diagnose.py" --output "$D/runs/recheck_001"
CUDA_VISIBLE_DEVICES='' "$PY" -I -B "$D/fit_probe.py" --output "$D/runs/recheck_001/FIT_PROBE.json"
CUDA_VISIBLE_DEVICES='' "$PY" -I -B "$D/report.py" --output "$D/runs/recheck_001"
```

实际CPU诊断使用同目录独立systemd服务运行，完成记录在`standalone_jobs/monotonic-diagnosis-20260921-01/`和`monotonic-fit-probe-20260921-01/`。日志中的`COMPLETE`仅代表诊断程序完成。监控 http://127.0.0.1:18770/ 保留最终测试进度，新增 `/diagnosis` 报告入口。

原始逐步日志由旧实验的封存组SHA核验；`INPUT_MANIFEST.json`保留来源引用。此目录只保存派生诊断，不复制场景、底模或原始RGB/语义数组。已发布六份小模型的SHA及加载后不变检查均保留。
