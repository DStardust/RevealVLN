# EVENT 修复的首分歧定位

当前有效结果：[报告](runs/diagnosis_002/REPORT_ZH.md)。本目录只读分析 event_grounding_repair_v1 的封存数据，未启动训练或导航。

- `runs/diagnosis_002/CASES.json`：全部192主任务配对、实际首分歧、模型分支分数及有界CPU复算。
- `runs/diagnosis_002/LOST_PAIRS.csv`：23个退化配对的索引。
- `runs/diagnosis_002/INPUT_MANIFEST.json`：实际输入/权重/封存组来源。
- `runs/diagnosis_001/`：缓存键命名空间修正前的诊断保留，不能用于缓存可用性结论。

运行命令（使用新的输出目录，不能覆盖已有记录）：

```bash
PY=/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/.envs/q35n_qwen_g2_v1/bin/python3
"$PY" -I -B tests.py
"$PY" -I -B diagnose.py --output "$PWD/runs/diagnosis_NEW"
"$PY" -I -B report.py "$PWD/runs/diagnosis_NEW"
```

动作组件交换仅回答同输入下哪一部分分数改变了动作；不是新增策略、没有counterfactual rollout或新成功率。场景仍是已暴露单屋，不能宣称泛化。
