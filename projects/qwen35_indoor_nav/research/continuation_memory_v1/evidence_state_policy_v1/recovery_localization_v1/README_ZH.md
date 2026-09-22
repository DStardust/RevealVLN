# ORIGINAL 恢复定位与当前正向证据

[完整报告](runs/diagnosis_001/REPORT_ZH.md)；[证据账本](runs/diagnosis_001/EVIDENCE_LEDGER.json)；[下一修复目标与事件数据审核](runs/diagnosis_001/NEXT_STEP_WITH_EVENT_AUDIT.json)。

本轮已经执行，不是仅提出方案：3个冻结 ORIGINAL 模型、1800次真实CPU序列前向、73776个因果递归步骤，关联192条已封存主任务。无新GPU工作、仿真或优化器更新。5项诊断CPU测试通过。

开发集正向信号是 MONOTONIC 主任务66/192 vs DIRECT55/192（+5.73pp），控制98/192 vs78/192。四屋后续优势较弱且分母未全部识别，尚非稳健论文结果。最新ALIGNED教师修复47/192 vsORIGINAL66/192，不采用。

新发现：DEV缺失事件条件的68/96在接管时就累计了错误历史；FIT首次真实事件检出354/354，DEV30/96；已检出后的丢失0。对应原始全历史、缓存模型输出与现场日志均有引用。0.5只用于描述性分类，不是新部署阈值。房屋、历史和种子有关联，不能把这些计数当作独立泛化样本。

`EVENT_INDEX.json` 从900条既有物理轨迹核验导出10329个去重输入×角色标签；不是新采集数据。原始完整动作教师保留，未来事件读出不进入actor。`EVENT_SUPPORT.json` 分列FIT/DEV、可见尺度、困难负例及目标词/房间组合覆盖。下一步优先事件识别迁移及误报累计，不继续截短动作监督。

复算须给新输出目录，避免覆盖已封存结果：

```bash
PY=/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/.envs/q35n_qwen_g2_v1/bin/python3
CUDA_VISIBLE_DEVICES='' "$PY" -I -B diagnose.py --output runs/recheck_001
"$PY" -I -B event_support.py runs/recheck_001
"$PY" -I -B report.py runs/recheck_001
```

实际诊断服务：`q35n-recovery-localization-20260922-01.service`。独立监控：`q35n-recovery-localization-monitor-20260922-01.service`，服务器 `http://127.0.0.1:18770/`，保留前一实验终态入口 `/last-experiment`。

模型、数组、缓存和旧失败记录均只读；`INPUT_MANIFEST.json` 固定已消费证据及来源SHA。公开索引包含原始资产的路径/哈希，不包含许可场景或RGB/语义数组。此处没有新训练结果、普通VLN-CE增益、论文创新或录用承诺。
