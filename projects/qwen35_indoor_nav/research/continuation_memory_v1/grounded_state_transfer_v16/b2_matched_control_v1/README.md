# B1 / B2 / Terminal-only 匹配对照

已实现独立八卡流水线：共享真实 Qwen 特征缓存 → 九模型训练 → 固定最终模型诊断 → 八卡完整组续接 → CPU 重算。此文件是运行说明，效能结果以 runs/matched_001/RESULT.json 为准。

新增覆盖与原数据合并为 75 个变体，仍来自 40 个原始父族、5 个已暴露开发房屋；FIT 59 变体/32 父族，DEV 16 变体/8 父族。父族均衡的共享 1200 步 schedule，三个 seed，B1/B2/Terminal-only 仅辅助监督不同。全部三臂共享动作标签、普通动作监督、初始状态与数值路径。

每卡训练一模型，GPU0 另外训练第九个模型。评测每卡 16 个完整九模型组；共 128 组、1152 次自主续接。每臂主任务 192 条、控制任务 192 条，分别报告接管时 terminal present/absent 和 seen/missing 历史。各变体与种子不作为独立房屋样本。底模不更新，不据此声称普通 VLN 或论文泛化收益。

特征仅补算未命中已封存缓存的新输入；统一在 GPU1 完成此公共阶段。GPU0 外部任务保留，只使用可用余量。GPU2–7 经现有占位租约释放，结束或异常时释放租约并核验占位恢复。

## 查看进度

在本目录运行：

```bash
cat LAST_JOB.txt
cat runs/matched_001/STATUS.json
cat runs/matched_001/FEATURE_PROGRESS.json
cat runs/matched_001/TRAIN_PROGRESS_*.json
cat runs/matched_001/EVALUATION_PROGRESS_*.json
```

尚未到达的阶段没有对应进度文件。`standalone_jobs/<LAST_JOB>/STATUS.json` 记录服务退出情况，`runs/matched_001/attempts/` 保存分阶段、逐 GPU 日志。训练每 200 步保存完整优化器/RNG；评测只有完整九模型组可续用。独立 systemd 服务无需 Codex 在线。

## 恢复

基础设施异常先查看失败日志，保留所有 attempt。使用新服务名指向同一 run-id 并加 `--resume`，源码/配置/数据身份须保持相同。已完成低分模型或条件不能重跑挑分；改策略或评测语义需要新版本。

实际 CPU 验收记录见 CPU_TEST_RESULT.json；真实 GPU 训练/评测证据由流水线写入，CPU 通过不代表方法收益。
