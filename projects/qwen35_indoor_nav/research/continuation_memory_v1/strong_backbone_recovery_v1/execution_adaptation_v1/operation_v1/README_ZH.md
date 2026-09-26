# 补采后的训练与 unseen 评测

2026-09-26 已启动真实 GPU 训练。独立服务为 `q35n-strong-execution-adaptation-20260926-02.service`，运行目录为 `../runs/experiment_001/`。这份启动记录没有新的 SR 结果。

## 已完成与本轮比较

- 补采 334/334 条既有轨迹，覆盖 28,859 个实际动作位置；补齐原先缺失的 21,396 个动作特征及 133 个 STOP 特征。独立路线新增数为 0。
- 单独训练验收通过：270 条 FIT、64 条 DEV，14,886 个已知 FIT 动作标签；保留 unknown mask，DEV/unseen 不进入训练损失。原始资产的历史 training_admission 字段不改写。
- CURRENT/DELTA 各三个种子（42/43/44），每模型 3,000 次更新，六卡并行。共享初始化、调度、数据、损失和容量；StreamVLN 底模冻结。CURRENT 写入当前特征，DELTA 写入相邻观测特征差；特征包含指令与动作上下文，不能宣称是纯视觉变化。
- 全动作位置覆盖与新的逐 token 残差接口是共同变更，不能将它们相对旧实验的变化单独归因于 DELTA。
- 已有 CONCAT 小幅正向结果保留为工程候选；新版本不会自动取代它。旧分数也不会冒充本轮同进程对照。

## 自动执行与监控

流水线自动推进 TRAIN → PREPARE_UNSEEN → FIRST_UNSEEN_GROUP → UNSEEN → REVIEW。每 200 步保存可恢复训练检查点。第一条真实七臂闭环组通过完整性与输入/动作审计后自动继续，不根据得分决定是否继续。

最终清单为固定、已经暴露的 369 条 val_unseen 路线：NATIVE 加六个轻量模型，共 2,583 次执行。完整组封存后才计入正式配对；不称完整 1,839 条或盲测。评测最多使用 0–7 号卡。共享 GPU 时延单列解释。

原监控地址不变：`http://127.0.0.1:18770/`，新接口 `/api/execution_adaptation`。原补采、路线复验和历史页面仍可访问。服务不依赖 Codex 在线。

```bash
ROOT="$(git rev-parse --show-toplevel)"
BASE="$ROOT/projects/qwen35_indoor_nav/research/continuation_memory_v1/strong_backbone_recovery_v1"
systemctl status q35n-strong-execution-adaptation-20260926-02.service
cat "$BASE/execution_adaptation_v1/runs/experiment_001/STATUS.json"
tail -n 80 "$BASE/standalone_jobs/strong-execution-adaptation-20260926-02/job.log"
```

仅在服务已退出且原因已排除时，用新服务名指向原 run 并添加 `--resume`；不能启动第二个生产者：

```bash
CONTROL=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
JOB="strong-execution-adaptation-resume-$(date -u +%Y%m%d-%H%M%S)"
"$CONTROL" -I -S -B "$BASE/standalone.py" start "$JOB" -- \
  "$CONTROL" -I -S -B "$BASE/execution_adaptation_v1/operation_v1/pipeline.py" \
  --run "$BASE/execution_adaptation_v1/runs/experiment_001" --resume
```

训练累计上限 12 GPU 会话小时，评测 64 GPU 会话小时，总墙钟 24 小时；新增产物 20 GiB。源锁、模型与数据身份一致才能恢复。训练恢复优化器与 RNG；评测只保留完整封存组，未封存尝试留档。不按分数重试，只清理身份核验属于本任务的进程，退出释放已有占位控制器租约。

## 验收与已知尝试

新增 5 项 CPU 编排测试通过，覆盖不完整分母、配对胜负、进程所有权、恢复冲突和最终权重完整性；已有模型/数据/运行时/训练的 36 项 CPU 记录保持原有范围。新监控的六个路径在临时端口和正式端口均实测 HTTP 200。它们都不代替实际闭环验收。

首次服务 `...-01` 使用了相对路径，独立服务 cwd 下找不到 pipeline，0 GPU、0 参数更新即退出；记录保留。`...-02` 改用绝对路径后真实训练，启动快照显示六模型已经更新且梯度有限。没有修改策略、数值路径或选择更好分数的重试。

入口证据：`../runs/experiment_001/{PROTOCOL,SOURCE_LOCK,LAUNCH_RECEIPT,TRAINING_STARTED_SNAPSHOT}.json`；训练身份与准入：`../runs/train_001/{TRAINING_CONFIG,DATA_ADMISSION}.json`。完整特征、检查点和私有逐步日志保留在服务器；GitHub 启动归档不等于完整权重/场景可下载，也不声称评测已经完成。
