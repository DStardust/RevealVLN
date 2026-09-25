# Recovery confirmation V2

独立后台复验上一轮 CONCAT 正向信号：三个种子 × CONCAT/LOCAL，六个模型。
旧恢复采集、特征缓存、源码和结果只读复用；新证据均在 `runs/confirmation_001/`。

- `confirmation_model.py`：原 CONCAT 方程；LOCAL 用当前观测写入，不读新增历史。
- `train.py`：共享 FIT 池、同种子初始化/schedule、3000 步、200 步优化器/RNG 断点。
- `worker.py`：原生＋六个模型的真实自主执行；对原生以及每 seed 的两种模型分别做前缀审计。
- `pipeline.py`：独立推进训练、DEV、unseen、复核；按完整组恢复，禁止按分数重试。
- `confirmation_review.py`：固定分母、逐 seed/屋、胜负列表与缺失识别界。
- `test_confirmation.py`：实际 CPU 前反向、LOCAL 历史隔离、初始化/方程兼容、前缀错误及不完整分母。
- `PAPER_STORY_ZH.md`：有数据支持的主线与近邻工作；不把正在运行的实验写成结果。

运行服务：`q35n-strong-confirmation-20260925-01.service`。
监控：<http://127.0.0.1:18770/>，顶部新卡片；JSON 为 `/api/confirmation`。
主日志位于上一级 `standalone_jobs/strong-confirmation-20260925-01/job.log`。
完成与否以新 run 的 `STATUS.json` 和 `RESULT.json` 为准。

```bash
systemctl status q35n-strong-confirmation-20260925-01.service
```

初次启动已提交。不要重复启动。若真实基础设施中断，使用新服务名指向相同 run 并加 `--resume`；源码、配置、共享数据和权重身份必须相同。旧半组保留在 failed_attempts，重新运行完整组。服务不依赖终端或聊天 API；不承诺跨断电恢复。结束时通过原 GPU 占位控制器释放 lease，由控制器恢复用户占位。

当前预算沿用已授权的资源管理上限：64 GPU 会话小时、24 小时墙钟、32 GiB 新产物；不是完成时间预测。GPU0–7 可用时共享使用，训练六卡，评测最多八卡。
