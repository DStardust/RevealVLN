# 评测运行器修复 R1

原 `gpu_v1/runs/gpu_001` 已完成九模型×1200更新及缓存诊断。首次启动物理服务时，旧 V16 `ContentStore` 仅允许写入旧 V16 根，因本实验为其兄弟目录而触发 `CONTENT_PATH`，导致连接重置。原失败发生在首次环境 reset 前：0条完整续接，0个完整配对组。不是训练失败，也没有方法效果结论。

本目录是同一实验的基础设施修订。复用九个 final1200，逐个核验原权重 SHA；数据、schedule、注册清单逐文件一致。新增训练更新为0。原训练、失败日志、源码锁均只读，旧GPU开销也计入累计预算。

`continuation_service.py` 只重写存储器的目录初始化边界，允许本运行器 `runs/` 下真实目录，拒绝符号链接逃逸。数组哈希、无损写入、去重、损坏检查及产物上限仍调用原 `put_array`。模型、传感器、历史、动作、数值前向、500决策和成功定义不变。新源码锁同时绑定旧源文件与新运行文件。

`prepare.py` 复用并验证已完成的模型和诊断，不重新训练；其他执行代码版本化复制以保留原冻结文件。`PROTOCOL.json.runtime_revision` 和 `TRAINING_REUSE.json` 明确修订与继承关系；原实验累计10800更新不是本次新增更新。

四项CPU验收通过：真实Habitat Python服务的socket握手、存储读回/去重/上限/目录越界、三模式因果前向一致、清理拒绝外部owner。见 `CPU_TEST_RESULT.json`、`CPU_TEST_attempt_002.log`。这不替代真实闭环完成证据。

当前独立服务：`q35n-evidence-state-gpu-20260921-03.service`；run为本目录 `runs/gpu_001`。七卡1–7自动完成评测、复核、只读校准、占位恢复和结果上传。监控仍是服务器 `http://127.0.0.1:18770/`。历史训练loss与优化器文件保留于原 `gpu_v1/runs/gpu_001/train/`。

```bash
systemctl status q35n-evidence-state-gpu-20260921-03.service
```

`standalone_jobs/evidence-state-gpu-20260921-03/job.log` 查看总流程；`runs/gpu_001/attempts/` 查看各卡日志；`evaluate/session_gpu*/GROUP_*.json` 及对应状态封存才表示完整九模型组。只读网页显示完整组分数，半组不混入。

仍是一个已暴露DEV房屋上的控制任务实验，不是独立TEST或普通VLN成功率。当前没有采用决定。恢复按完整组边界，服务脱离Codex；资源/正确性错误会停止并保留证据。
