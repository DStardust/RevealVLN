# 当前运行入口：训练完成，真实 unseen 评测已启动

2026-09-26：六个 CURRENT/DELTA 模型各完成 3,000 次更新，合计 18,000 次；训练约 0.728 GPU 会话小时，底模更新为零。固定训练结果位于 `../runs/train_001/`。已有正向候选继续保留，没有自动替换。

当前独立服务：`q35n-strong-execution-adaptation-20260926-03.service`。当前运行：`../runs/experiment_002/`。监控仍为 `http://127.0.0.1:18770/`，接口 `/api/execution_adaptation`。

旧 `experiment_001` 在首组启动导入时失败：本地 `model.py` 占用了 StreamVLN 的 `model` 命名空间，0 个完整导航组。旧运行、源码、源锁及失败日志保留。StreamVLN 的 `model/` 没有 `__init__.py`，因此仅给本地类换加载别名还不足以解决问题；还必须在本地接口解析后，从导入搜索路径移除包含该同名文件的目录。新 worker 采用显式文件别名，保留原模型类的实际源码和训练后参数。

本修订只修复导入并增加 executor 身份记录；执行动作、输入、前向、评测定义、权重、3000 步预算及 369 条 unseen 清单不变。`test_imports.py` 已实际导入官方 evaluator 并在 CPU 加载注册的 CURRENT_s42 权重。随后 GPU 运行写出了 `model_loaded=true` 的身份记录，首条路线已经实际执行动作；此处尚无完整 unseen 效果结论。

新流水线跳过六个已完成模型，复用原有最终权重。旧运行的训练/失败评测 GPU 会话小时与墙钟带入累计预算。`EXECUTOR_BINDING.json` 将新 worker 绑定到运行证据；原 source lock 不被覆盖。监控六条新旧路由实测均 HTTP 200。

后续自动执行：首条七臂完整组审计 → 最多八卡完成剩余组 → 全分母复核。NATIVE 加 CURRENT/DELTA 三种子，共 369×7=2,583 次执行；完成前不发布完整 SR，不按首组分数决定是否继续。数据已经暴露，不能称盲测或完整 1839 条。

查看：

```bash
ROOT="$(git rev-parse --show-toplevel)"
BASE="$ROOT/projects/qwen35_indoor_nav/research/continuation_memory_v1/strong_backbone_recovery_v1"
systemctl status q35n-strong-execution-adaptation-20260926-03.service
cat "$BASE/execution_adaptation_v1/runs/experiment_002/STATUS.json"
tail -n 80 "$BASE/standalone_jobs/strong-execution-adaptation-20260926-03/job.log"
```

只有该服务退出、原因排除后才恢复；用新的 service 名运行 `operation_v2/pipeline.py --run .../runs/experiment_002 --resume`。不要重跑训练或使用 v1 入口恢复当前运行。无需 Codex 保持在线。

训练/评测代码、协议、源码锁、完成训练摘要、模型 SHA 与启动身份归档到 GitHub。大特征缓存、权重文件及私有完整轨迹留在服务器；公开归档不等于所有场景和模型资产均已上传。
