# FIRST 发现集完成与固定权重扩大复验

本次存档包含已完成 80 条发现集（13 臂、1040 次执行），以及已启动的 289 条复验（7 臂、计划 2023 次执行），两者不能混成同一个未经选择的盲测结果。

发现集：DELTA-FIRST 和 CURRENT-FIRST 三种子平均 SR 都为 55%，原生 53.75%；DELTA-FIRST 相比自身 ALL 为正，但相对 CURRENT-FIRST 尚无平均 SR 增量。DELTA 的总动作减少与成功路线 SPL 小幅下降同时存在，不能只宣称更高效。

- [诊断与新入口](../../research/continuation_memory_v1/strong_backbone_recovery_v1/execution_adaptation_v1/first_confirmation_v1/README_ZH.md)。同目录 DISCOVERY_DIAGNOSIS.json 与 DISCOVERY_REPORT_ZH.md 提供逐路线证据。seed 44 的主要损失是未救回 CURRENT 救回的原生失败，并非都破坏原生成功或 STOP 过多。
- [完整发现集结果](../../research/continuation_memory_v1/strong_backbone_recovery_v1/execution_adaptation_v1/scope_validation_v1/runs/scope_001/unseen/RESULT.json)。完整轨迹、身份和封存证据在 [日志包](DISCOVERY80_SEALED_LOGS.tar.gz)，路径相对仓库。
- 扩大复验：原 369 减去发现集 80 的全部 289，8 屋，种子 42/43/44 全保留，六个已有 head，0 新训练。已有其他方法结果暴露，不能称为盲测。结束后另给 369 条描述性汇总。
- 独立服务：q35n-strong-first-confirmation-20260926-01.service；监控为原 SSH 转发的 http://127.0.0.1:18770/ ，API /api/first_confirmation。首个七臂组通过后自动使用 GPU 0–7；无需 Codex 在线。
- launch_snapshot/ 仅是提交时运行快照，不能代表最终完成。查看服务器对应 runs/confirm_001/STATUS.json 得到最新进度。
- 新 CPU 3 项、真实上游导入与已有 head CPU 加载、8 个监控路由验收均通过；这不等于新导航效能通过。基座已开始真实自主执行，具体启动身份见快照。

张量权重、RGB/场景、特征等保留在服务器，源码锁登记路径/SHA。日志包仅包括已经完成的 80 条发现集，不将进行中的新结果伪装为完整效应。旧 CONCAT 候选保留。
