# FIRST 固定权重复验

状态：已实现，运行事实见 `runs/confirm_001/STATUS.json`。本轮不训练、不修改 FIRST 规则，不把接口测试写成效能结果。

已完成的 scope80 同时包含 ALL/FIRST 和三个种子。DELTA-FIRST 与 CURRENT-FIRST 平均 SR 均为 55%，原生为 53.75%；DELTA 尚未证明优于 CURRENT。这里测试原 369 条清单去除发现集 80 条后的全部 289 条，按原顺序固定；原生及两种 FIRST 的三个种子共 7 臂、2023 次执行。剩余路线已有其他方法结果暴露，不能称作新盲测。

`prepare.py` 固定分母、六个已有 head、基座与源码。`pipeline.py` 先完成一个真实七臂组，再在可用的 GPU 0–7 上并行推进；每组在同一进程内运行，完整组封存后才能计入结果。无新优化更新，不按分数重试。`evaluate_worker.py` 延续真实 StreamVLN/Habitat 执行路径，`scope_processor.py` 与旧版本逐字节一致。

主报告是剩余 289 条；完整后另生成 369 条描述性汇总，保留 80 条曾参与 FIRST 选择的事实。成功率、救回/改错、双方成功时 SPL 与动作变化、全分母预算成本、各屋与种子分别报告。动作总数降低不自动意味着成功路径更高效。没有自动部署或更换训练规则。

运行入口：项目控制 Python 执行 `prepare.py`，随后通过上层 `standalone.py start NAME -- PY -I -S -B pipeline.py --run runs/confirm_001` 提交独立服务。中断后使用新服务名、相同 run 路径和 `--resume`；已完成组不重跑。实际绝对命令见 `LAUNCH_RECEIPT.json`。监控为原网站的 `/api/first_confirmation`，CPU 测试为 `test_followup.py`。

本地模型、场景、特征和私有完整轨迹仍保留在服务器；GitHub 提供代码、配置、来源指纹、诊断及紧凑结果，不能仅凭仓库中的文本宣称复现了模型。
