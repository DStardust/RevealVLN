# V16 网页审阅入口

仓库 DStardust/RevealVLN；分支 `codex/q35n-grounded-state-v16-20260920`。
路径 `projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16/`。
既有证据基线 `754d574bc98fd49fe30b2be76e30cbc53f0f25c3`，旧 V15 保留不改。

先读本目录 `REPORT_ZH.md`、`RESULT.json`、`GITHUB_HANDOFF.json`。
最后一个文件指向带索引和 SHA 的上传快照；时间点与服务器实时状态不同。
当前实际执行入口为 `runs/v16_formal_001`，独立服务为
`q35n-v16-formal-20260920-01.service`。服务会自行推进采集、特征、训练、
评测和复核，结束后尝试推送结果；失败会保留原因，不把未运行条件改成 FAIL。

本轮仅比较共享覆盖修复后的 B1/B2/Ours。冻结 best4k、现有 8×64 记忆、
2048 维因果特征、输入窗与 SEE2；共同取消 native STOP 强制覆盖，采用
method logits argmax。旧碰撞 UNKNOWN 标签保留，新安全终点要求完整证据、
严格事件顺序、主动 STOP、总决策不超过500、全程无碰撞。
共同数据、STOP 和评测修复不归因于 Ours。

计划 FIT16 / DEV2 / TEST8 族，三臂三种子各1200步，主任务每臂 N=192；
task_T 另计144次，合计720次自主续接。主要增量为 Ours−B2，δ=10个百分点，
未知分配界、屋/种子方向、B1和匹配记忆诊断共同约束判读。没有匹配机制
子集时明确写不可辨识。结果不足不自动扩大实验或换检查集。

可直接检查 `pipeline.py`、`build_data.py`、`objective.py`、`train.py`、
`select_action.py`、`evaluate_continuations.py`、`continuation_service.py`、
`evaluator_v16.py`、`review.py`、`diagnose.py`。
B2 的 query 诊断用精确四位状态解析组合，不能用其未训练的 query head。
CPU 梯度/恢复通过、数值先导通过与方法收益必须分别判断。

日志包包括失败采集、基础设施中断及参数/输入身份记录。原始 RGB/语义
数组、底模、缓存与完整 optimizer/RNG 二进制在服务器上，公开快照提供
已核验的引用/哈希。网页读到这些记录不等于独立加载或重现实验。
本入口不声称已有正向方法收益、论文贡献、R2R SR40或真机部署。
