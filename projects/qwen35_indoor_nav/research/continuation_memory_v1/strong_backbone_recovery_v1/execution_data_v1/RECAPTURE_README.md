# Chunk后续动作的因果补捕获计划

`token_index/ACTION_TOKENS.jsonl` 的 `memory_feature_index` 对齐的是**实际物理时刻**。它适合动作后观测变化审计，不能直接作为原四动作chunk后续token的actor输入：生成该chunk时，后续物理动作还没有执行，其RGB和记忆更新也尚未可见。

`recapture_plan.py` 把缺actor feature的位置按原query分组。整个chunk只能读取递归重放至 `query_start` 的记忆，即 `memory_features[:query_start+1]`；后续token可包含同一chunk里已经生成的动作token，但不能包含动作后来执行后的新RGB、更新记忆或结果。输出将 `allowed_memory_feature_index=query_start` 与 `physical_feature_index_audit_only=实际step` 明确分开。

这一步只生成请求。没有新actor feature，没有仿真，没有新训练。原query首动作的known mask、实际动作所在监督区间两个字段分别原样保留，不把unknown失败前缀升级为已知标签，也不因生成请求而自动准入后续token训练。

当前部署只修改query的第一个动作。即便随后完成所有后续token的feature捕获，也不能直接声称已经修复上线STOP；让residual修改所有token属于新版本，必须对所有比较臂共用，同时重新登记数值/输入和评测边界。旧代码、索引及现有评测保持不变。

从 `strong_backbone_recovery_v1` 运行：

```bash
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
"$PY" -I -S -B execution_data_v1/test_recapture_plan.py
"$PY" -I -S -B execution_data_v1/recapture_plan.py
```

产物在 `recapture_plan_001/{REQUESTS.json,RESULT.json}`，目录存在时拒绝覆盖；复算需显式指定新的 `--output`。RESULT列出全部缺位、原监督掩码统计和重点终止STOP子集。原mixed feature包含视觉、指令和上一动作embedding，不是纯视觉变化。
