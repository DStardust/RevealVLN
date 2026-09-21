> 指标更正：请先读 [CORRECTION_ZH.md](CORRECTION_ZH.md)。初版 false_stop 统计的是教师动作不一致，不能直接解释成任务错停；原拒绝记录保留。六份权重已冻结用于同级 state_stop_readout_eval_v2 的真实闭环。

# 状态到 STOP 的局部读出修复 V1

针对真实诊断中的“预测状态正确、动作仍漏停”，只训练113参数的STOP残差。原MONOTONIC三个种子的记忆、事件预测器、动作分支与best4k全部冻结。三个运动动作分数逐值保持不变；无oracle STOP、无动作禁用、无新标签。

训练：原FIT59变体和原ordinary FIT；3种子各1200步，继承原共享schedule、普通动作CE和完整native KL，权重均为1。只读缓存严格因果原模型预测，未来查询和真值状态不进入读出。每200步保存优化器及完整RNG。旧新屋测试数据不参与训练或检查。

固定最终checkpoint完成后，自动检查FIT/DEV漏停、错停和ordinary CHECK退化；规则见PROTOCOL。失败会保留实际结果并结束，不自动改权重重训。通过后，在原暴露DEV16变体×4历史×2任务×6模型上执行768次真实续接，完整六模型组封存；主任务每臂192，task_T单列。历史结果不会接入本次分母。

本轮已完成3600次读出更新，局部检查未通过：DEV漏停245→241，错停154→189；闭环0/768，未采用。详见 runs/repair_001/MEASURED_REPORT_ZH.md。这是工程修复结果，不是独立测试或论文创新证据。修复前完整诊断见上层monotonic_diagnosis_v1。

## 独立运行

使用PROTOCOL.json中的standalone_python；从本目录执行：

```bash
export V16_STANDALONE_PYTHON=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
"$V16_STANDALONE_PYTHON" -I -S -B "$PWD/standalone.py" start state-stop-repair-20260921-01 -- "$V16_STANDALONE_PYTHON" -I -B "$PWD/pipeline.py" --config "$PWD/PROTOCOL.json" --run-id repair_001
```

已提交的服务不要再次启动。进度页面仍为 http://127.0.0.1:18770/ ，10秒刷新；也可读 runs/repair_001/STATUS.json 和 TRAIN_PROGRESS_*.json。独立systemd运行不依赖终端或Codex额度。停止/完成后只清理本任务，恢复原2–7卡占位；自动发布最终代码与日志到现有GitHub分支。

基础设施中断：先保留失败证据，用新服务名、原run-id及 --resume；校验源码/配置/数据/模型绑定后恢复完整优化器或剩余完整组，不能重跑低分已完成组。
