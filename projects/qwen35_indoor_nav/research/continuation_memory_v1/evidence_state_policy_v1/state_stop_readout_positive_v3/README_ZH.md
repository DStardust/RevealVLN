# 单向 STOP 修正 V3

唯一策略变化是 `delta = max(0, frozen_readout(...))`。冻结现有113参数读出、原MONOTONIC三个种子与best4k，0新增训练。三种运动动作分数不变；原MONOTONIC已选择的STOP保持STOP。保护对象是MONOTONIC最终策略，不是原始Qwen的STOP。未来查询、精确任务状态和位姿不进入actor。

对照：原MONOTONIC vs STOPPLUS，三种子，原DEV16变体×4历史×2任务×6模型=768次真实续接；每臂主任务192、task_T192。所有旧结果只读，本轮不复用旧导航轨迹。输入、传感器、动作、500决策和安全成功定义不变。完整六模型组封存，384次前缀比较；失败attempt保留。

CPU测试验证负残差不再否决STOP、正残差保留运动分数、旧模型权重/记忆不变，并复核上轮5个失败和1个新增成功的首次分歧输入。这不是6次新导航，也不提前宣称成功率提高。新增过早STOP仍可能造成失败，需完整实测。

源码及协议在首次运行前冻结。本轮只做一次固定对照，不搜索阈值、不改教师标签、不自动扩数据或训练。主差值、配对胜负、task_T、未满足STOP、碰撞和全分母成本均报告；仅原暴露DEV开发证据，不自动采用或宣称论文创新。

独立后台服务使用原机制；原监控 http://127.0.0.1:18770/ 自动刷新。评测结束后自动复核、上传GitHub并恢复2–7卡占位。服务不依赖Codex或终端。

```bash
export V16_STANDALONE_PYTHON=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
"$V16_STANDALONE_PYTHON" -I -S -B "$PWD/standalone.py" start state-stop-positive-20260922-01 -- "$V16_STANDALONE_PYTHON" -I -B "$PWD/pipeline.py" --config "$PWD/PROTOCOL.json" --run-id positive_001
```

已运行的服务不要重复提交。基础设施中断后使用新服务名、原run-id、--resume；先核对源码/配置/权重绑定，只恢复未封存的完整组。
