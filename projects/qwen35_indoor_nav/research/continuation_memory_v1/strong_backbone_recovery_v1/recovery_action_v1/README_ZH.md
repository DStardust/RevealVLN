# 真实恢复动作与历史纠错结构对照

本轮执行用户批准的下一步：先提升可用纠错动作，再比较结构。上一轮 Ours gate 的 unseen200 SR 为56%，与原生相同；这些记录只读保留。本目录代码和实际产物不能提前视为有效贡献。

新集合固定334条件：FIT 142个原生失败条件（117条路线）＋128个原生成功条件；DEV 32个失败＋32个成功。失败历史来自既有官方TRAIN数据，取最后一个不超过64步的原生模型查询点。先真实执行这段历史并逐帧核对RGB，再以离线路径教师实际执行至目标。全程最多500决策，主动STOP才成功。没有中途传送；教师失败保留并不提供伪标签。教师以目标位置生成普通R2R恢复动作，不宣称完成额外顺序语言程序。

成功轨迹提供普通动作保持监督。教师轨迹提供实际恢复后缀动作；错误的原生前缀只供因果记忆展开，不标作正确动作。查询标签在模型产生当前特征后才用于teacher forcing，后续动作不进入当前读出或记忆。训练的每一个监督切点都沿真实历史完整反传到起点，基座冻结。

两臂共享同一初始state_dict、采样顺序、3000次更新、优化器和损失。CONCAT直接拼接当前特征与8×64记忆；EVIDENCE加入按当前决策读取记忆、并减去空记忆下的动作读出。两个模型存储相同参数集合；CONCAT不使用query/key路径，实际参与计算的参数和成本并不相同。这次比较整个结构组合，不能分别归因给attention或差分；差分公式不自动证明用到了有价值的旧事件。没有介入开关、没有交叉结果loss，也不把此结果算作Ours监督贡献。

训练只用官方TRAIN，所有新增头从零初始化；不加载EU6或旧任务头。缓存与现场共用同一base/processor/dtype/输入和实际动作边界。训练动作logits保持现场FP32残差加法（生成器将BF16基座输出转换为FP32）。DEV只在固定最终训练结束后诊断，不搜索阈值或选checkpoint。随后自动运行原生＋两种新头的同进程配对，在固定已暴露unseen200上执行600次真实导航。这里不是完整1839，也不是盲测，单种子结果不是论文验收。

独立执行：

```bash
PY=/mnt/data_nas/deeprobotics/daiyang/vla/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3
BASE=projects/qwen35_indoor_nav/research/continuation_memory_v1/strong_backbone_recovery_v1
"$PY" -I -S -B "$BASE/standalone.py" start strong-action-architecture-20260925-01 -- \
  "$PY" -I -B "$BASE/recovery_action_v1/pipeline.py" --run "$PWD/$BASE/recovery_action_v1/runs/action_001"
```

中断后使用新service名、同run路径和`--resume`。源锁、参数、配置与数据身份必须一致；仅完整封存组进入汇总，每200步优化器/RNG断点恢复，未完成评测组完整重跑。最多64 GPU会话小时、24墙钟小时、32GiB新产物；只使用已授权设备，自己的占位通过既有控制器释放并在结束后恢复。

监控继续使用原网站，新增最上方“真实恢复动作”卡片；旧评测显示最终200条结果。训练、教师成功、CPU测试、unseen收益分开报告。若两种新头共同提高，数据贡献与结构增量分别讨论；若EVIDENCE不优于CONCAT，不宣称复杂结构有效。
