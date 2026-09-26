# 新记忆训练与闭环接口

本目录实现 CURRENT 与 DELTA 两个同容量头。两者共享 8×64 递归状态、实际上一动作编码、单位范数读出和全部参数形状；唯一结构差别是额外投影读取当前混合特征，还是当前减上一时刻的混合特征。混合特征含视觉、指令和动作编码，不能称作纯视觉后果。两者都不是已验证的导航改进。

`model.py` 为模型，`data.py` 读取实际补采缓存，`train.py` 为可恢复训练，`runtime.py` 和 `evaluate_worker.py` 对接原 StreamVLN/Habitat 的真实生成与环境执行。所有旧源码和当前任务保持只读。

## 已落实的接口

- 每次实际到达的新观测更新一次记忆。上一动作必须是实际执行动作。STOP 不产生新观测。
- 同一查询最多四个动作 token 使用查询开始时的记忆；各 token 使用自己真实前向的 actor feature，不能读取生成后才到达的观测。
- 保留原完整词表、EOS 和空输出 STOP 处理。四类 argmax 是动作诊断；日志另记实际生成 token 和环境执行动作，不把两者混为一谈。
- 原来的动作加权 CE、普通动作 CE 与 preservation KL/margin 系数保留。权重仅由补采后的 FIT 已知标签重算。两臂复用原始初始化中的共有参数和同一新增参数初始化、同一 schedule，固定 42/43/44 三种子，各 3000 步。最终检查点不按 DEV/unseen 成绩选择。
- 缓存中的原始 `new_training_admission=false` 保留。完整封存、真实重放与 FIT/DEV 隔离通过后，训练准备阶段生成新的数据准入记录；未完成补采时直接报告缺项。
- 每 200 步和暂停边界保存模型、优化器、全部 RNG 和 schedule 游标。FINAL 是完成点，缺少结果回执可以从其重建。

## 验收边界

CPU 测试覆盖真实旧 FIT 特征上的更新和精确断点恢复，也覆盖合成全 token 数据的因果索引、真实执行器适配接口、配对审计。旧缓存测试明确为 `OLD_QUERY_ONLY_CPU_SMOKE`，不冒充新的全 token 已采集或已训练。

新底模前向、真实 Habitat 闭环、正式六模型训练尚需补采完成后的 GPU 验证。接口 CPU 通过不表示 SR 提升。新增代码不接管当前 V4 或补采服务。

## 独立运行入口

在本目录设置 `HERE`，`BASE` 为其父目录，`PY` 为项目的 `.envs/b33_streamvln_v1/bin/python`。补采完成后先检查：

```bash
"$PY" -I -B "$HERE/train.py" --run "$HERE/runs/train_001" \
  --capture-run "$BASE/execution_data_v1/recapture_v1/runs/capture_001" --prepare-only
```

正式训练通过既有独立服务入口提交，指定已经核对可用的 GPU UUID。无需新增守护进程或依赖会话持续在线：

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" V16_STANDALONE_PYTHON="$PY" \
  "$PY" -I -S -B "$BASE/standalone.py" start "$JOB" -- \
  "$PY" -I -B "$HERE/train.py" --run "$HERE/runs/train_001" \
  --capture-run "$BASE/execution_data_v1/recapture_v1/runs/capture_001"
```

恢复使用新的 `$JOB`，相同训练目录并加 `--resume`。本轮没有提交这个正式训练任务，现有评测和补采优先。

六个最终模型完整后，生成闭环运行身份与源锁：

```bash
"$PY" -I -B "$HERE/prepare_evaluation.py" \
  --training-run "$HERE/runs/train_001" --run "$HERE/runs/unseen_001"
"$PY" -I -B "$HERE/evaluate_worker.py" \
  --run "$HERE/runs/unseen_001" --output "$HERE/runs/unseen_001/evaluation/session_001" \
  --ids 0,1
```

后一个是分片工作器接口，生产调度须使用独立服务并登记全部分片；两条不能冒充全量。每个 episode 必须完成原生及六个头、通过前缀审计和会话参数封存，才能计入完整组。未完成组不能跨会话拼接。

## 最终比较

`EVALUATION_PLAN.json` 在新头训练前固定原 V4 的全部 **369 条 val_unseen 路线**，七臂共 2583 次执行。此清单已经暴露，不能称为新的盲测。主比较为同种子 DELTA−CURRENT，同时报告相对同进程原生策略的 SR/SPL、步数、改正/改错和成本。训练数据与动作 token 覆盖的共同修复，不能单独归功于 DELTA；CPU 结果不能替代这次 unseen 比较。
