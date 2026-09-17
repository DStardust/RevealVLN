# 动作预算单次耐久提交：独立 CPU 交付

结论：`READY_FOR_MAIN_AGENT_REVIEW`。实现和 21 项纯内存 CPU 测试已完成；**未接入正在运行的 compact_loop_v2，未改其代码、V1、旧规划类或状态文件**。未运行真实 fsync benchmark、仿真、GPU 查询或训练，不声称实际吞吐加速。

## 实现与不变量

[budget.py](budget.py) 提供 `SingleCommitBudgetLedger`，继承从精确路径、SHA256 校验后的原 `planning.py` 读取的 `BudgetLedger`。原始源码保持只读，复用同路径已加载模块以保留异常类型身份。

唯一正常路径变化：原 `reserve_action` 的“check_time 提交 clock-only 快照，再提交增加动作计数的快照”，改为在相同一次时钟读取下检查时限与动作预算，然后**一次提交包含 clock 和动作预扣的完整快照**。`persist` 必须同步完成耐久确认后才能返回。

- 每次合法 reserve：persist 回调从 2 次变为 1 次；稳定状态、返回值、创建/阶段时间与计数保持一致。
- 预算拒绝、非法 count、无 active phase：仍先提交 clock-only 快照后抛出原异常；不增加动作计数。
- 非法/回退时钟：不提交、不预扣；保持旧语义。
- 其他方法与状态 schema 继承原实现；不批量预留、不重置相位、不退款、不自动重试。
- 持久化异常（包括 KeyboardInterrupt）使对象 poisoned；更换回调也不能继续操作。明确要求 persist callback，不提供生产 `persist=None` 分支。
- 可选 helper `step_after_durable_reservation(backend_step, ...)` 先完成一动作预算预扣与耐久确认，再且仅再调用一次 backend；backend 抛错后也不退预算。

## 状态等价的精确边界

**成功和正常拒绝路径的最终快照等价，不是逐条 journal 字节等价，也不是全部故障瞬间内存状态等价。**

当 persist 失败时，新版已经将动作保守预扣到内存；旧版如果在第一个 clock-only 提交就失败，内存动作数可能还未增加。因此该故障状态可能是旧 0、新 1。二者均不会调用 backend，但新版可能多保守占用一个预算，必须如实记录，不得退款或把未获确认的内存快照当作恢复依据。

崩溃后恢复仍必须使用经过单独审查的实际耐久 journal/HEAD 状态，不能仅凭 `snapshot()` 或自称“回调已写过”恢复。变少的是冗余中间提交；不能用伪异步、返回后再 fsync 的 callback 绕过耐久保证。

## 测试结果与局限

[test_budget.py](test_budget.py)：**21/21 PASS，0.006 秒**。包括：

- 正常一次/多动作预留、每调用提交计数；分阶段/总动作上限、发现/认证/总时间精确边界。
- 非法参数、无相位、回退/非有限时钟，成功与拒绝状态和异常的旧类差分。
- 同相位恢复保持原始起点与计数，关闭后不能重开，过期相位可正常关闭，损坏状态/改预算恢复拒绝。
- persist 成功确认先于 backend；失败前/可能写入后等六类**内存回调故障模拟**均不调用 backend，后续一直 poisoned。
- persist 中断、拒绝分支 persist 失败、移除 callback、深拷贝隔离、backend 异常不退款。

这些是预算适配器的 CPU 控制流测试，**没有实际注入原 Journal 的文件系统写入/HEAD/fsync 故障，也没有测量真实耐久回调时延**。六类故障点名称表示模拟回调的失败阶段，不能宣传为已验证六个真实存储崩溃场景。

复跑命令（项目根）：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B -m unittest discover -s projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/feedback_diagnosis_v1/budget_optimization_cpu -p 'test_*.py' -v
```

## 后续接入条件

由主 agent 单独批准下一版本，不能热换活跃 V2。先在新版本把 sink 接到原耐久 Journal，检查 schema/回读与尾部错误规则，再按明确许可做同序列真实 fsync 对照与精确计时。保持相同动作/墙钟预算、候选顺序和全部物理质量门槛，分别报告工程吞吐与合格族产率。

当前唯一已经测得的变化是 **CPU fake-sink 回调次数减少**；真正导航/存储吞吐、合格族收益和科学贡献均未验证。
