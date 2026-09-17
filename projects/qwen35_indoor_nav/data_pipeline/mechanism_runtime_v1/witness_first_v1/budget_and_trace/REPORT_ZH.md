# witness-first：预算日志与中断轨迹 CPU 接入验收

结论：**READY_FOR_MAIN_AGENT_REVIEW**。15 项 CPU 测试全部通过（4.624 秒）。真实文件 Journal 已接入、少量实际 fsync 已执行以检查耐久调用顺序；这不是 fsync 吞吐 benchmark。未使用 GPU、仿真或模型，未修改任何原实现或活跃 worker，也未自动启动 witness-first 数据生成。

## 接口

[adapters.py](adapters.py) 提供：

```python
from adapters import Journal, durable_budget, PartialTraceRunner

with Journal(new_run_directory / "journal", frozen_config) as journal:
    ledger = durable_budget(journal, frozen_limits, clock=time.monotonic)
    ledger.start_bundle(candidate_id, "discovery")
    runner = PartialTraceRunner(backend, compiler, ledger,
                               lambda kind, payload: journal.append(kind, payload))
    # BudgetExceeded 必须继续传播给原阶段停止处理，不得改成合格轨迹。
    trace = runner.run(position, yaw_bin, actual_planned_actions, seed=1109)
```

以上仅接口示意，不是运行授权。调用方须将本子目录置于模块导入路径，并使用**本适配器导出的 Journal**（精确来源已验证），不能传另一未验证同名类、异步 sink 或任意回调冒充真实 Journal。原 planning、core_bridge、runtime_journal 和 SingleCommitBudgetLedger 的文件均经其已有 SHA256 清单核对后只读导入。

`durable_budget` 把 SingleCommitBudgetLedger 的 persist 绑定为原 Journal 的同步 append。每次合法 reservation 实际新增一条 budget 记录，测试捕获到三次真实 fsync 调用；确认 HEAD 已指向对应事件后才允许 backend 调用。其“失败瞬间可能更保守预扣一个动作、poisoned、不退款”的边界与前一 CPU 交付一致。

## PartialTraceRunner 的精确范围

正常路径直接运行原 TraceRunner，仅以实例局部代理捕获实际 backend 返回，不 monkey patch 原类/全局 backend。正常输出、trace_hash、事件、计数在含 STOP 与数值 join 的 CPU 对照中逐值相等。

BudgetExceeded 时：

- 只记录已经返回合法 bool collision 状态的动作；尚未调用或返回状态未知的动作不当作 confirmed。
- observations 仅包含真正通过原 observe 检查的返回，不猜缺失下一帧。
- backend.observe 已返回、但随后预算检查中断的原始返回，单独保存于 `raw_observation_returns` 且 `validated=false`，不混作合格观测。
- 如数值 join 重复观察同一步，诊断记录按真实调用顺序保留，可能重复 step；不声称是可直接训练的标准完整轨迹。
- partial 固定 `complete=false`、`training_admission=false`、`diagnostic_only=true`，保留预算快照、当次预扣数、confirmed 数、重建证据未完备声明及错误。
- 先通过调用方 emit 保存 partial，再重抛**同一 BudgetExceeded**。`partial_emit_acknowledged` 仅在 emit 返回后为 true；如果 sink 出错，错误继续传播，原 BudgetExceeded 保留为异常上下文，不声称 partial 已持久化。
- 通用 backend OSError 不被伪装成预算失败或完整轨迹，动作预扣不退还。本适配器只对 BudgetExceeded 增加中断轨迹，不宣传所有异常都有完整回放记录。

`last_partial` 是内存诊断副本，不是耐久落盘证明。partial 没有生成 see2 任务标签或未来证据；不能进入完整 family/训练认证。

## 本次真实文件测试

[test_adapters.py](test_adapters.py) 共 15 项：

1. 原 Journal 的真实新建、append、HEAD、关闭、验证恢复与继续预算。
2. backend 调用时直接回读事件与 HEAD，确认预扣记录已经确认。
3. event fsync 后、HEAD 前故障：保留追加尾部、ledger poisoned、backend 零调用、恢复拒绝。
4. 三个 fsync 阶段分别注入 OSError：不调用 backend，不退预算，原对象不能继续。
5. 预算拒绝也保留耐久计数与时钟，不调用 backend。
6. 正常动作、STOP、空轨迹与数值 join 和原 runner 输出/事件精确对照。
7. 动作后预算截断不补下一帧；观察后预算截断把未验证 raw 单列；动作预算截断仅保留已执行前缀。
8. reset 前超时不声称初始状态已确认；partial sink 故障不继续执行；backend 异常/状态未知不退款或冒充 confirmed；多 trace 不串捕获缓存。

重要限制：第三个 fsync（目录同步）失败发生在 rename 之后，当前仍运行的文件系统可能已经显示一致的 event/HEAD，测试允许其被只读校验读出；**这不证明真实掉电后耐久性**。当前 ledger 仍 poisoned、未执行动作、保守预扣不退款。不能把“恢复能够解析”写成“已模拟并通过物理掉电恢复”。

第一轮 13 项测试有一个测试 fixture 错误：join 后直接 STOP，缺失原规范要求的下一边界运动/观察，原 runner 自己就拒绝。修正 CPU fixture 为 join 后合法运动再 STOP；未修改原质量门槛。最终扩充为 15 项全部通过。

所有测试临时文件仅位于本目录新建 TemporaryDirectory，测试结束已清理。未重读或改写旧特殊族来冒充新物理结果。

## 后续

主 agent 核验本交付后可在独立新版本接入。仍需真实同候选预算下验证 walltime、合格族产率和严格导出；本节点没有新增物理族、没有实测加速、没有科学收益 PASS。
