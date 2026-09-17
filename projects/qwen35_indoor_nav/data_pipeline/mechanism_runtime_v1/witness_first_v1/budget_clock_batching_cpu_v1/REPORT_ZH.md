# CPU 工程结论：有明确减少同步 I/O 的收益，未授权切换生产

`ClockBatchingBudgetLedger` 的 25 项 CPU 测试通过，最终复跑 0.442 秒。包含真实 Journal 三阶段 fsync ACK、三处 fsync 故障注入、BaseException poison、动作不退款、精确超时边界、恢复原起点，以及崩溃丢失时钟水位的已知非等价反例。未使用 GPU，未修改活跃或封存源码。

## 真实旧轨迹说明了什么

[ACTUAL_JOURNAL_FACTS.json](ACTUAL_JOURNAL_FACTS.json) 对已关闭 batch_02r2 原 journal 做完整 canonical/hash chain/HEAD 校验。certification 中 15,318 次已确认动作对应 46,332 条成功纯时钟记录和 15,318 条预约记录，约 **3.025 条纯时钟 + 1 条预约/动作**；不能误称 4 条都是纯时钟记录。没有逐 fsync 计时，不能从条数推导真实导航 wall-time 占比。

## 实测 CPU 同步文件循环

最终代码对应 [benchmark_v3_final/REPORT.json](benchmark_v3_final/REPORT.json)，3 对交替运行，每次 48 个 mock 动作，动作前两次/后一次成功时钟检查；使用原 Journal 和实际 os.fsync，动作事件本身也同步写入。确定性时钟比较状态，perf_counter 独立测墙时。

| 每次 trial | 原 SingleCommit | 新 ClockBatching |
| --- | ---: | ---: |
| budget 记录 | 196 | 51 |
| 全部 journal 记录 | 245 | 100 |
| 已返回成功的真实 fsync | 736 | 301 |
| events 字节 | 115,713 | 38,296 |
| 3 次墙时中位数 | 2.7502 秒 | 1.2768 秒 |

budget 记录减少约 74%；每次 reserve 在 mock backend 前仍完成原 event、HEAD、目录 3 次真实 fsync。配对墙时比值中位数 2.244（不同于两个总体中位数之比）。所有成功最终 snapshot 精确相同。共享服务器/NAS 未隔离，小样本只能支持**这段 CPU 日志循环**省 I/O，不能写成物理数据生产加速倍数。

先前 `benchmark_v1`（每动作 4 次纯时钟）和 `benchmark_v2_observed_pattern` 均保留；前者是更密集工作负载，不能替代实际约 3 次的主模式。最终代码额外包含 callback 被移除时立即 poison 的失败路径检查，故以 v3 为最终代码测试。

## 审核兼容与接入裁决

[AUDIT_COMPATIBILITY.json](AUDIT_COMPATIBILITY.json)：6 份实际 benchmark Journal 均通过原 sealed acceptance parser 的 chain/HEAD/config 校验，741 个 budget snapshot 通过原 BudgetLedger schema 校验。原 phase/恢复审计按预约、确认动作、phase boundary 提取证据，并非硬要求成功 time-only 的固定条数；这是静态接口观察，不是完整族验收。

仍须遵守 [SPEC_ZH.md](SPEC_ZH.md) 的恢复限制：崩溃后丢失的纯时钟水位并不与旧持久化行为完全等价。当前代码不绑定 boot 身份，未经审核的跨 boot 恢复不授权；新 runtime 可选择完全禁止 resume。

建议下一批之后，单独冻结一个工程升级节点接入；不得修改已在运行的 cohort 输入。完整真实族与资源/退出审计尚未跑，`runtime_pass=false`、`scientific_pass=false`、`training_admission=false`。科学标签、数据质量阈值和批次门槛没有变化。
