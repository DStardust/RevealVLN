# 成功纯时钟检查合并：独立 CPU 工程提案 V1

状态：CPU 验收完成，`executable=false`（真实生产未授权），不接入任何现有 runtime、冻结 cohort 或输入锁。

## 保持的约束

以 SHA 固定的 SingleCommitBudgetLedger 为父类，只合并成功且无结构变化的 `check_time` 持久化；每次依旧真实调用时钟并检查回拨、总 wall 和 phase wall，阈值仍为 `>=`。超限状态必须同步持久化成功后才抛 BudgetExceeded。每个动作 reserve 必须同步持久化成功才允许调用 backend；不异步写、不退款、不重置 created、phase started 或 actions。所有持久化 BaseException 均 poison，之后拒绝继续。

phase start/finish、同 phase 重试边界、动作预约均 flush，状态 schema 仍为 version 2。关闭已超时 phase 的原诊断行为保持，不由 close 授予额外动作。成功的 time-only `snapshot()` 是内存状态；`durable_snapshot()` 才是最后已确认持久状态。

## 明确不等价处与恢复边界

日志字节/条数和持久化时刻有意不同。成功操作返回值及每步内存 snapshot 与父类等价；不宣称每步磁盘 snapshot 相同。I/O 故障最迟于下个必须写入边界发现，不能声称与原实现发现时间相同；每个物理动作仍必须先取得 reserve 的 durable ACK。

进程崩溃可能丢失最后几次成功 time-only 水位，但不丢已提交动作、总起点和 phase 起点；同 boot 单调时钟下经过的墙时不能被重置。若崩溃后时钟回拨恰好落在旧持久水位和丢失内存水位之间，新状态不能恢复原实现的回拨检出能力（有显式反例测试）。因此禁止未经审核或跨 boot 恢复；未来接入须验证 boot 身份，或完全拒绝进程恢复。本类未伪造 boot 字段，未自动授权 resume。

## 下一版最小接入条件

主 agent 审核本提案；独立新 transport/runtime 文件与事前冻结输入、批次身份，所有原 wall/action/resource 阈值不变；同步 Journal callback，不改封存旧 ledger/auditor/cohort。当前锁定输入不允许原地替换此类。

下一版必须证明所有成功出口都有 phase close/最终 durable 状态一致性；所有超时与信号退出仍闭合或如实 resource-censored；若允许重启则加入同 boot 身份审计。再在一个完整真实族上跑原 18/27/54、预算、资源、cleanup 和内容审计，才能宣称 runtime 工程通过。CPU benchmark 不是此授权，也不是科学收益。

## 复跑

项目根目录下运行项目 Python 的 `-I -S -B -m unittest discover -s projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/witness_first_v1/budget_clock_batching_cpu_v1 -p 'test_*.py' -v`。所有临时目录只在本模块下。

`benchmark.py --output <本目录内不存在的新目录> --actions 48 --repeats 3 --post-checks 1` 生成真实同步 Journal 的 CPU 测试。`audit_compatibility.py` 只证明已封存 parser/schema 兼容，不把模拟动作文件当成数据族。
