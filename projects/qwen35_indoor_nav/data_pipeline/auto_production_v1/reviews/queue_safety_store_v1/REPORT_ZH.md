# 自动生产队列独立 CPU 安全复审

结论：`REVIEWED_CPU_SAFETY_CASES_PASS`。可以交主 agent 针对具体冻结计划作最终准入；不是通用恶意任务沙箱、物理实验验收或训练授权。

最终只读审核 `queue.py` SHA256：
`e606eab59bb654bb31583333cfa2367780f908be70356594f97c9eeb27422683`。

独立 `test_safety.py` 8 项 CPU 测试实际全部通过，耗时 0.373 秒。所有 child 均为 Python 假对象，无子进程/GPU 实验，临时文件及互斥锁仅在本审核目录内创建并清理。审核者未修改生产队列。

验证覆盖：Popen 后 child_started 日志注入 I/O 异常仍 drain child；子进程使用独立 session 并继承唯一 lease FD；输入核验期间收到 drain 不启动；核验消耗剩余预算后不启动；拒绝路径逃逸 job id；拒绝无界 audit；transport 与 audit 必须共同装入总预留；精确边界合法。主 agent 原有队列测试不计入这 8 项。

## 修改与证据链

最初静态审查发现日志异常可能在 child 未恢复时释放锁、继承信号组、drain 检查过早、id 路径/审核预算问题。主 agent 实现修复。本审核随后对 `7c61bef0b1b53907d3a679a7f28712573cdcb79c116d7991e7c2de8e3a113f61` 实测得到 7 通过/1 失败：输入哈希核验期间假时钟由 0 推进到 150，wall=200、job=100 时仍启动。最终版本在 verify 后及 Popen 前复核完整预算，该反例转为通过；未放宽测试。

## 使用边界

- `transport_upper_seconds` 是具体原运输实现的申明上界，不是本队列额外杀进程计时器。主 agent 必须审核实际 entry、冻结依赖、原监督/恢复预算与上界一致；本模块只检查总数值相容。
- 子 supervisor 继承 flock 可覆盖仅队列父进程死亡的情况；不声称任意整进程树 SIGKILL、内核故障或存储断电仍能自动恢复。单独绕过队列启动的程序也不自动服从本队列锁。
- I/O 失效可能导致最终 RESULT 写不出，不能据此补填成功；已有 lane/reservation 不自动重试。持久文件采用 fsync，但未测存储断电/目录项耐久。
- CPU audit 超时仍按失败停 lane；审核条目的语义质量由原独立质量报告决定，退出码不等于训练准入。
- 所有权、占位恢复及资源硬约束继续由冻结原 transport 负责；本审核未放宽任何旧约束。

复跑命令（项目根目录）：

```sh
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/data_pipeline/auto_production_v1/reviews/queue_safety_store_v1/test_safety.py
```
