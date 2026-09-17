# V5 worker 遥测附加审核

只读，不在任何生产 INPUT_LOCK 中，不改 V5、wrapper、旧 strong family auditor 或运行文件。CPU 合成接口/反例测试不是对实际运行的验收。

审核原始 XML 的 UTF-8 SHA，重新调用原 parser 和 check_gpu。不能直接把 JSON parsed_sample 内字符串 PID 键当作原解析结果输入 guard。逐事件核对 parsed 值、guard 结果、first_error、attempt、原 retry_deadline 和 gross 条件；unsafe 旧样本不能被后来低读数洗掉。最多原查询 + 两次补采，重采 deadline 必须固定首次错误时钟+1 秒，timeout 不可大于剩余量；不得隐去 failed event、重排资源样本或接受 guard_pass 后的 sampling_stopped。

每个最终 guard_pass 与 RESOURCE_SAMPLES 的 UUID/memory/utilization/processes/own_upper 顺序精确绑定，原 wall<3900、disk<7 GiB 保持。事件 monotonic、query_started 和各 resource.elapsed 必须能兼容同一个原始 started 区间。日志读取限制2 GiB/文件、10000事件、780采样（原循环每5秒、<3900秒）。完整闭合入口先验原 source/config/hash/journal/resource/store/lease/cleanup 和 V5 版本/前次失败绑定，再给遥测附加结论。

证据边界：wrapper 写的是持久化前的事件时间，不含每次 fsync 返回时刻，也未单独写 supervisor started。日志可直接重算记录时间和1秒窗口；最后 I/O 后仍未过 deadline 的证据来自已锁 wrapper 返回前检查 + 后续真实 RESOURCE 对应关系。报告明确 `durable_ack_timestamps_directly_recorded=false`，不虚构独立测得的 ACK latency 或精确绝对 started。

本模块不单独验证 lease/restore 采样策略，也不取代 family 18/27/54、M2、因果/身份/内容审核。V5实际闭合后，只有原强族审核与本附加审核同时通过，才能接收该新运输版本的完整数据证据。仍不是模型收益、统计泛化或投稿创新结论。

主入口：`audit.py --run <BE>/batch_06r2/run_v1 --output <此目录内新输出目录>`。实际运行未闭合或资源失败时拒绝，不把缺少日志判成零错误。GPU 操作为0。
