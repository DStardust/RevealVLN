# 内容存储与资源检查

只写本runtime的新目录，不复用或改写旧factory。模块不发送信号、不查询或操作GPU、不启动仿真，CPU测试也不进行这些操作。

`ContentStore(root, max_bytes)`要求root位于本runtime之下且尚不存在，父目录由调用方预先创建。单写者，不支持恢复。`put_array(array, 'rgb'|'semantic')`及`put_raw(raw_bytes, kind, shape, dtype)`返回pixel_sha256、NPY文件sha256、绝对path、相对relative_path、shape、dtype、nbytes与file_bytes。RGB严格uint8 HWC三通道，semantic严格小端uint32 HW。数组按C顺序存储；hash基于原始像素，不是NPY文件。

NPY采用标准v1头，无需numpy依赖。相同像素/类型去重，既存文件必须字节一致，包括shape/dtype；坏文件或冲突不覆盖。通过新partial文件、fsync、不可覆盖的原子hardlink、目录fsync提交。失败保留partial（若已经写出）及FAILURES.jsonl；不删除或修复证据。写入前预算检查为内容文件保留64KiB错误日志预算，partial也计入内容额度。遇文件系统耗尽时日志可能无法持久化，原错误仍抛给supervisor；不得据此声称所有系统故障都能落盘。

`ResourceGuard(wall, ram, disk).check(samples)`是纯检查器，单位秒/字节/字节，samples键为wall_seconds、ram_bytes、disk_bytes。超限抛BudgetError，带resource_censored=True与as_dict()；恰好到限通过。它不是自主watchdog：主agent必须提供真实子进程树的RAM/墙钟/整个run目录磁盘采样，并在动作前/定时检查，进程组终止与GPU租约恢复由主agent的独立注册supervisor负责。不能仅统计ContentStore代替整个runtime磁盘，也不能仅检查主进程RAM。

旧factory_v2 journal有固定目录约束，不可猴补或修改原文件。本runtime如需journal，应独立版本派生、重新测试、记录来源及唯一允许根。本模块没有提供任意PID kill helper，避免误把外部SFT视作本节点子进程。

以上只证明CPU存储/资源接口；实际Habitat、硬性异步watchdog与真实GPU恢复仍由主agent运行验收，不能由单测推出运行或科学PASS。
