# Recovery V2：原子文件计量恢复

原两次失败只读保留。第二次为 `du -sb` 返回 1；原 supervisor 未保存 stderr，不能确证具体消失文件。当前静态 du 成功，编译器确实以 `<sha256>.png.tmp` 原子提升，故监控与提升竞态是与现象一致的解释，不伪称已捕获原始竞态。

本版本采用不跟随符号链接的 apparent-byte 扫描，只允许精确登记临时路径的 FileNotFoundError；普通文件消失、权限、IO、目录消失和符号链接均失败停止。每个消失临时文件保守加 1 MiB，199 GiB 停止阈值给 200 GiB 总预算留 1 GiB 裕量。运行时新增量无法完全原子测量，仍每 60 秒检查，不能称精确文件系统快照。

固定原 1,000 jobs，恢复前 60 条唯一 ledger 终态，无 unfinished route 目录。原首 shard 及 10 条 quarantine、旧 live progress、代码、两轮 RESULT 均锁定。新 worker 只追加未尝试路线和 ledger/shards；新 quarantine/live 汇总写此目录。原逐路线严格回读阈值及基础编译器未改。

4 小时总墙钟预算扣除原两轮实际耗时 138.21933245658875 + 66.73443341255188 秒。GPU2 仅使用空闲资源，外部进程不发送信号、不释放占位。子任务未运行生产或 GPU；主 agent 审查后才启动 run.py。代码与输入锁由 INPUT_LOCK.json 固定，ledger 仅启动前验证完整前缀；不可用同目录重启覆盖本次 run。
