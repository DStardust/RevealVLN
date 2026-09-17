# Recovery V3：非独占渲染与中断终态

V2 因新出现的 500 MiB 外部 context 停止，原结果保留。新增外部 PID 不再自动构成停止条件；本批是数据渲染，不是独占 GPU 吞吐实验。初始利用率需为 0；外部每 PID 至多 768 MiB、总和至多 2048 MiB，自身小于 4096 MiB，超限仅停止自己的子进程。每个 GPU 快照在资源断言之前写入，绝不发送信号给外部进程。

唯一中断路线 r2r_383f0490ffb5248c87ed，仅留 112 字节 replay_certificate.json。主 agent 先审查执行 prepare_interruption.py：将该精确目录移动至本版本 interrupted/，保留路径与 SHA256 清单；原路径新建仅 result.json 的 RESOURCE_INTERRUPTED 占位结果，append 账本终态。该路线不重试、不进入训练、不记为科学失败；原严格 audit 会将它作为非 CERTIFIED 的空监督终态排除。此脚本失败不自动恢复或二次覆盖，必须重新审查。

旧 67 条终态与代码/失败/首 shard/10 隔离锁定；准备后 68 终态，剩余 932 个固定 job。总 4 小时预算扣前三轮真实墙钟 226.318073520204 秒。仍采用 199 GiB 保守停止、200 GiB 总磁盘预算；恢复计量仅识别本版本精确原子临时文件名，所有真正 IO/权限错误停止。

子任务只完成 CPU 文件准备与测试，没有移动原路线、追加账本、启动生产或操作 GPU。主 agent 执行前须复核 INPUT_LOCK.json 与实际空闲条件。
