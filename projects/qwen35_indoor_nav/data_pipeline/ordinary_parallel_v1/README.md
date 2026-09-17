# Ordinary parallel V1：下一批四分片 CPU 准备

状态为 executable=false。这里的任务清单不是已生成轨迹，更不是合格训练数据。当前 GPU5 ordinary_scale_v1/recovery_v4 生产者及全部旧封存只读；本版本未编写 GPU launcher、未运行仿真或操作 GPU。

数据沿用首批冻结的官方 R2R-CE v1-3 train 与 RxR-CE v0 train_guide 英语部分，使用同一物理路线 SHA256 定义与 51 个 FIT 房屋划分。排除 pilot 全部 100 候选以及现有批全部 1,000 候选，不因失败、隔离或中断而重新选入。剩余 R2R/RxR 各最多选 500 个不同物理路线，短缺如实登记。

选择采用稳定的按房屋轮询覆盖；四个分片进一步将完整房屋分组，按路线数量贪心均衡，以固定 SHA256 破同分。每条路线的所有原始英语指令一起保留到同一分片；跨来源相同物理路线不重复。分片只是并行生产归属，不改变 FIT/dev/confirm 划分，不产生新未见场景主张。

拟议的独立输出根为 production/shard_0000 至 shard_0003。每个后续 renderer 只能写自己的 routes/content/ledger/shards/quarantine/live 状态，不能共同追加 ordinary_scale_v1 或公共 ledger。GPU 绑定、租借恢复、资源预算、生成/回读质量门槛需另行准入；本次不授权这些运行。

单一合并器待实现：只在各分片终态明确且严格回读后，以只读方式校验内容 SHA256、路线互斥和全部指令归属，生成 merge/TRAINING_INDEX.jsonl。全局索引保留 shard_id、独立 reference_root、原 job_id/physical_route_hash/source，引用必须解析在各自 shard 根内。合并采用新临时文件完整写入再原子提升，禁止多个 worker 同写；失败/隔离/资源截断汇总单列，不当作合格数据。不得修改原 strict audit 阈值绕过当前 RxR 位移异常。

CPU 重现：在项目根使用标准库 Python 执行本目录 plan.py。已有产物只允许完全相同内容的幂等核验；不同内容要求新版本。INPUT_LOCK.json 绑定来源与旧冻结数据和代码，MANIFEST_HASHES.json 绑定本批输出。plan.py 不读取活跃 ledger 或 GPU 状态。
