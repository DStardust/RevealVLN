# GPU1 独立自动队列 V1

复用已经审核的 GPU2 auto_generation_v1 完整源码，通过唯一字符串替换、源码 SHA256 及反替换逐字相等验证进行独立适配。仅变更 GPU1 固定 UUID、batch_200…211、输出目录、顶层命令 `-I -B`，并加入旧源码闭包和 GPU2 全部 36 条冻结候选排除断言。原 guard、工厂、F/L/R、query160、27 重放、18 cells / 54 evaluations、强审核标准不变。

每批 3 个距离至少 1m 的 physical hub，最多 12 批 / 36 个不同程序。沿原固定四银行轮转，所有历史进入尝试均排除，不自动 retry；GPU2 已冻结未执行候选只算预约并排除，不算真实数据。候选不计合格族；同屋同 hub 多语义内容仍相关。

GPU1 无占位借还；原共享监督器支持 GPU1，所有外部/自身显存和利用率检查不变。每批 supervisor3900 秒 / factory3600 秒 / 60000 动作 / 7GiB 磁盘，job transport4200+audit1200≤5400，顶层总限12小时。源码及队列先冻结，CPU prepare 后主 agent 才可审批启动。

CLI: queue.py 然后 prepare.py 均只有 CPU 文件/来源工作。queue_v1/JOBS.json 交主调度；audit.py --batch PATH 输出 gate3 授权范围下 auto_generation_gpu1_v1/batch_NNN，不覆盖 GPU2 或旧 cohort。运输失败停本 lane，不自动重试；family 不通过如实记录。scientific_pass=false，不训练。

GPU2 的 batch_100…111 无论当前状态均仅按冻结 QUEUE/EXECUTION_CONFIG 预约排除，不读取或锁定其 journal/HEAD/PROGRESS/INPUT_LOCK 等运行文件，也不将临时 committed prefix 变成新源。旧已闭合批次继续原完整历史来源检查。
