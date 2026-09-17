# 自动特殊数据队列 V1

CPU 事前冻结最多 12 批、36 个尚未尝试程序；每批 3 个距离至少 1m 的物理 hub。按固定来源顺序轮转银行，每个银行按 house/hub 轮转其已封存排序。全部候选与排除原因保留。已有进入尝试和实际 action 分开记账；已冻结未运行配置只算预约，不充当物理尝试。语义去重忽略 A/B 交换、raw 别名、padding 和 seed，并保守合并 1m 内同屋同语义。

原 442 候选不是合格数据族。每个候选仍必须经过原 27 重放、18 cells / 54 evaluations 和完整强质量审核。同屋/同 hub 的不同语义程序相关，不构成独立泛化样本。

只适配已审核 batch08 所用原 frozen fresh clock transport：路径绑定新队列，INPUT_LOCK 工程文件数 1024→2048；动作、计数、事件、query、预算与所有 GPU 资源护栏不变。只允许 GPU2 固定 UUID，无占位借还、不自动重试、不恢复旧 run。每批原 3900 秒 supervisor / 3600 秒 factory / 60000 动作 / 7GiB 磁盘。顶层 job 5400 秒包含 transport_upper_seconds=4200（readiness、supervisor、清理）和 audit_seconds=1200；全队列 12 小时，不能保证全部 12 批都在最坏预算内结束，剩余保持未运行，不延长总限。

CLI：先 queue.py 冻结，再 prepare.py 仅 CPU 生成 batch_100…111。JOBS.json 供主 agent 调度；CPU 准备不是启动授权。顶层必须逐 GPU 串行，运输失败停 lane，不自动 retry；family 质量失败如实记账并允许按事前顺序继续。audit.py --batch PATH 只在 cleanup 后复用已冻结 gate3 observe_run，独立审核、不加入旧 cohort。

源代码、来源闭包、候选与队列均在运行前锁定，不删除 provenance 以凑 cap。只写新 auto_generation_v1 与批准的新 BE 批目录；旧锁、旧结果和全部失败不改。scientific_pass=false，不训练、不声称模型收益。
