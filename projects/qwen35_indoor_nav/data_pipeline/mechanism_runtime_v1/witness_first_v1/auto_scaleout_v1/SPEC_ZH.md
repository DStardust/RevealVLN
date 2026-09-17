# 特殊数据四卡扩量来源 V1

本节点仅 CPU 冻结候选与全部来源，不操作 GPU 或占位、不写主授权。原真实银行 442 个程序，已观察的全部历史配置及两条 36 候选队列一律排除（无论成功、失败、partial 或尚未执行）。只读已冻结 EXECUTION_CONFIG/QUEUE，不读取活跃日志、HEAD、PROGRESS 或临时 journal prefix。失败保留，不换 ID 重试。

语义排重沿已审核规则：同屋 1m 内物理位置，A/B 类别与房间无序组及 T 类别房间相同即重复；raw 别名、A/B 交换、padding、seed 不是新程序。每批恰好 3 个距离至少 1m 的 hub。同屋和同 hub 多程序相关，不增加独立 hub 数。

候选按剩余 hub 组容量由大到小分批，平局按房屋和完整 3D 位置，再沿原冻结银行顺序。可以跨银行合批，保留被使用各银行的全部 SOURCE_LOCK，不删除 provenance。此排序只依赖事前真实候选供给，不看导航/认证结果。batch_300 起连续编号，按 GPU3/4/5/7 固定轮转，队列间互斥；至多128批，每lane最多12小时，未运行尾项保留，不自动转卡或重试。

SOURCE_LOCK 须≤1900、合计≤30GiB，为原运行2048文件/审核32GiB上限留余量；不修改原任何质量或资源阈值。每批保持原3候选、3900秒supervisor、3600秒factory、60000动作、7GiB磁盘，18cells/54evaluations/27真实重放与强审核不变。

接口：queue_v1/QUEUE.json 的每项含 id/gpu/prepared_snapshot/candidate_ids/source_snapshot/source_indices，以及 source_items 原始银行与行号。输运由独立 special_scale_transport_v1 在主 agent 审核和冻结 holder/authorization 后准备。每卡借还必须以精确 holder 身份或上一批真实 RESTORATION 成功链绑定，禁止复用过时 PID。

原候选容量不是合格族，也不是训练目标已达标。本节点不采新 hub，不声称泛化或模型收益；scientific_pass=false。后续万族规模仍需更多真实 hub/场景供给。

单独 continuation_queue_v1 依主 agent 新授权迁移从未启动的旧203…211→220…228、旧104…111→240…247，共51候选，保留原完整候选与词面。旧run须只有CONFIG/INPUT_LOCK，额外绑定旧202/103已cleanup终态；绝不复接这两次partial。运行时必须再次检查 previous_run_root 未启动，且旧lane不会恢复原预约。该明确迁移例外不改变四卡新池的全部历史配置排除规则。
