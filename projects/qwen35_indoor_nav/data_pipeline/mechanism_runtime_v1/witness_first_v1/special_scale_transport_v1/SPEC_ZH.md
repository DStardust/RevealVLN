# 特殊数据扩量运输 V1：GPU1/2 无占位分支

本目录首次封存仅 GPU1/2 nonexclusive。后续借还使用独立 sibling，不向本目录新增代码或修改本版本。

沿用封存 BalancedFactory、原 PartialTraceRunner、同步 reserve-before-action 的 fresh-only clock-batching、完整 18 cells/27 certification replays/54 evaluations、原查询长度/碰撞/标签及 FIT 约束。每批仍为三个事前固定候选，3900 秒 supervisor、3600 秒 factory、60000 action、7 GiB；未知失败不替换、不重试。不使用训练或 GPU0。

唯一物理运行协议修订是活跃 worker 采样。原 XML 在 parse 前同步 fsync；原 parser/UUID 检查保留。仅 worker 在进程清单中且原 guard 的唯一错误为 MEMORY_ACCOUNTING 时，使用 max(device,sum(processes))，同时更严格要求 gross<4096 MiB、外部每进程<=768/合计<=2048、own upper=gross-external<4096。原 idle/final guard 和 finally cleanup 不变。派生 PENDING/ACCEPTED 与原数据分开；不称不一致样本通过旧 guard。日志失败立即中止；无异步写入。

新版审核逐条验证 XML hash→原 parse→重算 decision→RESOURCE 索引/原值/上界/时间；遗漏、改值、假通过、重复及未闭合记录拒绝。最终显存仍按原判别，所有原科学内容核验保留。合格族标独立等级 QUALITY_VERIFIED_FIT_AMENDED_ACTIVE_ACCOUNTING_NOT_MODEL_GAIN，不替换旧 cohort 或宣称模型收益。

配置/队列/授权/完整源码在任何 GPU 动作前冻结。main 须另写每批 MAIN_AGENT_SCALE_APPROVAL.json 精确审批。预先冻结 12 小时队列，单批总预留 5400 秒（transport 4200 + audit 1200），不足预留不启动新批。源选择由独立 source 模块负责；迁移预约必须旧 run 始终恰好只有原 CONFIG/INPUT_LOCK，完整候选结构相同。旧 batch202/103 的失败与 partial 保留，本版本不重放。

这是工程运输修订，不是导航效能或统计泛化结果。原 batch202/103 失败触发 XML 未保存；只能确认原错误 MEMORY_ACCOUNTING，不能用先前成功采样猜测触发数值。
