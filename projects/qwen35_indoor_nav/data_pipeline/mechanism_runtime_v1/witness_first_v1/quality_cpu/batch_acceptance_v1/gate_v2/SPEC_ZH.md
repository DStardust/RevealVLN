# 初步生产门槛 CPU V2（不授权生成或训练）

旧 SPEC 的 required_predeclared_batches=2 按主 agent 确认解释为至少 2 批；旧代码 exact2 保留，不修改旧冻结文件。新纯评价函数允许 2、3 或更多批，其他要求不变：每批至少 3 个独立尝试 hub 且至少 2 个常规合格 hub，总至少 6 个常规合格 hub、至少 3 个 FIT 屋；同屋距离小于 1m 连通合并。恢复数据不能补常规批次资格。

全历史暴露账必须包含给定 batch_execution_v1 下所有 batch_*/run_v1/EXECUTION_CONFIG.json，含未启动、资源中断、活跃和语义拒绝。不得只传成功批。

合法的新工程版本允许使用**事前冻结评价 cohort**，不让历史失败永久否决未来可证伪的稳定判别：

- manifest schema_version=q35n.production_cohort.v2，记录 cohort_id、mechanism_version、transport_version、criteria_version=AT_LEAST_TWO_BATCHES_V2、全部 declared_run_roots。
- 至少 2 个不同完整批路径；每个成员 INPUT_LOCK 必须事前包含同一 manifest 的准确 SHA256，manifest 与 INPUT_LOCK 本地时间都早于成员 PROCESS.started_unix。
- 不从 cohort 中删除失败成员，不临时增补成功成员。全历史其他批仍在曝光账，但不参与这个新 cohort 的成功判定。
- 当前没有这样的新事前 cohort，不会把已经看到结果的 batch02r2 追认为注册成员。当前调用不提供 manifest 时只能输出历史现状，不能获得 cohort 稳定 PASS。

恢复文件和常规完整运行数据库存单独报告，包含数据来源等级、hub 去重与 FIT 屋数；它们不是导航收益、训练授权、统计稳定或模型泛化证明。GPU 借卡/占位恢复应另核验 lease 终态，不能用 renderer 消失替代。
