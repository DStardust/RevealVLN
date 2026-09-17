# 数据使用与隔离

当前只验收一个V3 interface_only族，不自动允许将其投入完整训练或宣称独立泛化。普通SFT会话继续只用ordinary pilot；不要混入本族改变其已冻结action-only协议。

1. FAMILY_MANIFEST给族/任务/历史/续接分组。SUPERVISION_ONLY每行是一个完整cross cell，包含18个实际求值；6个fail也是可靠BCE负例，但它们的动作CE mask全0。
2. POLICY_INPUT是每cell在共同决策边界的短窗口。**不能只读这一行就声称使用了历史。** POLICY_PREFIX_INDEX列出6条task×history完整因果prefix流，从memory=0依次处理所有时刻，同一prefix可接3个训练期query。换task须重算memory。
3. content/<pixel_sha>.rgb.npy为224×224×3 uint8 RGB，semantic.npy仅供离线审核。原始像素hash不是npy文件hash；SHA256SUMS另覆盖文件字节。无需人工标注；不要把semantic或角色真值给策略。
4. physical_traces含27次完整实际重放、逐步pose/像素计数/hash及explicit normalization_events。full_log_ref使用旧core整数pixel-key哈希codec，SOURCE_AND_CONFIG已声明；seal_readback.py演示正确复核。新compiler query/hash使用JSON字符串key规范化，能稳定序列化往返。
5. query仅在因果prefix memory计算完后进入独立训练reader。只能使用query_schema_version/coordinate_frame/sequence；action_trace_ref/rgb_content_refs/样本和家屋ID不是特征。固定类别整数词表不随TV/chair角色交换。
6. ACTION_DEDUP给unique CE owner与完整lineage；动作首个target step=cutoff即a_t，不是未来观测。对未来任一决策先按实际观测扩展自己的因果context，不能把整个续接作为当前策略输入。
7. 三个seed只检查确定性，不能拆成train/dev/test。此族来自已暴露17DRP5sb8fy；其变体/续接/措辞一律同组。后续正式机制训练必须独立冻结多族、场景/模板划分与匹配M1/M2/M4试验，不能从这18格准确率声称导航收益。
8. 必须披露：任务实例为TV/sink V3，非原餐椅任务；真实历史到达后有一次<=1e-5m/rad数值共同状态重建。原始/重建证据双留，事件未变；不可描述为未干预的原始精确像素汇合。
