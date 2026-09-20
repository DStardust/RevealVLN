# Q35N V15：下一步裁决入口

本包供审阅实际结果并决定下一步，不代表新增实验或新的正向结论。实验结果提交为 `62344f890d040032d5c9c8467133b1a9b376752b`；本次仅补齐可下载资产、网页可读摘要和推送通道。分支为 `codex/q35n-v15-legal-data-20260919`。

建议先读 [完整中文报告](../../research/continuation_memory_v1/legal_closed_loop_v15/REPORT_ZH.md)，再读下列最小证据链：

|审查问题|实际文件|
|---|---|
|当前状态、哪些事项未完成|[CURRENT_STATUS.json](../../CURRENT_STATUS.json)、[RESULT.json](../../research/continuation_memory_v1/legal_closed_loop_v15/RESULT.json)|
|完整分母、成功/失败/未知、动作成本|[ROLLOUTS.csv](ROLLOUTS.csv)、[配对审计与汇总](../../research/continuation_memory_v1/legal_closed_loop_v15/CONTINUATION_REVIEW.json)|
|三种监督、两档数据、三种子的全部训练结果|[TRAINING_REVIEW.json](../../research/continuation_memory_v1/legal_closed_loop_v15/TRAINING_REVIEW.json)、[TRAIN_PROTOCOL.json](../../research/continuation_memory_v1/legal_closed_loop_v15/TRAIN_PROTOCOL.json)|
|真实数据是否构成想验证的历史问题|[DATA_INFORMATION_AUDIT.json](../../research/continuation_memory_v1/legal_closed_loop_v15/DATA_INFORMATION_AUDIT.json)、[raw_run_008 审计](../../research/continuation_memory_v1/legal_closed_loop_v15/raw_run_008_AUDIT.json)|
|B2 识别/误报/保持问题|[EVENT_STATE_DIAGNOSIS.json](../../research/continuation_memory_v1/legal_closed_loop_v15/EVENT_STATE_DIAGNOSIS.json)|
|查询是否比精确状态提供更多信息|[QUERY_STATE_SUFFICIENCY.json](../../research/continuation_memory_v1/legal_closed_loop_v15/QUERY_STATE_SUFFICIENCY.json)、[objective.py](../../research/continuation_memory_v1/legal_closed_loop_v15/objective.py)|
|历史或视觉捷径|[INITIAL_OBSERVATION_CONFOUND_AUDIT.json](../../research/continuation_memory_v1/legal_closed_loop_v15/INITIAL_OBSERVATION_CONFOUND_AUDIT.json)、[BEST_VISIBLE_WITNESS_AUDIT.json](../../research/continuation_memory_v1/legal_closed_loop_v15/BEST_VISIBLE_WITNESS_AUDIT.json)|
|可运行代码、真实梯度与查询防火墙|[实现说明](../../research/continuation_memory_v1/legal_closed_loop_v15/README.md)、[CPU_TEST_RESULT.json](../../research/continuation_memory_v1/legal_closed_loop_v15/CPU_TEST_RESULT.json)|
|原始日志、失败尝试、执行时源码|[EVIDENCE_LOGS.tar.gz](../../research/continuation_memory_v1/legal_closed_loop_v15/EVIDENCE_LOGS.tar.gz)、[逐文件清单与 SHA](../../research/continuation_memory_v1/legal_closed_loop_v15/EVIDENCE_MANIFEST.json)|
|全部 18 个已训练轻量模型与初始化|[ARTIFACT_CATALOG.json](ARTIFACT_CATALOG.json)、[train_run_001](../../research/continuation_memory_v1/legal_closed_loop_v15/train_run_001)|
|固定因果特征缓存，支持 CPU 读出复核|[FEATURES.pt](../../research/continuation_memory_v1/legal_closed_loop_v15/features_run_001/FEATURES.pt)、[DATA.json](../../research/continuation_memory_v1/legal_closed_loop_v15/DATA.json)|
|不依赖 Codex 的后台执行|[standalone.py](../../research/continuation_memory_v1/legal_closed_loop_v15/standalone.py)、[真实退出后存活测试](../../research/continuation_memory_v1/legal_closed_loop_v15/STANDALONE_CPU_TEST_RESULT.json)|

本轮事实：18 个模型、10,800 次轻量更新、0 次基座更新、216 次完整自主续接；1 PASS / 128 FAIL / 87 UNKNOWN，唯一 PASS 来自 B1。144 项组内输入和原生动作前缀审计通过，logits 最大差 0；训练缓存与现场的跨会话比较另有 5 次原生动作翻转，不混为组内配对失效。GPU 会话累计约 3.2034 小时，当前已停止。

训练族的历史分叉可学习，错误记忆交换会破坏动作，匹配 sham 保持结果；但 B1/B2/Ours 均具此信号。所有臂和种子在新屋教师分叉均为 0/12，尚无交叉监督独有收益。这里不是新基座、R2R SR 改善、SR40、论文成立或机器人部署。

87 个 UNKNOWN 来自冻结检查器对碰撞轨迹的处理；它们没有被删除、转为负例或用于筛选模型。没有一个 Ours 对照比较具备全部 12 个已知标签，因此本包不编造完整配对成功率差。

数据只有 1 FIT 屋、1 CHECK 屋、各 3 个相关族。1→3 是同屋数据扩量。事件类别、房间组合、尺度与房屋变化相互混杂，不能据此证明“更多数据无效”或“必须换架构”。每个任务还存在对所有历史通用的已测 PASS 后缀，当前教师分叉主要刻画有限路线集合中的效率机会。精确状态足以重建全部交叉标签，B2 必须保持为强对照。

大型基座、best4k 增量权重、第三方场景、原始 RGB/语义数组和软件环境未上传。本包上传本轮轻量检查点、初始化、最终因果特征缓存、代码、证书和逐步日志；未上传项与哈希入口见 ARTIFACT_CATALOG.json。GitHub 可读不等于已在另一机器完整复现仿真。原始图像展示 HTML 是本地材料，不能假设网页审阅者已经看过像素。

需要给网页版 Pro 的任务见 [PRO_REVIEW_PROMPT_ZH.md](PRO_REVIEW_PROMPT_ZH.md)。请优先判断下一项最小判别实验，而非预设必须继续扩大 Ours。
