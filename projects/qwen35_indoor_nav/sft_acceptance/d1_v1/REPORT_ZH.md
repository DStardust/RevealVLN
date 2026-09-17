# D0 + D1 双卡小集验收

结论：`BLOCKED_ENGINEERING_GATE_NOT_SCIENTIFIC_FAILURE`。DDP接口通过：False；10训练路线可学性通过：None；scientific_pass=false。

固定10训练路线、565决策，只用既有训练split；不读dev挑配置、不运行闭环。旧SFT来源593项哈希通过，原失败不覆盖。

## 实际执行

每rank完成训练更新：[0, 0]；实际训练决策：0；动作暴露：{}。单卡参考与DDP诊断更新另外见REAL_EQUIVALENCE，丢弃后才开始D1。工程失败时未执行指标保持null，不记为科学FAIL。

双卡执行记录：`{"error": "AssertionError()", "returncodes": [1, 1], "wall_seconds": 118.04664421081543, "GPU_count": 2, "conservative_GPU_seconds": 236.09328985214233, "real_tasks_stopped": 0, "all_leased_holders_restored": true}`。所有借用占位恢复：True；真实任务停止0。

## 小集评价

频率常数CE=1.005154；始终前进accuracy=0.600000。

before：`null`

after：`null`

阈值在运行前冻结：accuracy>=95%、macro recall>=90%、STOP precision及recall>=90%。200步内未通过不证明无限预算下不可学；通过也只证明小训练集拟合，不能当泛化/导航收益。

## 诊断与边界

D0_SUMMARY.json保留旧terminal的正常记忆尺度、指令/图像/零记忆敏感性和梯度。置换产生分布外输入，不足以证明正确语义利用；梯度大也不独立证明实现错误。

本节点不改原架构、学习率或类别权重。DDP、类别均衡及后续稳定化不当作原创贡献。不进入全量D2或机制训练；下一步：Review numerical/input/optimization evidence; no automatic D2。

报告与result记录实际测量，rank日志保留全部失败。缓存/tmp不封存（占位可继续使用其缓存），根目录证据和checkpoint单独SHA256SUMS封存。
