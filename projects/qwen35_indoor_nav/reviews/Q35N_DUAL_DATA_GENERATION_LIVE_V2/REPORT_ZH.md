# 双数据真实生产进展 V2

2026-09-09 20:22（Asia/Shanghai）运行中快照，不是终态报告。动态进度和终态以所链文件为准。

## 实际启动

- 普通数据：GPU5，tmux `q35n_ordinary_scale_v4`，恢复同一冻结1,000路线批次。启动时68条已记终态，剩余932条不重做旧路线；实测推进超过82条。原始来源仍为官方R2R-CE train及英语RxR-CE train，场景为合法取得的MP3D，限冻结FIT。
- 特殊机制：GPU1，tmux `q35n_mechanism_feedback_v1`，固定3个房屋候选的实际状态反馈构造。快照52条完整探测轨迹、5,866动作、0碰撞；新增完整物理认证族0、训练准入族0，不能按探测数量冒充合格机制量。
- 普通已处理102/1,000路线，其中前两片严格导出验收83不同路线、249指令、16,923指令条件决策（首片39/117/7,188；次片44/132/9,735）。次片`integrity_pass=true`，另4条严格隔离；未发布的物理回放成功条数仍需独立导出审核。历史pilot99条单列，不混作本批增量。

## 工程修复及诚信

原转向位移异常仍按严格阈值隔离，不放宽阈值。后续计量失败、外部GPU上下文中断和未完成路线均留账。新恢复版只处理未尝试任务，不覆盖旧失败、旧源文件或旧结果。

GPU2出现实际外部任务后，迁移普通生成至核实占位的GPU5。主agent独立复跑V4全部14项CPU测试及输入锁预检通过；唯一AST改动是物理渲染设备2→5。监控覆盖图形和计算进程，并以总显存扣除外部进程作为自身用量保守上界。Supervisor在finally清理自身并恢复精确核实的占位；当前借用尚未结束，不能提前称已恢复。

机制旧supervisor仅计算进程计量会漏图形renderer；当前实际用量较低，独立全进程补充guard正在准备，不改活跃输入锁源码。该监控补充不改变数据质量阈值或延长原预算。

## 用户追加的多卡扩展

已记录“稳定之后几张卡都可用于生成”。普通路线可按确定性、互不重叠的任务分片增加生产者，每卡独立目录/日志/临时文件，唯一审核合并器发布索引。下一批四分片清单先CPU准备，不能四个进程同时写现有BASE。

先观察新分片持续通过双回放、严格导出、资源监控，再扩卡；普通与机制准入分开。特殊构造若仍未产生完整族，不能靠增加GPU把不合格探测包装成量产。NAS落盘可能限速，显存和GPU利用率低不代表应无限加卡；扩展按合格路线/小时和落盘吞吐评估。

主线V3不改；没有训练、本轮导航收益或科学PASS。

### 20:24 补充

机制前两个候选已分别以`RESOURCE_CENSORED`（17DRP5sb8fy，600秒发现预算）和`REJECTED / CONFIGURATIONS_EXHAUSTED`（1LXtFkjw3qL）关闭；第三个继续。不据此声称已获得新族，也不自动放大该特殊管线。

下一批[四分片清单](../../data_pipeline/ordinary_parallel_v1/PLAN.json)已实际准备：1,000不同物理路线、3,000原始英语指令、48 FIT屋，R2R/RxR各500；分片250/244/253/253条，所有路线及指令整体归属单片，排除pilot全部100和当前批全部1,000候选。主agent复跑6项CPU测试通过；该节点仅CPU清单，GPU runner及合并器未实现，`executable=false`，新增生成量0。用户多卡授权已记录，无需重复询问是否可以用卡，但实际扩展前仍须完成隔离runner、资源恢复和严格合并验证。

## 权威入口

### 20:26 最后主agent快照

普通GPU5已处理168/1,000路线，前三片严格通过121路线、363指令、25,899指令条件决策；恢复后连续运行约345秒。特殊115条探测/8,731实际动作，仍0新合格族。生成在独立tmux中继续；终态文件出现后须优先读取终态和GPU恢复结果，不把本快照无限沿用为运行中证明。

机制全进程补充guard的11项CPU测试通过，但真实启动在发信号/建立监控前因项目Python缺少pidfd接口退出；见supplemental_guard_v1/START_RESULT.json。不能称补充guard运行通过。原有界supervisor继续、预算不变；本节点低GPU总显存经只读查询确认，但原机制监控图形进程计量缺口仍须在下次runtime修复，特殊线暂不扩卡。普通V4已实际包含图形进程监控，不受此缺口影响。

- [授权](../../authorizations/DATA_GPU_TRANSPORT_AND_SCALE_V2.json)
- [普通实时进度](../../data_pipeline/ordinary_scale_v1/recovery_v4/PROGRESS.json)
- [普通严格分片目录](../../data_pipeline/ordinary_scale_v1/shards/)
- [普通当前运行记录](../../data_pipeline/ordinary_scale_v1/recovery_v4/run/)
- [特殊实时进度](../../data_pipeline/mechanism_runtime_v1/feedback_generation_v1/run_v1/PROGRESS.json)
- [特殊运行目录](../../data_pipeline/mechanism_runtime_v1/feedback_generation_v1/run_v1/)
