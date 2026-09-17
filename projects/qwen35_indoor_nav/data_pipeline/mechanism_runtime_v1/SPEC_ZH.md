# Habitat 后端与 V4 导出接入节点

本节点落实用户“执行下一步”：接入并验证上一节点尚缺的生产接口；沿用用户与SFT并行分工。所有新内容仅写本目录，mechanism_factory_v2及旧数据完全只读。主agent独占根状态/GPU运行，子agent只实现与CPU测试。

阶段固定：

1. 实现独立HabitatBackend（继承既有固定版本传感器/动作，禁止旧engine全局补丁），V4导出和loader，预算/资源监控接入。CPU测试与旧日志差分。
2. 主agent审核代码、环境/场景真实路径及资产hash、GPU lease后，另登记精确执行配置，先验收旧单族9条实际轨迹（seed1109，最多600秒/12000动作/2GiB新增）。这只验证后端/导出接通，非新族、非独立收益；GPU只用空闲或已核实占位，不触碰SFT。
3. 仅在上述实测通过后主agent冻结P0运行准入，最多5候选bundle（每个FIT屋一个），沿用原protocol的64配置/720秒/20000动作发现、480秒/20000动作认证，全批6000秒/200000动作/16GiB磁盘/16GiB RAM/8GiB GPU上限。所有失败保留，首个完整preflight冻结后不替换，批后停止，不自动P1，不训练。若前置失败，收口工程缺口不假装造数完成。

任务主线/事件阈值256/同实例2帧/数值重建≤1e-5均不改。当前旧族仅interface_only；重导出仍interface_only不升级训练。下一生成族也仅candidate_fit_pool，未做反捷径准入前不标正式机制训练就绪。

Backend routes必须纯几何提案（不执行隐藏agent动作）。将navmesh路径转成离散动作只算提案，是否可行完全由TraceRunner真实执行验证。方法主张不放在路径提案算法上；若新离散规划低产，应归因工程不归因记忆监督。实际step预算、动作确认返回、碰撞、所有观测和raw/canonical数值边界留证。

导出必须从完整真实trace和Compiler V4重新求值，policy/监督/query分离；完整task-conditioned因果prefix、成功续接动作owner、强M2、group/split与像素内容hash审计。旧日志导出验收无GPU并明确reexport，不计新物理族。V4 loader独立版本，不写旧loader或SFT。

运行时资源watchdog仅终止本节点已登记子进程组，外部SFT/未知进程不可操作。所有GPU lease成功/失败/中断后复原。不能用CPU测试宣称Habitat实际运行通过。
