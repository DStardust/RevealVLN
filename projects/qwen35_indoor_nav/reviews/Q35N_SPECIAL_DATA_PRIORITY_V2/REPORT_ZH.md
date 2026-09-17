# 特殊数据优先 V2：主agent执行记录

2026-09-09，用户要求优先特殊数据，普通数据持续后台生成。主线仍为V3交叉续接监督执行记忆；不改算法创新主张，不重开文献检索，不训练。

## 旧批次已完整关闭

feedback_generation_v1：三候选、141保存trace、14205实际完成动作、0碰撞、0新增完整物理族/训练族。两个600秒构造截断，一个16配置穷尽。原GPU1进程已清理，无占位借用。旧失败全部保留。

主agent接收[语义/几何审计](../../data_pipeline/mechanism_runtime_v1/feedback_diagnosis_v1/semantic/REPORT_ZH.md)与[存储审计](../../data_pipeline/mechanism_runtime_v1/feedback_diagnosis_v1/storage/REPORT_ZH.md)。语义子模块4项CPU测试经主agent复跑通过；审计核对85897条日志链与141个trace。关键结论：

- 1LXtFkjw3qL全16次anchor_A目标生成为空，不是证实场景无可达目标，也无证据断言错楼层；当前欧氏排序起点没有先保证四角色可达。
- 另两屋已保存动作约80%为转向；原构造额外执行膨胀逆轨迹，再执行真正使用的紧凑回返。
- 71473条budget日志，每次成功journal append包含三次fsync；可定位156.284秒的checkpoint提交和邻接控制区间，不把其余所有耗时都编造为IO。
- 14205实际动作与14144个完整trace动作差61，来自两个截断尾；这些动作已计预算但不可当训练轨迹，不能说所有尝试都complete。

## 新版已经启动

[compact_loop_v2规格](../../data_pipeline/mechanism_runtime_v1/compact_loop_v2/SPEC_ZH.md)只修发现过程：

1. 几何预筛四角色都有<=8m合法navmesh候选的源坐标；不挪动原起点。排名固定、最多4坐标×4朝向，不因结果替换语义角色。
2. 提前实测公共尾部没有关键事件，排除最终矩阵必然拒绝的配置。
3. 实际执行去程+紧凑回返，检查事件和agent/RGB/semantic完整位姿闭合<=1e-5；不执行未采用的膨胀逆轨迹。完整matrix、数值join/边界、公共近期窗口、27重放54求值与导出质量仍由原验证器执行。

主agent12项CPU测试通过。首次V2在零仿真动作时被旧store输出根保护拒绝，完整留在compact_loop_v2/run_v1；[恢复修订](../../data_pipeline/mechanism_runtime_v1/compact_loop_v2/recovery_v1/AMENDMENT_ZH.md)将只读加载store实例的输出根严格限定为新recovery目录，另外3项测试实际写入/读取/越界拒绝通过。没有把旧失败改PASS。

实际运行：GPU1，tmux `q35n_special_compact_v2r1`，worker3099129；监督XML已经包含G renderer，实测自身显存保守上界266MiB。只清理自己的Popen进程，不碰外部任务，不依赖不可用pidfd接口。恢复扣除首启动预算，上限2990秒，不增加动作/候选/磁盘预算。

20:43快照：第一屋35个源坐标中7个通过四角色几何可达预筛；已有实际anchor_A 40动作闭合回路，agent和两sensor位置差2.384e-7m、姿态差8.941e-8rad。未采用的186动作展开轨迹没有执行。此为真实子步骤证据，不是新增完整族、加速实测或论文收益。当前完整新族0，继续检查anchor_B/无关绕行/全部续接矩阵。

普通数据保持原GPU5生产，不重启、不扩普通多卡。20:43普通已处理499/1000，严格通过346路线/1037指令/72814指令条件决策；后续以实时进度及终态为准。

## 下一判断与停止边界

先完成这次特殊构造的有界结果，若得到完整族，紧接反捷径、同数据强任务状态标签及因果输入回读；只有这些通过才准入训练。若仍不能成族，按具体event/loop/continuation失败归因，不增加无效探测冒充规模。预算提交合并只在独立CPU目录准备，不修改当前活跃代码。

重点是获得能训练/检验主创新的可靠数据关系；构造优化本身不宣称独立算法创新，工程回路成功不当作泛化导航正收益。

20:46补充：[单次预算提交CPU实现](../../data_pipeline/mechanism_runtime_v1/feedback_diagnosis_v1/budget_optimization_cpu/REPORT_ZH.md)已接收，主agent复跑21项测试通过。正常成功/拒绝最终预算状态等价；实际动作前仍须耐久确认，失败poison不继续。故障模拟为内存callback，不是原Journal真实文件故障注入；未做实际fsync计时，未接入当前活跃V2，后续需独立集成验收。不存在本轮已测整体加速或新增训练收益。

## 动态入口

- [特殊当前进度](../../data_pipeline/mechanism_runtime_v1/compact_loop_v2/recovery_v1/run_v1/PROGRESS.json)
- [特殊当前运行目录](../../data_pipeline/mechanism_runtime_v1/compact_loop_v2/recovery_v1/run_v1/)
- [普通后台进度](../../data_pipeline/ordinary_scale_v1/recovery_v4/PROGRESS.json)
- [本次授权](../../authorizations/SPECIAL_DATA_PRIORITY_COMPACT_V2.json)

本报告是运行中快照；结果文件出现后需读终态，不沿用“运行中”断言。
