RESOURCE_NOT_RESERVED

本轮工程未启动，未形成有效评测，干预收益 **UNKNOWN**，不采用恢复规则。
正式启动0次、完成0/100对、A与B各0/100条，环境决策0、模型加载0、optimizer update 0。
这是启动前资源阻断，不是控制器失败或成功。此次回交完成的是阻断证据、旧日志CPU分析、
研究规格与CPU契约；没有新增导航成绩、学习模型、SR40或真机部署结果。

1. 实际来源和修改

审查基准为`9d22783e646ef86232ba3a3a59d7bb9ce8a8c0a7`；实际起点为
`09bdc47e5684cd06411b8d9ba8bb1aa5bbcf30db`、`codex/q35n-cycle-pair-v4-20260917`。
两者之间只有已经提交的V4资源阻断包和CURRENT_STATUS引用。未把后者冒充固定提交。
根目录`.gitignore`、`AGENTS.md`、`README.md`及无关未跟踪资产均保留。

已实际读取的指定文件、额外真实资产与无法访问项见[READ_SCOPE.md](READ_SCOPE.md)、
[READ_MANIFEST.json](READ_MANIFEST.json)和[WORKSPACE_AUDIT.json](WORKSPACE_AUDIT.json)。
本轮重新核验V3源锁198项，0项不匹配；best4k SHA为
`c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775`，模型源码SHA为
`89c2aac37d0d0519c01f51acf8f19b42087434f42e4cf56fc76fd01cf5297389`。
指定既有本地文件没有访问失败。GPU运行身份和数值路径未测，不能用磁盘SHA代替。

旧V4已经终止，本轮在其`codex_return_20260917/`下追加预检和预算证据，不覆盖旧报告。
其PROTOCOL、SOURCE_LOCK、RESOURCE_LEASE、RUNTIME_IDENTITY、CPU_TEST_RESULT、逐步日志、
RESULT、REVIEW、REPORT_ZH全部保留。新规格/代码只在本回交目录和
`research/continuation_memory_v1/`；CURRENT_STATUS仅新增本裁决引用。
冻结模型、恢复算法、训练和评测代码、部署配置、旧FINAL_REVIEW均未修改。

2. 资源、预算与分母

本机未发现调度器分配环境或可核验的调度器命令，也没有实际GPU使用者协调的专用窗口凭据。
没有按空闲显存或本地flock冒充预留，没有轮询抢卡、换卡、启动GPU或向外部进程发信号。
证据见[本轮PREFLIGHT.json](../../closed_loop_bench/ordinary_cycle_pair_recovery_v4/codex_return_20260917/PREFLIGHT.json)。

只用已有V2/V3日志吞吐，假设两臂各保持历史best4k的19122决策，外推约4332–5247秒，
超过4100秒；额外全状态指纹开销尚未知。该情形不是本轮控制组结果，也不是最低必需时间：
B轨迹长度未知、旧部分日志可能有偏，故结论为`AT_RISK_NOT_DEMONSTRATED_FEASIBLE`，
不能声称已证明绝对不可完成。未增加估时试跑、加时或资源预算。
详见[BUDGET_FEASIBILITY.json](../../closed_loop_bench/ordinary_cycle_pair_recovery_v4/codex_return_20260917/BUDGET_FEASIBILITY.json)。

冻结分母仍为INTERNAL_DEV五屋100对、共200个episode，最多100000决策。
本轮A/B的SR、SPL、nDTW、OSR、ΔSR、胜负ID、按屋差值、时延与区间均为null。
旧SR21%不充当A；V1/V2/V3部分轨迹不拼接。原配对门槛及历史保护线均未改变。
无实际干预前缀，不能声称严格logits一致已验证。

V4尚缺evaluate/launch的全参数与持久buffer指纹、处理后输入哈希、在线首分歧失败及安全清理集成；
`runtime_integration_ready=false`。旧15项CPU合同通过不是运行链路已就绪，本轮未重跑或改写其结果。
若以后在启动前发现源码不符或明确预算不可行，应另记`PREFLIGHT_BLOCKED`，不冒充运行中断。

3. 分别保留的历史事实

|范围|已记录事实|本轮解释|
|---|---|---|
|历史41800、完整1839|SR22.02%|已暴露完整集；不是best4k成绩|
|保留best4k、开发100|SR21%、SPL0.18017、nDTW0.36588|旧开发证据，非本轮A|
|更长训练/历史扩展|全epoch开发SR14%；历史实验未选出可保留新基座|不证明所有训练无效或2B容量不足|
|256训练/128检查|训练100%、检查67.19%；检查CE约0.9147升至2.2252|可拟合小集，泛化仍有问题|
|旧循环日志|23条失败存在完整输入重复；8556/19122决策重复|不能换算为可挽回成功数|
|数值复现|跨进程logits差异及部分动作翻转；V1比较无效|FLA autotune仅是线索，未归因|
|恢复V2/V3|资源中断|收益未知、未采用|
|独立部署入口|16输入动作一致，logits非逐位一致|仅离线接口验收；SR40和真机均未完成|

逐项来源和可否定条件见[EVIDENCE_LEDGER.json](EVIDENCE_LEDGER.json)。5487种原生FIT输入和
同一256/128小集没有重跑。CURRENT_STATUS与旧FINAL_REVIEW用于事实恢复，旧STATUS不恢复任务。

新做的只读STOP分型覆盖旧100条：21成功；18进入范围后离开才远处STOP；8进入范围后耗尽；
37从未进入范围而远处STOP；16从未进入范围且耗尽。26条进入范围未成功者中6条也有重复决策。
逐episode和STOP margin见[STOP_FAILURE_AUDIT.json](STOP_FAILURE_AUDIT.json)；距离仅用于离线分析，
没有进入策略输入、修改STOP或构成新的A/B证据。

4. 研究主张仍为UNTESTED

唯一候选主张是交叉续接结果对部署时任务条件记忆的增量，状态固定为
**RESEARCH_HYPOTHESIS_UNTESTED**。控制器结果不证明或否定该学习方法。
[研究主张契约](RESEARCH_CLAIM_CONTRACT_ZH.md)与[研究目录](../../research/continuation_memory_v1/README.md)
给出schema、防火墙、真实接口、模型/梯度规格、强对照、未来预算和终止条件。
没有实现神经记忆模型、训练器或研究导航评测器；规格完成不等于方法完成。

最强简单替代B2直接从同一精确检查器自动生成组合式执行状态，已有CPU适配函数
`check_tasks.py:exact_state_targets`。若z是充分状态，则Y=f(z,g,events(c))，交叉结果不包含
比z更多的信息。未来必须在同架构、同输入、同初始化、同轨迹池和预算下比较B1/B2/Ours；
若B2相当或更优且更简单，撤回交叉续接模型增量，不能以“自动标注”或换名称维持主张。

5. 真实资产和下一关缺项

已用现有Compiler/FamilyLoader回读一个真实候选族：3历史×3续接×2任务，18标签重算一致，
424个内容数组通过校验、492个动作监督owner。全库215份导出元数据涉及11屋；这不等于
215个独立合格族或11个新测试屋，独立物理族数量尚未认证。

该族是已暴露的SEE2后STOP语义，不是走入房间；H_A_I是已完成子目标的合法重访，不能
冒充事件无关绕行；缺历史无关g2负控制。9条规范轨迹含既有微小共享状态归一化，不能
将规范RGB等同于整条原始回放逐位相同。完整族隔离、可见事件、共享物理状态与全部关系
仍须按新契约验收，状态为`DATA_ASSET_GAP`，没有自动补造数据。

按真实事件位置，从首次两帧关键证据覆盖到监督点需要233/236次观察；保守完整前缀249次。
32步展开会漏掉H_B全部关键事件，短展开或逐步detach不能声称训练了旧事件写入。
CPU合同未测Qwen跨步梯度；实际可微展开的显存/吞吐、未来资源上限与合格基座SHA仍待确定。
没有虚构GPU小时或优化步数；草案null字段阻止训练启动。

6. 运行命令、验收与消耗

本轮实际执行的两项CPU命令（vla根目录）：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/reviews/Q35N_CODEX_RETURN_20260917/audit_existing.py
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/research/continuation_memory_v1/test_contracts.py
```

输出采用exclusive创建且已经封存，勿直接重跑覆盖。资产审计记录墙钟18.405秒；
新CPU合同14/14通过，覆盖真实资产投影、未来输入/ID防火墙、任务切换、历史回写、
UNKNOWN、查询词义绑定、精确状态、族隔离与族损失权重。
测试不包括神经前向、autograd或闭环效能。产物清单/字节数与结构校验见
[DELIVERY_VALIDATION.json](DELIVERY_VALIDATION.json)，最终指纹见[SOURCE_LOCK.json](SOURCE_LOCK.json)。
本轮新增GPU小时0、仿真交互0、真实造数0、正式launcher墙钟0；未测本轮模型显存/RSS。

7. 停止与最小决策需求

工程下一次判断先需要真实GPU预留、4100秒内可完成的可信既有证据，以及V4运行审计集成；
随后仍须完整有效100对才可判收益。研究训练还需合格基座、合法完整族、无泄漏与跨步梯度、
强B2和明确预算。G1/G2/G3均未通过，见[NEXT_GATE.json](NEXT_GATE.json)。

本轮在此停止并回交，不自动进入第二个导航实验、完整val_unseen、研究造数/训练、方向B或上线。
已暴露INTERNAL_DEV及1839公开验证不能通过改名变成盲测；未来SR40也只按声明协议解释，
论文泛化须冻结后独立场景或合规隐藏测试。
