# 多族生成协议与 CPU dry-run：待主 agent 审核

结论：`CPU_DRY_RUN_COMPLETE_RUNTIME_NOT_AUTHORIZED`。已给出20个不同来源物理路线的候选种子，首批固定5个、每个FIT_PILOT房屋1个；**没有生成新机制族，也不知道这些种子能否构造合格几何族**。当前`executable=false`，不启动仿真/GPU/训练，不读取SFT结果。

## 1. 实际完成与来源边界

CPU从已封存的SOURCE_CANDIDATES逐条聚合：10,819条来源记录、3,603条物理参考路线、61房屋。候选只来自原5个FIT_PILOT，原56个RESERVED_UNASSIGNED及其旧暴露标签完全保持。只读取现成manifest元数据，不打开房屋资产、保留房屋图像或新观测，不导出原指令文本/坐标。已通过9项脚本测试；12个列举的共享元数据及compiler源文件在dry-run前后哈希一致，详见[保护检查](PROTECTED_HASH_AUDIT.json)。这不是重新认证所有场景资产。

实际文件：[协议](protocol.json)、[20候选](CANDIDATES.json)、[61房屋清点](HOUSE_INVENTORY.json)、[CPU结果](CPU_RESULT.json)、[测试结果](TEST_RESULTS.json)。

## 2. 分阶段目标与不偷换的统计单位

P0首先只试5个候选bundle，各房屋一个。房屋沿用SPLIT的顺序，每房屋按source physical route hash排序，首先rank0；P1草案才扩rank1–3，共另15个。每bundle的runtime起点仅来自该路线原始start/reference waypoint前8个不同位置；不足8个不补齐。yaw采用0/180/90/270度分散覆盖，tail先LRLRLRLR再FFFFFFFF，循环tail→yaw→u，上限64配置。这样首批不会再次在两个起点耗掉全部预算。dry-run不读取原路点，实际u坐标保持null。

发现首个完整预检族即冻结；其认证失败不在该bundle中换第二个候选。P0无论成功多少都收口，主agent先审核产率/失败/捷径，再批准P1。若5个均失败，先完成有界构造诊断，不自动遍历剩下房屋找正例。20个真正不同几何族是后续P2目标，不是20候选必定产生20族；每房屋最多4个目标几何cluster，达不到则诚实报告，不能拿同一族的三seed/18格/改写/交换任务名称充数。

去重分两层：完全相同的house/asset/config、汇合状态、实例角色和物理历史续接属于同族（忽略seed/ID/措辞/角色别名）。保守几何聚类在同房屋中按汇合位置水平<1m且高差<0.5m连接，或按同见证实例集合且0.25m网格路线占用Jaccard>=0.8连接；连通分量只计一次几何目标。不同hash的R2R路线只证明来源路线不同，不证明生成后的族几何不同。正式不确定性仍按房屋聚类，不能把4个同屋族当4个独立屋。

## 3. 预算草案（尚未获运行授权）

P0每bundle发现上限12分钟/20,000实际运动原语/64外层配置，首个冻结族认证上限8分钟/20,000原语。全批最多100分钟/200,000原语、16GiB新增磁盘、16GiB RAM、1张卡且8GiB GPU；所有规划试走、失败支路、raw inverse均计入实际原语。任一限额先到即停止，资源截断独立标记，不当作语义拒绝。单族固定27真实回放/54任务求值，当前首族实际8,631运动原语只是预算参考，不保证其他族同成本。不得挤占外部SFT；卡号/lease必须另行核定。P1/P2需独立累计预算，不能复用P0授权。

账本同时报planned/started/ineligible/unknown/rejected/resource-censored/frozen/certified/duplicate/geometry-clusters及逐房屋失败、动作、时间、字节。至少给certified/started和certified/planned两种yield，未开始不得当失败；crash重试有新attempt_id且共享旧预算，不静默清零。

## 4. 当前已知捷径：不能直接复制首族训练

并行捷径审核已反馈：单族task+query/短窗经验上限14/18，全历史动作计数+task+query可经验拟合18/18，M2程序oracle也为18/18。这是单族有限表格的身份拟合风险，**不等于已测部署模型作弊，也不证明方向全局不可行**。相同总长度远不足以排除F/L/R累计次数等捷径。

因此P0只诊断构造。正式机制训练前需通过以下至少一条辨识门：真实同task/query/累计F-L-R计数但不同Y的历史反例；或事前冻结、在真正不同留出几何族上完成严格count-only捷径审计，所有参数仅在fit选择、房屋分组不混。无序历史/视觉bag、历史动作计数、程序状态M2均作为明确强对照。所有18格仍保留，报告原始先验准确率；机制主要诊断用同task/query的历史对立配对及balanced accuracy，不把12/18多数类值称收益。

现有H_A/H_B/H_A_I是2:1历史频次，P0承认该不平衡。正式训练前可另版真实构造H_A/H_B/H_A_I/H_B_I的对称族，或审核固定分组权重；新增历史不能纸面复制，需新schema、重放与预算。当前“先见anchor再末端见bed”可能只需记住anchor曾出现，不能由此宣称一般时间顺序推理。若要同事件集合而顺序不同、标签相反，必须另版扩展程序并真实重放，不虚造负例。

阈值仍固定同实例连续两真实帧各>=256像素。并行敏感性诊断192px改变3/54，224–320稳定；不据诊断调整主标签。

## 5. 数值重建与质量门

继承首族显式数值干预：历史真实回到初始合法u之后、公共8步之前最多一次重建，位置及角度修正分别<=1e-5m/rad；初始target在结果出现前固定。保存raw/canonical pose、RGB和semantic，边界SEE2实例集合必须完全不变，规范化后的共同u/s及传感器/RGB/semantic严格一致。历史<=512，续接<=160，零碰撞，未知不转负，raw_exact_merge单列，不称自然逐像素汇合。当前首族最大约1.67微米，不作所有未来族的实测值。192/256/320px、1e-6/1e-5容差敏感性只作披露，不改主门。

每族必须从原始像素重新求Y，验证同task/query异历史反转、无关绕行不变、跨任务条件性、当前两RGB和最后8动作完全一致、完整因果prefix、未来query隔离及动作去重。失败的餐椅实例、原精确像素汇合及所有G1R搜索节点保留。

## 6. 必须先泛化的固定代码（不得原地改封存代码）

| 位置 | 已见硬编码 | 新版本需完成 |
|---|---|---|
| `G1F.../engine.py:20,94,214` | 17DR场景/GLB/u_pool过滤 | 配置化house、授权资产哈希、来源route和u枚举，cache key加入house/config/roles |
| `G1F.../engine.py:95,117` | GPU2，D/L驱动raw category过滤 | CLI设备参数与独立lease；显式event-role/category/raw/region映射，不靠D/L字母交换 |
| `G1R_TASK_INSTANCE_V3/worker.py:38` | u_index22/yaw0/tail、F17及TV/sink任务 | 候选配置只读载入、全量尝试账本、通用无语义ID、模板来自结构程序 |
| `mechanism_family_v1/compiler.py:13,188` | TASKS/ANCHOR、bed终点 | 显式任务程序/词表版本；保持语义query编码独立于角色或标签 |
| `mechanism_family_v1/certify.py:40,195,253` | V3专用schema改写、17DR、interface_only | 编译schema版本与candidate配置一致；house/split/legacy来自冻结账本，不自动升级train |
| G1R数值worker/reader | 继承链、全局core.OUT和KINDS | 新包显式依赖注入、每进程独立配置/缓存；不可并行复用可变全局对象 |

建议未来CLI契约为 `generate --config FROZEN_CONFIG.json --candidate-id MP5_00 --authorization AUTH.json --output NEW_NODE` 和 `certify --frozen-candidate ... --authorization ...`。**当前不实现这两个执行命令**，不让dry-run绕过准入。现有可运行的唯一CLI是：

```bash
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/parallel_readiness/v2/multifamily_plan/dry_run.py --dry-run
.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3 -I -S -B projects/qwen35_indoor_nav/parallel_readiness/v2/multifamily_plan/test_dry_run.py
```

## 7. 分组与下一准入

现有5屋只是FIT_PILOT，不提供独立家屋dev/confirm。首个17DR机制族维持interface_only，不自动回填训练。56屋保持reserved；未来需主agent先登记house分组/旧暴露，再读取资产，不根据生成成功或SFT结果挑confirm屋。模板/程序组合另行分组，同一几何族及其所有角色别名、前缀、续接、措辞同组；当前只有TV/sink→bed单组合，不能声称组合泛化。协议给出fit/措辞probe模板草案，但措辞probe仍是seen-house表面表达检查。

主agent下一步可以批准**新版本通用构造器CPU实现与测试**，随后在捷径门审计和runtime预算另获授权后运行5候选构造诊断。尚未批准大规模生成、机制训练或科学PASS。
