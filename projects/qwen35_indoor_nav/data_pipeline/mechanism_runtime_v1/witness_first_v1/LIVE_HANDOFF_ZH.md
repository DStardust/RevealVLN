# 持续生成任务当前入口

## 自动接续最新覆盖：2026-09-10 08:12 CST

- 后续实测补充（本段优先于下述启动快照）：GPU1 batch200也独立3/3强质量PASS，已自动进入batch201（child3505843）；GPU2 batch101持续生成。两张特殊卡都完成一次“生产→审核→下一批”的真实接续，本轮新增6合格族。普通autoV4已过222秒，73完成路线持续增长，40次遥测接受/1次明示修订接受/0拒绝、6次磁盘检查；越过旧25秒及172秒中断点。首4路线已在线审核/957条件动作，但整批merge未完成，不加入上方历史正式总量。

- 本段覆盖下文旧活跃状态。三条实际队列：GPU1 `q35n_auto_special_gpu1` / `auto_production_v1/special_gpu1_queue_v1`（queue3487086，batch200开始）；GPU2 `q35n_auto_special_gpu2` / `special_queue_v1`（queue3477647，已经自动从batch100接续batch101）；GPU6 `q35n_auto_ordinary_gpu6_v5` / `ordinary_queue_v5`（queue3504153，supervisor3504180，worker3504208）。不要重复启动这些PLAN。
- 特殊自动接续已有真实证据：batch100正常结束后，`quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/auto_generation_v1/batch_100/result.json` 独立3/3质量PASS、18cell/27replay/54eval逐族通过，随后调度器自行启动batch101（child3498174）。GPU1/2合计24固定批次、72语义排重候选，10FIT屋/22物理hub；每lane12h墙钟，正常生产→强审核→下一批。72是预约候选，不是72合格族，也不是72独立hub。
- 普通实际使用 `ordinary_fullscale_source_v1/auto_generation_v4`、独立sibling `auto_generation_v4_production/shard_0002→0004`，1362从未尝试路线/4085官方指令。输入锁280项SHA `8dfe1664f70ba1f38863deb0fc0b2fad3c6badd8ab44c112d35749b7358c9eb3`；主agent真实stdlib/Habitat双审批probe、12遥测+8运行测试通过，独立审核12+6测试与560差分通过。首片已真实完成8路线，尚未闭合严格merge，不计正式增量。
- 修复版本历史全部保留：queue_v1/v2审批exact-dict错误；queue_v3/runtimeautoV2 60终态+1partial后嵌套数据扫描FileNotFoundError；queue_v4/runtimeautoV3 0终态+1partial后GPU_MEMORY_ACCOUNTING。新source排除所有这些终态/partial；未改旧FAIL/旧锁或重跑旧数据。普通链209.07168883481063秒已扣除，剩11910.92831116519秒，GPU6借用前精确holder3491246/start135025717/pane%151/target vla_idle_occupancy_20260904:3.0，结束由原transport finally恢复。
- 普通autoV4是明示工程遥测协议修订：仅active worker计量不一致用max(device,total-process)保守计费，并要求不一致时全卡该上界<4096MiB，外部进程/总量上限不变；raw原值先持久化，派生接受另存，不能冒充原guard通过。idle/restore/disk/数据质量阈值不变。真实运行19.879秒已记录device883/process947的原guard失败及新规则保守接受，非仅CPU模拟。数据位于metadata目录外，避免原扫描把PNG临时重命名误当文件损坏。
- 已闭合严格普通合计仍5013路线/15039官方指令/1223394指令条件动作；本轮active及autoV2未merge的60终态不加。GPU0及GPU1/2外部进程未动；后部其他卡当前占位、不再误报七卡生产。量级目标未完成，队列有预算而非无限扩源，不训练。
- 只读实时入口 `data_pipeline/auto_production_v1/status.py`。源码queue.py保持SHA `e606eab59bb654bb31583333cfa2367780f908be70356594f97c9eeb27422683`；运行中禁止再运行会争抢GPU flock的queue测试。失败会停本lane并留账，不循环重试。状态查询须同时看producer PROGRESS，不能只看tmux/GPU利用率。

## 自动接续最新状态：2026-09-10 07:42 CST

- 用户要求“现在补齐自动生成”。真正调度入口为data_pipeline/auto_production_v1/queue.py，已主审19测试+独立8反例通过，源码SHA e606eab59bb654bb31583333cfa2367780f908be70356594f97c9eeb27422683。互斥锁由child继承；SIGTERM/INT只排空当前原transport，不杀GPU；生产/审核失败停止该lane不自动重试；输入/审批锁、整批时间及CPU审核预算固定。status.py仅只读显示真实queue进程/终态，不用GPU利用率冒充产出。
- GPU2实际自动生产：tmux q35n_auto_special_gpu2，主PLAN auto_production_v1/special_queue_v1/PLAN.json，queue PID3477647、首批batch100 supervisor3477674/worker3477684。注册batch100…111共12批/36候选、10FIT屋22hub，12h总墙钟；每批原3900监督/3600factory/60000动作/7GiB不变、审核后再下一批。当前首批已有39完整trace/11018动作/0碰撞，未完成整族质量验收。审核必须用WF/auto_generation_v1/audit_adapter_v2/audit.py（修正原audit的OUTPUT_SCOPE，旧文件不改）。12批全部source主审通过。GPU2无holder借用，外部context不动。
- GPU6实际自动普通生产：tmux q35n_auto_ordinary_gpu6_v3，主PLAN auto_production_v1/ordinary_queue_v3/PLAN.json，queue3480803/supervisor3480805/worker3480831；runtime ordinary_fullscale_source_v1/auto_generation_v2，新输出其production/shard0002→0004，426+998=1424未尝试路线/4272原指令。当前首片42/426、41回放certified，严格审计未到批边界不计最终。锁276项SHA5fa26fa29a453136593ab6e9895d287bca50a6cdfce66be1d62da1065848a5ea；25测试+stdlib/Habitat两独立Python真实common.approved调用通过。借holder3474528/start134900793/pane%151/3.0，finally必须恢复。
- 保留启动失败：ordinary_queue_v1因main审批额外说明字段违反exact dict在prelane停止；ordinary_queue_v2桥接仅parent导致worker同审批失败，0route/0ledger，原finally恢复3474528，12.349秒已由新V2扣账。两个queue输出、auto_generation_v1及approval_bridge_v1均不改不重跑。新V2使用原精确五字段审批、独立runtime/output，无bridge、不改资源guard。
- 旧GPU6 shard2实际573终态的独立CPU回收已完成：auto_generation_v1/salvage_old_v1/run_v1，284严格路线/853官方指令/68345条件动作，唯一partial排除、旧GPU_MEMORY_ACCOUNTING失败保留。至此已完成普通池合计5013严格路线/15039原指令/1223394条件动作（不含旧pilot和活跃临时数据）。旧GPU3/4/7已正常closed并CPUmerge、holder恢复；不再是活跃队列。
- 特殊旧05r1/08/06r2已各3/3强审完成，06r2额外386 raw XML/RESOURCE闭合审核通过且0实际补采。原四成员cohort仍FAIL，不拿新批替换。GPU1独立auto_generation_gpu1_v1正CPU准备额外12批排重候选，尚未GPU；不允许它锁GPU2活动HEAD/journal，只以GPU2冻结预约排除。其他后部卡目前仍原占位，GPU0真实任务未碰。
- 所有数量目标依旧未完成；有限队列不是无限扩源。原英语源不足300万的事实保留，未获许可不混入RxR非英语。当前不训练。下方早期“七卡仍生产”已过时，禁止重启已关闭批次。

## 最新覆盖状态：2026-09-10 00:47 CST

- 已实际扩至七张GPU：GPU1特殊batch_05r1（q35n_witness_batch05r1）；GPU2特殊batch_08（q35n_witness_batch08）；GPU5特殊batch_06r2（q35n_witness_batch06r2）；GPU3普通runtime_v5/shard6；GPU4普通runtime_v6/rescue_production/shards0→1；GPU6普通runtime_v3/shards2→4；GPU7普通runtime_v4/shards3→5。全部tmux实读存在且持续产出。GPU0真实任务及GPU1/2外部context未动。不要重复启动。
- 普通三个已完成且独立验收组：ordinary_scale/recovery_v4=770路线/167870指令条件动作；ordinary_parallel/merge_receipt_v2=780路线/167127动作；ordinary_fullscale_source_v1/salvage_rescue_v1/run_v1=814路线/235410动作。合计2364严格合格路线、7091官方指令、570407指令条件动作；不含旧pilot或活跃临时计数。salvage保留原预算截断失败，2条interrupted永久排除，953未尝试路线已由GPU4新命名空间生成。
- 特殊本轮正常闭合强审核新增9族：batch03r1/04r1/07各3族，每族18cells/27实际认证回放/54求值，均0碰撞，审计quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/independent_03_04_07_v1。旧四批cohort仍含06r1资源FAIL，不能PASS，06r2/07/08不得补位。活跃05/06r2/08虽已有producer族声明，强审前不计正式合格。
- new_hub_scout_v1已正常闭合：8WUmhLawc2A同屋4个新hub，196完整trace/9387动作/0碰撞；new_hub_bank_v1已CPU完成96新候选（总银行442候选，不是442族）。batch08实际认证新hub indices[0,24,48]，同屋相关，非泛化测试。新锁406项SHA5692b3d17e2edf066ea5333b044fe02c71a907f18f71c1af13ecf2e7dddc0b80。
- GPU5 batch06r2为显式遥测协议修订的新工程试验：1061项锁SHA90ca2f42fc48e8fbd6e8677481c94a0e4fb8f58518b831e7d8a57c287224e077，旧06r1零运动但有1观察/真实trace的失败不改。raw先持久化、仅原MEMORY_ACCOUNTING且保守内存界全过才1秒内最多2次补采，原预算/阈值不变；目前没有实际补采事件，不能声称已证明旧错误根因或重采修复效果。独立telemetry_consistency_audit_v1主审20CPU测试/4哈希通过，仅正常闭合后额外审核。
- 普通活跃分配互斥：GPU6[2,4]=1998路线，GPU7[3,5]=1997，GPU3[6]=651，GPU4rescue[0,1]=953；5599活跃分配+旧0/1终态1041+interrupted2=原完整FIT池6642，物理/官方别名交叉重复均0。不要直接按重复shard编号相加。runtime_v3_gpu6_merge_v1主审11CPU测试/全锁ownership通过，只允许GPU6两片成功且恢复后独立合并；旧V3全局merge不能用于新ownership。
- 借用待finally恢复的精确holder：GPU3 PID3294720/start132284629/pane%153；GPU4 PID3294955/start132285808/pane%149（保留原cache环境）；GPU5 PID3288797/start132243828/pane%235；GPU6 PID3256028/start132020313/pane%151（自身临时sleep3265520/start132082125）；GPU7 PID3270116/start132111485/pane%152。GPU3/4/5/7使用已审dead-pane恢复，GPU6原safe sleeper恢复。成功/失败均须恢复，不能停止其他真实任务。
- 当前主审启动全部有界生产者；Ohm等待08正常闭合后独立强审，Newton等待05/06r2并做raw附加审核，Hilbert只读普通监测及GPU6关闭后独立merge。旧自动接续/恢复/salvage/scout后处理均已结束，不重跑。普通300万动作/特殊1万完整族目标尚未完成；RxR印地/泰卢固扩源仍待用户答复，不混入英语池；本轮不训练。

## 最新纠正：2026-09-10 00:14 CST

- batch06r1在启动11.27秒后MEMORY_ACCOUNTING/-15停止，cleanuptrue；GPU5已再次恢复3288797/start132243828/pane%235/29334MiB。它有实际零号trace/content/journal，不能记零尝试；失败GPU原始sample因旧代码先check再save而缺失，不能编造驱动根因。旧四批cohort因此不能PASS，后续重试不得替换此成员。
- batch07正常closed且独立3/3quality PASS（136traces/33444动作/0碰撞）；对应强审核目录independent_03_04_07_v1/batch_07。new_hub_scout已由session45747接续启动q35n_new_hubs_gpu2，正在4个新位置真实采集。GPU5现占位，其余6卡在产出。
- Hilbert正CPU-only准备telemetry_consistency_v1：仅对原MEMORY_ACCOUNTING且保守全卡观察值<4096等前置全过者，有界再次采样、durable保留每次raw，最终必须原guard通过；尚未批准接GPU或新trial。不得改06旧失败/来源锁/既有预算。
- Ohm已启new_hub_bank_v1/waiting_v1纯CPU等待器exec87376（75分钟上限、40秒采样），真实新scout正常关闭后才--check-only与一次prepare。不重复启动scout或后处理。

## 最新补充：2026-09-10 00:11 CST

- batch04r1正常closed：134 traces/28330动作/0碰撞，3/3独立强quality PASS（3屋3hub，每族18/27/54，M2重算），审计gate_v3_manifest_capacity/independent_03_04_07_v1/batch_04r1，17封存哈希。仅单批，cohort稳定仍待四批齐，不混batch07。
- GPU5已恢复并再核验holder3286258/start132229028/pane%235，MAIN批准batch06r1事前indices9/10/11，1041文件锁96ed7cdd58911e7bc20d9427c424b3a40a7789cccb360050a8a3feaac341448d；00:09启动q35n_witness_batch06r1，worker3288543。此借用待finally恢复。
- 主agent已部署三个有界自动接续/恢复等待器，不可重复启动：exec session50676等03r1正常关闭后校验05r1锁c93a7833ddb92e4f1afcb6be2cc872104bceca8fec0cf4a2d59c7e21966387de并启动q35n_witness_batch05r1；session45747等07正常cleanup后启动q35n_new_hubs_gpu2（4个新位置，不是新合格族）；session86053等普通GPU3/4终态仅精确识别空argv恢复竞态+原预算/正常关闭后，调用已审holder_recovery_v3/v2恢复原占位。各等待上限30分钟，不改活跃预算/未知错误不动作。
- new_hub_bank_v1已主审13测试/13SOURCE锁，Ohm只读等新scout关闭后先--check-only再CPU候选银行；Newton只读监测03/07及05/06，四批全部闭合后原cohort强审核。Hilbert已CPU准备ordinary_fullscale_source_v1/salvage_rescue_v1，但main未审该新模块，尚未运行数据回收/新来源/新增GPU任务。

## 最新补充：2026-09-10 00:03 CST

- GPU7 runtime_v4 已经主审源码、25项CPU测试通过/1项冻结后跳过、249来源哈希通过，并启动 tmux q35n_ordinary_full_gpu7_v4。真实worker3279926已产出完整路线，分片3后自动5，1997路线；新执行锁a10c94789002cd6373d5522595935cea9bbebceb082cb3242272657f8a107669。借用holder3270116；不创建sleep，dead pane保留至finally恢复。旧V3失败/原生产根metadata锁均不修改。
- GPU1/2/5特殊、GPU3/4/6/7普通，共7路真实生产。GPU0不动。new_hub_scout_v1主审13测试/138来源通过并已写MAIN_AGENT_APPROVAL，但尚未运行，须等batch07闭合GPU2释放后启动。
- holder_recovery_v3（只恢复GPU3/4已核实占位，不认证截断数据）主审14测试/9来源通过；预计旧普通shard0/1在00:16左右按原3480秒早停。Hilbert另准备CPU salvage/rescue，GPU恢复仍由main执行。

## 最新覆盖状态：2026-09-09 23:50 CST

- 用户准许稳定后使用其余占位卡增效；不动 GPU0 外部真实任务。任务仍为两种数据量产，无训练。后文旧状态全部仅历史，禁止重启旧批。
- 活跃特殊：BE=batch_execution_v1，batch_03r1 GPU1 / tmux q35n_witness_batch03r1；batch_04r1 GPU5 / q35n_witness_batch04r1（已借占位3242643，用后须恢复）；batch_07 GPU2 / q35n_witness_batch07（仅新fresh-only clock batching，原27认证不变）。03/04首族各physical=1，尚未整批quality审计。
- 当前事前cohort：multi_program_bank_v1/language_ready_v2/COHORT_V2.json，03r1/04r1/05r1/06r1 共12固定程序。05r1已prepare但未启动；06r1待GPU5当前批恢复后新身份prepare。旧03/04只有准备，因软件来源清单1024上限未启动；V4入口仅此上限2048，新审核必须用quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity，其他门槛不变。
- 活跃普通：../ordinary_fullscale_source_v1/runtime_v2，GPU3 shard0=999 / GPU4 shard1=997；原每片3480秒早停可能截断，不能改活跃锁。runtime_v3 GPU6已启动shards2/4（1000+998，预算12600/4200秒）；GPU7启动时临时shell→sleep竞态preworker失败，没有worker/data，旧失败保留。
- GPU7已由main在../ordinary_parallel_v1/preworker_holder_recovery_v3/run_v1独立恢复，新holder3270116/start132111485/pane%152/29334MiB。agent正准备runtime_v4仅GPU7重试原shards3/5，采用remain-on-exit保留dead pane、不创建临时sleep；尚未启动。
- 已完成普通：old ordinary_scale recovery_v4=770合格/167870指令条件动作；新ordinary_parallel_v1/merge_receipt_v2/merge=780合格/2340指令/55709独特路线决策/167127指令条件动作，4片1000全部终态，200隔离+20生成拒收。新索引SHA8af3c51a49b7ae77aee10090ee8744d1448b5c6e47d87b1928fce5921e960fba。两卡旧临时sleep空argv导致旧restore=false，main外部恢复都true；不改旧result。
- 旧特殊batch02r2自然closed：3尝试2常规quality PASS，GPU5恢复。加3个外部独立恢复FIT族=5不同hub/4屋；short_revisit_v3另有1文件但同hub不加独立数。production_inventory_v1有3恢复族的main准入索引；非模型收益。
- 12屋scout全部closed：原6屋+bulk_source_v1/shard_00三屋+next_shard_runtime_v1三屋。银行multi_program_bank_v1/2/3为215+63+68=346候选，20有候选hub；候选不是族。新new_hub_scout_v1 CPU准备8W屋4个距旧hub>=1m新hub，未GPU。
- 数量目标保留DUAL_PRODUCTION_TRAINING_SCALE_V2.md：普通>=300万有效动作，特殊工程规划1万完整族。现英语全源8742路线预计约150万动作，尚有缺口；main已异步问是否允许独立RxR印地/泰卢固附加池，未获答不混入。两种正式规模均未完成。
- 主agent只负责GPU launch/restore及共享状态。Ohm：新hub CPU准备；Newton：3个特殊批只读监测/closed后质量；Hilbert：GPU7 dead-pane运输修复、普通监测/资源截断后独立恢复与rescue源。任何INPUT_LOCK/SOURCE_LOCK内源码不改。

## 早期历史（不是当前执行状态）

用户要求：在找到稳定生成目标特殊样本的方法之前继续；后来询问是否已有普通/特殊稳定方法。已答普通批量管线可用、特殊首族合格而跨屋稳定仍待完成。不要从这条状态询问推断用户要求停止。

当前两个独立真实批次（禁止修改任何所锁定源码）：

- batch_execution_v1/batch_00/run_v1：GPU2，tmux q35n_witness_batch00，worker3159138。固定3屋各1候选，原winding recipe。
- batch_execution_v1/batch_01r1/run_v1：GPU1，tmux q35n_witness_batch01r1，worker3179642。固定3个未使用hub、含第四屋；新有限文案规范于动作前冻结，其余角色/动作/任务结构不变；readiness_v1接受真实util0，不放宽原资源护栏。

旧batch_01仅创建3个preworker文件后退出，具体异常未被记录，绝不虚构GPU_NOT_IDLE已经实测。独立BATCH_01_PREWORKER_EXIT_RECEIPT.json保留；不能把它计物理尝试或改旧目录重跑。

已通过独立质量的首族：quality_cpu/batch_acceptance_v1/v3_final_audit/FAMILY_REPORT.json，18cells/27认证replays/54evaluations/F16L116R116。与batch_00首候选同hub，不能计两个独立hub。仅受控FIT机制数据，不是模型收益。当前论文主线、checker/阈值和训练禁令不变。

目标初步工程门槛：quality_cpu/batch_acceptance_v1/SPEC.json，至少6合格独立hub/3FIT屋、至少2事前固定批次，每批至少3独立hub尝试且至少2合格，距离<1m合并。仅初步跨屋重复生产，不是统计稳定/导航泛化。正式审计必须整批自然closed、内容/预算/资源/来源全部锁定；builder逐认证phase选27实际轨迹，不能把discovery/组件/重复seed凑数量。

两个scout已自然收口：scout_v1旧3屋，scout_next_v1/shard_0新3屋；共6屋组件。片1没有批准且没运行。scout_next_v1/runtime_v1是新分片transport，不能重跑已完成片0。下一未使用hub候选在winding_balance_v1/next_source_v1/snapshot_v2/HUB_OVERLAP_PLANNING.json；只有配置index0与batch_01r1重合，其余5个待候选准入。未启动第三真实批次。

普通recovery_v4已自然完成1000路线：960重放证书，其中770严格合格、190隔离；另40失败。严格2307指令/167870决策。GPU5占位已恢复PID3133144并核实cmd/cwd/tmux；当前普通不是活跃worker。不将全部1000或960都计训练量。

GPU1/2一直使用低显存非独占context，不曾停外部进程；本轮没有新借用占位，完成后只清本worker。其他已占用GPU没有操作。所有内容仅本项目内。

分工：mechanism_store只读监测两批并在closed后独立质量/总批次审计；mechanism_data_prepare做新quality_redteam_v1 CPU负例测试；ordinary_recovery_v2已交全新3屋快照及语言实现。共享STATUS/README/AGENTS由主agent独占。新代码文件可添加，但不要改batch_00或batch_01r1 INPUT_LOCK中已存在文件；哪怕主prepare/shared更新也会使审计失败。
