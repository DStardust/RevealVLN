# Q35N 独立路线工作约定

- 2026-09-17 当前用户要求针对性修复、推进基础导航及后续方向，必要时上传GitHub新分支交Pro指导。最新入口为 `CURRENT_STATUS.json` 与 `reviews/Q35N_RECOVERY_20260917/REPORT_ZH.md`；下列旧运行/禁止条目按各自历史节点理解，不据此重复借卡或重启已关闭实验。
- 本轮400步 `ordinary_learnability_v1` 已完成PASS；条件LoRA扩覆盖跳过，小集checkpoint不部署。`deployment/ordinary_v1` 独立接口通过。`ordinary_cycle_recovery_v1` 因干预前数值/动作差异已主动终止，不采纳部分分数；`ordinary_cycle_pair_v2` 因外部任务进入GPU4资源中断；`ordinary_cycle_pair_gpu1_v3` 迁移GPU1后同样因外部任务进入资源中断（19/200）。本节点已关闭，不重试或复用旧run，全部自有进程已清理；按用户兜底方案准备新分支与Pro交接，旧源码及结果保持只读。
- 用户新增“小物体语言导航”为未来备选1，见 `FUTURE_OPTION_1_SMALL_OBJECT_NAV_ZH.md`，尚无方法新颖性或新数据/训练验收。

继承项目根 AGENTS.md。此文件只约束本目录，不改写 A/B/C/UAD 的规范。

- 用户最新将本会话终点明确为两类训练量级数据生产，见data_pipeline/DUAL_PRODUCTION_TRAINING_SCALE_V2.md：普通至少300万有效动作记录；特殊工程规划1万去重完整族/18万交叉结果及至少100万合法动作记录，非同行充分性定理，30/300仅检查点。ordinary_parallel_v1/runtime_v1正准备GPU6/7四分片下一1000路线（auth ORDINARY_PARALLEL_PRODUCTION_V1），不可将此批当最终量。特殊batch00/01r1因资源中断未闭合；batch02占位退出竞态preworker失败，main已恢复GPU5新PID3210151/pane%235；只准新gpu5_transport_v2/batch02r1按新授权重试，不改旧锁。CPU恢复审计独立grade、bulk_source准备可以并行，不训练，保留全部失败与资源账。

- 最新造数状态：short_revisit_v3已独立quality_pass（仅受控FIT数据，不是模型收益），与batch00首候选同hub，不能凑独立数。batch_execution_v1/batch_00 GPU2、batch_01r1 GPU1已按authorizations/WITNESS_CROSS_HOUSE_BATCH_V3.json及WITNESS_CROSS_HOUSE_BATCH_01R1_V1.json启动；只读监测，绝不修改其shared/prepare/来源锁。batch01仅preworker无观测退出，无物理尝试，失败receipt保留。两个scout共6屋已closed，不重复启动。后续以不同hub、至少两事前批次的独立审计为准；新批先规范有限词面，再冻结/实际回放，未知文案不做自由LLM标签。普通batch已closed/GPU5恢复，不训练。

- 持续造数迭代当前见authorizations/WITNESS_SHORT_AND_SCOUT_DATA_V2.json：GPU1 scout跨3个FIT屋真实组件采集；GPU2 short_revisit_v2仅缩续接，原history/阈值/checker不变。revisit_v1因QUERY_LENGTH关闭0族，保留；导出使用candidate_fit_pool而非FIT接口字符串。普通recovery_v4已自然完成1000条、770严格合格，GPU5占位恢复，不再视为活跃。新批配置必须先冻结；CPU质量/批次规划可按当前分工并行，不以组件候选或单屋成功充当稳定量产。

- 最新用户要求持续迭代直到稳定生成特殊样本，按authorizations/WITNESS_FIRST_PERSISTENT_DATA_V1.json执行版本化造数：先witness_first_v1/assembly_v1在GPU2对已冻结真实组件做相同F/L/R计数的实际拼接认证；20项CPU/真实文件Journal测试通过后启动。普通GPU5与旧GPU1不干扰。数据发现可从真实观察选择角色，必须记录选择、在新重放前冻结；不把单屋成功当稳定、多屋候选当合格量，不放宽最终物理/因果门槛，不训练。后续每批仍单独冻结预算/配置；CPU模块可并行委派。

- 最新用户改为特殊数据优先，普通GPU5/recovery_v4后台保持不变、不扩普通多卡。feedback_generation_v1已关闭3候选0族（2资源截断+1配置耗尽），不覆盖。authorizations/SPECIAL_DATA_PRIORITY_COMPACT_V2.json准入compact_loop_v2同三候选有界新构造：四角色可达源起点筛选、提前公共尾中性检查、实际紧凑回返验证，不再执行未使用展开逆轨迹；完整交叉矩阵/汇合/27重放不变。GPU1不动外部进程，XML包含图形渲染，12项CPU测试通过后运行；不训练。CPU诊断可继续分工，旧结果/主线只读。

- 当前数据执行以 authorizations/DATA_GPU_TRANSPORT_AND_SCALE_V2.json 为准：普通批次恢复迁移至 GPU5/recovery_v4，唯一生产者；主agent已复跑14项CPU测试和preflight。只借精确核实的占位并finally恢复，不动GPU2外部任务。用户已批准稳定后多卡造数；扩展须先分片独立输出、统一严格审核、冻结资源预算，不允许多进程共写旧batch。特殊反馈构造仍GPU1有界运行，尚未认证的新族不计训练量。

- 最新“开始生成数据”批准 authorizations/FEEDBACK_MECHANISM_DATA_EXECUTION_V1.json：机制新feedback_generation_v1前3个冻结FIT角色候选，真实状态反馈动作、原交叉族核验，GPU1非独占低资源context并存有界运行；只停自己，绝不操作外部context/占位，旧封存只读，不训练。普通GPU2继续恢复批次；新小CUDA context出现本身不是物理质量失败，按冻结资源上限监控并完整留账。

- 双数据主审reviews/Q35N_DUAL_DATA_PRODUCTION_MAIN_REVIEW_V1：ordinary_scale_v1已启动，50条首片审核发现10个RxR转向位移异常。原失败保留，recovery_v1仅逐路线严格隔离，不放宽阈值；39路线/117指令/7188决策首片验收通过，继续剩余任务。允许恢复向原batch新增未尝试route、append账本和新shard，旧结果/代码不改；实时进度在recovery_v1，不读旧PROGRESS当最新。不要启动第二生产者或干扰GPU2。mechanism_scale_v1 CPU17测试/13哈希通过，43屋1376候选但新合格族0；先反馈式合法轨迹/回返修订，不把候选灌进旧P0冒充量产。

- 最新用户明确要求批量生产两类合格数据，按 authorizations/DATA_PRODUCTION_SCALE_V1.json 执行：ordinary_scale_v1 准入首批1,000新路线/官方train R2R+英语RxR、GPU2空闲卡有界生产及独立回读；mechanism_scale_v1 准入CPU语义组合/质量分级修订，首个新runtime另冻结审核。主agent负责共享状态，可按已有CPU模块并行分工委派机制实现。旧封存只读，不训练、不改环境、不停止外部进程；这条覆盖下方历史“规模规划无生产准入”的本次范围。

- 用户要求至少按同行室内 VLN 数据量准备，见 reviews/Q35N_VLN_DATA_SCALE_SURVEY_V1：1,000 路线仅批次，正式全 R2R/RxR 所选语言 train 配置，基础档规划 300 万有效单步记录、增强档另外至少 1,070 万。规划不等于已生成或与所有论文预算等价；按路线/指令/前缀/动作目标/场景分别去重统计。机制族另计，V3 主线与历史 FAIL 不变。当前仅规模/资产/效率准备，无新生产或长训准入。

- action_balance_v1已完整200更新并封存，不采用为有效修复；accuracy62.65%、STOP0、两卡已恢复。最新主审reviews/Q35N_ACTION_BALANCE_MAIN_REVIEW_V1。用户随后纠正数据/训练规模不足，按SCALE_PRIORITY_CLARIFICATION_ZH调整为普通数据扩量准备、正式训练预算/加速及短接口归因；不要再以10路线95%阻塞数据准备，不从本轮FAIL推断整个Qwen路线失败。没有新生产、安装、GPU或长训自动准入。

- 用户最新“执行下一步”准入 authorizations/SFT_ACTION_BALANCE_SMALLSET_V1.json：仅 action_balance_v1 同10路线/同initial/同顺序双卡200步温和类别加权。GPU5/6核验占位后借用并恢复，旧封存只读，环境不改；不自动全池、闭环或机制训练。权重/源码/顺序须运行前冻结，按原可学性门槛收口。

- 根项目仍是 `/mnt/data_nas/deeprobotics/daiyang/vla`，本路线目录为其下 `projects/qwen35_indoor_nav`。
- 最新主审reviews/Q35N_EFFICIENCY_D1_MAIN_REVIEW_V1：efficiency_run_v1已完成真实双卡200更新并封存，小集accuracy62.48%/macro recall28.54%/STOP0，未过可学性，不启动全池。缓存正确但收益<5%未用；GPU7三角重写虽快1.94x但数值失败，不采用。GPU5/6/7均恢复。下一步先版本化动作类别均衡小集修复；官方kernel和4卡另测，不覆盖任何旧节点、不开机制训练。
- 当前训练期间另准入TRIANGULAR_KERNEL_GPU_BENCH_V1：CPU代数/梯度6测试通过后，GPU7独立最多300秒、0参数更新的真实Qwen局部算子速度/数值检验，写sft_acceptance/triangular_gpu_v1；不改活跃GPU5/6训练或环境。只借核验占位并恢复，不以短测代替完整训练或论文贡献。
- 用户最新要求推进基础导航基座，批准authorizations/SFT_EFFICIENCY_AND_D1_EXECUTION_V1.json：只在sft_acceptance/efficiency_run_v1做真实缓存吞吐/同次反向bucket归约验收，然后原10路线最多200步。新关卡不放宽或改写旧D1的15%更新向量失败；旧代码只读导入且helper输出绑定新目录。GPU5/6仅借核验占位并恢复。不自动全池D2或机制训练。
- D0+D1双卡V1已按停止规则关闭，主审reviews/Q35N_D1_MAIN_REVIEW_V1：D0及NCCL/FP32 toy通过；真实BF16梯度容限通过，但AdamW更新relative L2为0.202/0.163超过0.15，未启动200步D1（0训练更新，另有诊断更新）。GPU5/6占位均恢复。禁止覆盖d1_v1或放宽阈值改PASS；下一步先新版本重复性/逐模块更新归因，不自动重试或进入D2。
- 用户最新批准D0+D1小集与双卡验收，见authorizations/SFT_D0_D1_TWO_GPU_V1.json。主agent只写sft_acceptance/d1_v1，固定10训练路线/565决策、最多200训练更新；GPU5/6仅释放核验占位并恢复。旧SFT已收口，92项封存哈希复核通过，技术接口通过但导航效果未通过；旧v1只读，不再按外部正在训练处理。无全量D2、机制训练或新导航episode准入。
- P0已实际收口：mechanism_runtime_v1/p0_execution_v1，5候选=4个metadata_ineligible+1个resource_censored，0冻结/0认证，2393动作/74探测轨迹；GPU2清理完成。不得重跑覆盖run、自动P1或把0/5当算法否证；后续先CPU语义覆盖/程序组合及提案/持久化效率修订，再新批准入，旧失败保留。SFT分工不变，论文范围见该节点PAPER_SCOPE_CLARIFICATION_ZH.md。
- 本轮“执行下一步”批准authorizations/MECHANISM_P0_EXECUTION_V1.json：一次固定5候选P0实际构造诊断，新增只写mechanism_runtime_v1/p0_execution_v1，旧封存代码/结果只读。主agent核验后精确准入；原预算不变、资源截断不作科学失败，不替换候选、不自动P1、不训练、不干扰外部SFT。
- mechanism_runtime_v1已实测收口：58接口CPU测试+32个P0调度/监控CPU测试；GPU2真实旧单族9轨迹/2877运动动作及V4导出/loader通过，GPU清理完成。旧族仍interface_only，新增独立族0，不覆盖或重跑smoke_v1。P0的五候选232配置与driver已准备，但CONFIG_DRAFT仍runtime_allowed=false；下一步按NEXT_P0_ZH单独审核代码/资产/GPU及实测吞吐再准入，不从CPU通过推断扩量或机制训练授权。外部SFT保持分工。
- 最新“执行下一步”批准authorizations/MECHANISM_RUNTIME_INTEGRATION_V1.json：只在data_pipeline/mechanism_runtime_v1接Habitat/V4导出与loader，按SPEC分阶段主审后验收runtime；CPU子模块可延续并行委派。旧factory_v2封存不改，外部SFT不得干扰。runtime须先登记精确代码/资产/GPU准入，不从草案直接启动P0。
- mechanism_factory_v2 CPU核心节点已收口PASS，80测试；不覆盖acceptance_v1与封存代码。Habitat backend/watchdog/V4生产导出及loader接入仍未完成，下一步先补接入再另准入5候选runtime；不把文件回放backend当物理运行。SFT保持原分工。
- 最新“实现下一步”批准authorizations/GENERIC_FACTORY_CPU_V2.json：只在data_pipeline/mechanism_factory_v2实现通用构造器及CPU测试，延续用户并行分工，可委派互不重叠模块。旧封存/SFT只读，根状态主agent独占，无GPU/仿真/训练准入。
- parallel_readiness/v2三条CPU支线已由主agent接收并收口，43项复跑测试通过；权威裁决main_review/REPORT_ZH.md。保留所有封存产物，当前单族仍interface_only；动作计数捷径、强M2、真实query编码和长历史梯度缺口不得省略。20个路线候选不是生成族，下一步为版本化通用构造器CPU实现，GPU构造诊断需独立准入；不重跑覆盖已关闭节点，不改外部SFT。
- 本路线新文档、代码、环境、缓存、权重、数据、输出和日志均写入本目录；不向旧线注入新导入或配置。
- 最新用户“和SFT多线并行”批准authorizations/PARALLEL_DATA_READINESS_V2.json中的CPU数据审计、扩量协议和loader实现测试；只写parallel_readiness/v2各自子目录。共享封存节点/SFT只读，主agent独占根状态；本轮不启动GPU/仿真/模型或扩量，不干扰外部SFT。允许按此明确并行分工委派子agent。
- 当前机制节点已实测收口：reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1，1个TV/sink V3数值规范化真实族，18cells/27replays/54求值；原餐椅任务和原始精确像素汇合仍失败，不可覆盖。compiler执行源码已锁定，不改封存代码/候选/数据，后续实现另开版本。GPU3已恢复占位，完成节点不自动延长GPU借用或批准机制训练/扩量；SFT仍按独立handoff分工执行。
- 当前用户分工授权见authorizations/PARALLEL_SFT_AND_MECHANISM_V1.json：外部SFT会话仅写sft_acceptance/v1并按交接有界验收；主agent推进版本化真实机制族。共享环境/权重只读、根状态主agent独占；GPU3留给机制线，SFT不得操作它。覆盖下方历史禁令的范围仅以此授权为准。
- 用户“可以尝试”已批准G2最小接口节点，见authorizations/G2_INTERFACE_EXECUTION_V1.json。仅该节点允许独立模型依赖/权重获取和有界forward/backward/最多一次诊断更新，覆盖下方历史模型禁令；不授权完整训练、导航评估或机制造数。
- G2最小接口现已实测收口PASS（reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1），权重/环境保留，1次诊断更新未保存策略checkpoint；不要重跑封存probe。完整训练、真实机制query实现、G1R和导航收益实验仍需后续独立协议准入。
- 2026-09-09 用户“执行下一步”已批准普通导航100候选回放首批，独立范围见 authorizations/ORDINARY_PILOT_EXECUTION_V1.json 与 data_pipeline/ordinary_pilot_v1/SPEC_ZH.md；仅覆盖历史禁止中的该数据编译/回放部分，不授权训练、G2、G1R或扩量。
- 普通pilot现已收口99/100路线通过；GPU3借用按GPU_TRANSPORT_AMENDMENT_V1并完成占位恢复。监控修复与在途重试保留在recovery_v1；不得重新运行已关闭脚本、覆盖结果或自动扩量。训练/G2/G1R继续需明确的下一节点准入。
- G0R 当前已由主 agent 实测接收，证据在 reviews/Q35N_G0R_DEPENDENCY_RECOVERY_V1/。原失败不覆盖；独立 Python 的 sysconfig 重定位修复必须保留。新 session 不重复安装，不复用 quarantine_build_old_prefix，也不从既有环境通过推断 G1F/训练获批；下一节点及权限以 STATUS 为准。
- 用户本轮明确要求执行下一步，G1F 单族有界 discovery/replay 已获局部批准，精确范围见 authorizations/G1F_EXECUTION_AUTHORIZATION_V1.json；覆盖下方历史“family replay未准入”的表述，仅限该节点。G0R环境保持不变，G2、训练、下载和效能实验继续禁止。
- G1F 已执行并按256配置停止规则收口：witness前置通过，family discovery失败，见对应REPORT_ZH/result。不要重复运行已关闭节点、覆盖失败或自动进入G2；后续候选覆盖修订需单独版本与执行准入。
- 不复用旧线可写环境；不修改根 NEXT_EXPERIMENT、FROZEN_SPEC 或旧实验。必要的共享资产仅能经登记后只读访问，不使用跨线可写软链接。
- 开始工作先读 README、STATUS、02、03、04、05。状态不明时不得从文档草案启动训练。
- 当前允许：一手论文/代码与本地证据只读核验、方案文档和状态登记；用户另已批准 `Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1` 的独立环境安装、预算内依赖下载和最小 renderer smoke，精确范围见 `authorizations/G0R_EXECUTION_AUTHORIZATION_V1.json` 与 G0R 交接。算法训练、family replay、Qwen 下载/加载和效能实验仍未准入，不能从环境授权扩大到方法实现。
- novelty 的准入是 `PASS_FOR_SPECIFIED_CLAIM`，必须指向冻结的具体算法差异及最近工作证据。未命中检索不是 PASS；共同思想也不是自动 FAIL。
- 采用顺序科研角色审查；没有用户明确委派时不自动创建子 agent，不冒充独立人工评审。
- 不把仿真元数据、未来帧、目标坐标、场景 ID、路线真值输入部署策略。离线监督与策略输入分文件。
- 原始场景及其所有变体/前缀/措辞/分支整体分组；旧暴露状态与新数据划分分别记录。
- 不将未验收 B15 语义栅格、旧 UAD 模糊标签或 EgoCoT 测试答案作为训练真值。
- 首个试验只检验一个明确机制；每次变更独立版本，失败不得覆盖。基础 SFT、接口通过与真实导航收益分开记载。
- GPU 权限限 G0R：先用空闲卡，无空闲时可只释放经核实的用户后部 GPU 占位进程，并在成功/失败/中断后恢复原占位。停止前保存身份和恢复方法；不得触碰实际任务或用途不明进程、不得批量杀进程。该许可不是其他节点的永久 GPU 控制权。
- 文献主张/算法一旦准入，只有核心变化、直接新近邻或实质遗漏才重审；不反复泛搜，也不隐藏真正重叠。
- 运行文件/路径检查不等于科学验证。未运行指标用 null，不填零、不用预期值冒充结果。
