# Qwen3.5 通用室内导航：独立研究路线

当前入口（2026-09-17）：[针对性修复报告](reviews/Q35N_RECOVERY_20260917/REPORT_ZH.md)、[当前状态](CURRENT_STATUS.json)、[基线推理入口](deployment/ordinary_v1/README_ZH.md)、[未来备选1：小物体语言导航](FUTURE_OPTION_1_SMALL_OBJECT_NAV_ZH.md)。400步小集可学性与独立推理接口已通过；循环恢复对照先因跨运行数值差异停止，后两次因外部GPU任务进入而中断，未采纳部分分数。全部自有GPU进程已关闭，完整SR40目标未达成；本轮按用户要求准备GitHub新分支与Pro交接。

下文及旧 `STATUS.json` 保留历史记录，不作为当前进程或授权状态。

更新：2026-09-09。`Q35N` 是项目代号，不是论文方法名。

最新任务终点：[两类训练量级生产 V2](data_pipeline/DUAL_PRODUCTION_TRAINING_SCALE_V2.md)。普通至少300万有效动作监督；特殊工程规划1万去重完整族/18万交叉结果+至少100万合法动作记录，30/300族不是最终交付。GPU6/7下一1000路线分片正准备；特殊batch00/01r1资源中断，batch02仅占位退出竞态preworker失败且main已实地恢复GPU5占位。当前修复gpu5_transport_v2后以新batch02r1执行，原锁/失败保留；下段与更早“仍运行”为历史，实时以STATUS为准。

当前最新：特殊数据已取得[首个独立质量通过的新族](data_pipeline/mechanism_runtime_v1/witness_first_v1/quality_cpu/batch_acceptance_v1/v3_final_audit/FAMILY_REPORT.json)，尚未证明跨屋稳定。GPU2 batch_00因外部新增负载触发原护栏中断，只清自身，未完成整批封存，不能计合格批；GPU1 batch_01r1继续，另准备GPU5新固定batch02。6屋真实组件采集已收口，候选不计合格族。[质量红队](data_pipeline/mechanism_runtime_v1/witness_first_v1/quality_redteam_v1/REPORT_ZH.md)真实正例及7类损坏负例检查通过。普通1000路线批次自然结束，770严格合格/2307指令/167870决策，GPU5占位已恢复，目前尚未再次借用。[持续迭代记录](data_pipeline/mechanism_runtime_v1/witness_first_v1/ITERATION_LOG_ZH.md)与STATUS为实时入口，下方旧运行表述为历史。

最新优先级：[特殊数据优先 V2](reviews/Q35N_SPECIAL_DATA_PRIORITY_V2/REPORT_ZH.md)。旧反馈批次3候选0族已关闭并审计；新GPU1紧凑真实回返/四角色可达预筛已启动，15项CPU及存储scope测试通过。普通GPU5保持后台生成，不扩普通多卡；先争取完整、反捷径合格的特殊数据族，不把回路/探测数当作新训练族或论文收益。

当前运行：[双数据生产 V2](reviews/Q35N_DUAL_DATA_GENERATION_LIVE_V2/REPORT_ZH.md)。普通1,000路线批次由GPU5/recovery_v4恢复生成，特殊3个冻结候选由GPU1/feedback_generation_v1真实构造；旧失败不覆盖，未认证探测不计训练量。用户已批准稳定后多卡扩量，下一批先准备独立分片，禁止并行共写旧batch；GPU5本次占位恢复尚待当前生产结束。实时/终态见报告链接，下文旧运行状态仅为历史。

## 当前唯一主线

**续接检验驱动的执行记忆学习，用于可恢复的通用室内 VLN。**

让模型记住会改变后续任务完成方式的执行历史，而不只是看过的画面。
从通用预训练 Qwen3.5-2B 训练自己的导航策略，不从随机参数预训练 VLM，也不继续给旧 A/B/C 导航策略加补丁。

已决定 `GO_FOR_IMPLEMENTATION_PLANNING`。具体联合算法的研究预审为 `PASS_FOR_SPECIFIED_CLAIM`；贡献充分性、导航收益、泛化和竞争力均未验证。
本线已完成数据回放和最小模型接口验收；尚无经主agent接收的训练/导航收益，不宣称正收益、SOTA 或已能部署。

## 当前权威入口

1. [主线冻结 V3](MAINLINE_FREEZE_V3.md)：故事、数据编译、模型与损失、否证对照和停止规则。
2. [P1 裁决与近邻映射](reviews/Q35N_P1_PAPER_CORE_ADJUDICATION_V2/REPORT_ZH.md)及[机器结果](reviews/Q35N_P1_PAPER_CORE_ADJUDICATION_V2/result.json)。
3. [项目状态](STATUS.json)：当前授权和下一节点。
4. [机器人导航底座接口约束](ROBOT_NAV_BACKBONE_CONTRACT_V1.md)：未来应用边界，不是当前论文贡献或安全认证。
5. [完整模块、数据、训练与验证工作流](06_END_TO_END_METHOD_AND_VALIDATION_V1.md)：V3 的端到端展开与后续实验顺序，不代表 P2 已完成或授权运行。

核心贡献假设是**真实历史交叉续接的数据构造，与实际运行记忆的联合训练**；预测状态、自动机、BCE 和循环记忆本身不是原创。
必须优于同数据的强任务状态监督，而不只是优于低质量进度描述。

## 下一步固定

最新：[双分支批量数据生产主审](reviews/Q35N_DUAL_DATA_PRODUCTION_MAIN_REVIEW_V1/REPORT_ZH.md)：普通首批1,000新路线/51 FIT屋已启动，R2R和英语RxR各500。首片39路线/117指令/7,188动作监督完成严格验收，10个回读不合格隔离、1个生成拒收；恢复后继续生产，实时以ordinary_scale_v1/recovery_v1/PROGRESS.json及shards为准。特殊线已准备43屋/1,376语义候选，尚无新增真实合格族；下一重点为反馈式合法轨迹与交叉续接完整构造、反捷径验收，再批量生产。原训练失败保留，不训练、不改主线。下方“规模准备无生产准入”已被本次局部数据授权替代。

最新：[室内 VLN 数据规模调研及正式目标](reviews/Q35N_VLN_DATA_SCALE_SURVEY_V1/REPORT_ZH.md)已完成。按用户要求，1,000 路线降为生产批次，正式普通基座覆盖完整 R2R/RxR 所选语言官方训练配置，规划 300 万有效单步监督；增强档对齐 JanusVLN 的额外数据量，另规划至少 1,070 万有效动作对。样本、视频片段与独立路线不混计；这些是待生产目标，不是现有数据或效果 PASS。机制数据与 Qwen3.5-2B 主线保留，下一步为全来源 manifest/缺口与资源预算、训练加速和机制构造修订准备；不自动启动生产/长训。下方 1,000 路线终点表述已被本次规划替代。

最新：[同数据温和加权主审](reviews/Q35N_ACTION_BALANCE_MAIN_REVIEW_V1/REPORT_ZH.md)已收口：双卡200更新，accuracy62.48%→62.65%，STOP仍0，CE变差；不采用为已验证修复，GPU5/6占位恢复。用户指出当前数据/训练量不足，已据[规模优先级纠正](reviews/Q35N_ACTION_BALANCE_MAIN_REVIEW_V1/SCALE_PRIORITY_CLARIFICATION_ZH.md)调整：下一重点为1,000路线/至少20训练房屋的数据扩量准备、正式epoch预算及训练加速，短接口诊断同时规划；不以小集未过否定Qwen路线，不自动启动生产或长训。

最新：[双卡200步训练主审](reviews/Q35N_EFFICIENCY_D1_MAIN_REVIEW_V1/REPORT_ZH.md)已完成。真实归约/保存重载通过，训练29.8分钟、8.94秒/update；小集accuracy62.48%、STOP召回0，导航基座未完成。缓存无明显提速未用；GPU7三角重写数值不合格不采用。GPU5/6/7占位均恢复。下一步优先同小集动作类别均衡，不盲目全池长训；官方算子和4卡效率单独验收。下方“运行待开始”为历史状态。

效率优先补充：[训练缓存与吞吐方案](sft_acceptance/efficiency_v1/PLAN_ZH.md)。冻结处理器/视觉缓存代码已实现，12项CPU测试通过；未接入Qwen runner、未测加速。下一步聚焦真实缓存正确性及训练吞吐，再两卡/四卡同配置数据并行；旧D1等价关卡失败保留，不放宽阈值改PASS。

最新：[D0+D1双卡主审](reviews/Q35N_D1_MAIN_REVIEW_V1/REPORT_ZH.md)已收口。NCCL与FP32梯度通过、D0完成；真实更新等价差异超过冻结阈值，故D1正式训练0步，不能记为小集拟合失败。GPU5/6占位均恢复。下一步先同卡重复性与逐模块梯度/AdamW更新归因，不覆盖旧失败、不放宽阈值、不自动全量训练。

最新：原普通SFT已封存并经主agent读证，92项哈希通过；训练/重载接口通过，但训练后10条闭环均不STOP，导航效果未通过。用户已批准[D0+D1双卡小集验收](sft_acceptance/d1_v1/SPEC_ZH.md)：固定10条训练路线565决策、最多200更新，先查旧terminal与DDP梯度等价，再验证小集可学性。GPU5/6只租借已核验占位并恢复；不启动全量D2或机制训练。旧SFT不修改。

最新：[五场景P0真实诊断已收口](data_pipeline/mechanism_runtime_v1/p0_execution_v1/REPORT_ZH.md)：5/5候选已检查，4个固定角色组合元数据不适用、1个发现资源截断，新增族0；2393次动作、74条探测轨迹，不能充当训练数据或科学负结论。GPU2已清理，外部SFT未干扰。禁止直接P1，下一步先在已记录inventory上修订语义预筛/任务组合，并审计路线可执行性和存储吞吐，再单独准入新批。[论文主体及与UAD/EgoCoT的关系](data_pipeline/mechanism_runtime_v1/p0_execution_v1/PAPER_SCOPE_CLARIFICATION_ZH.md)已明确：统一VLN是目标，交叉续接记忆是主创新假说，CVPR贡献充分性仍待匹配实验。下方P0尚未运行的描述均为历史状态。

最新：[真实后端与V4数据接口验收通过](data_pipeline/mechanism_runtime_v1/REPORT_ZH.md)。58项接口CPU测试、32项P0调度/监控CPU测试通过；GPU2实际9条回放/2877运动动作，RGB/语义/姿态逐帧一致，18格与1242前缀/1410动作监督owner通过回读。GPU已清理，外部SFT未干扰。此为旧单族复现，新增独立族0。五场景232配置已解析，P0保持未执行草案；下一步按[固定P0交接](data_pipeline/mechanism_runtime_v1/NEXT_P0_ZH.md)单独准入，实际认证吞吐与预算需审视。下方“接入未完成”为历史状态。

最新：[通用构造核心CPU实现已接收](data_pipeline/mechanism_factory_v2/REPORT_ZH.md)，80项测试通过，27旧日志/54标签差分一致；候选调度、预算/冻结/几何去重、持久化账本与首批5候选准备入口已实现。无GPU/仿真/SFT操作，无新数据族。剩余Habitat backend、资源watchdog和V4导出/loader接入明确未完成，下一节点先补生产接入再首批5候选真实诊断，不把CPU回读当新回放或导航收益。

最新：[与SFT并行的三条CPU支线已完成并获主审接收](parallel_readiness/v2/main_review/REPORT_ZH.md)：真实loader、单族捷径审计、多族候选协议，主agent独立复跑43项测试通过。20个候选来自5个既有FIT屋，不是20个新族。发现累计动作计数可区分当前18格，强M2 oracle也解释全部标签；首族仍仅interface_only，不能拿拟合当创新收益。下一步优先版本化通用构造器CPU实现，再首批5候选真实构造诊断；实际生成尚未准入。未读取外部SFT结果、未改其协议或操作GPU。[真实长历史与匹配实验准备](parallel_readiness/v2/main_review/MATCHED_EXPERIMENT_DRAFT_ZH.md)另行记录，不能把G2两步toy接口当作207步机制训练已通过。

最新：[首个V3真实机制数据族验收通过](reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1/REPORT_ZH.md)，[主agent数据裁决](MECHANISM_DATA_READINESS_V1.md)。TV/sink→bed实例提供18格真实监督（12正/6负）、27完整物理重放/54求值、1,242条因果prefix记录、474个RGB数组。明确采用一次有界数值共同状态重建；原始精确像素汇合及原餐椅实例仍未通过。仅interface_only，不是模型收益。数据与落盘哈希复核通过，GPU3占位已恢复。

当前分工：[普通SFT交接](HANDOFF_SFT_ACCEPTANCE_V1.md)由用户管理的另一Codex会话按独立协议执行；主agent本轮未训练，真实机制族已收口。不混入SFT已冻结数据，不自动扩大机制训练/生成。下方“机制族待执行”等为历史记录。

最新：[G2最小Qwen接口实测通过](reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/REPORT_ZH.md)。固定2B权重、真实RGB两步前向、8槽记忆、官方MRoPE、早期特征/LoRA/reader梯度与1次诊断更新已验证，峰值torch显存约6.57GiB；GPU占位已恢复。只有合成query接口探针，不是完整M4实现或真实机制监督。下一步制定小规模普通SFT验收，机制G1R单独推进；训练与导航收益未开始。

最新：[普通导航首批生产已验收](data_pipeline/ordinary_pilot_v1/REPORT_ZH.md)。100个固定候选中99条不同路线、5房屋、298条原始指令配对、4,944张不同RGB像素内容已实际生成并通过双回放/导出检查；[数据索引](data_pipeline/ordinary_pilot_v1/TRAINING_INDEX.jsonl)已发布到本地。1条路线偏移失败与1次工程中断完整保留，GPU占位已恢复。下一步建议用现成数据验收Qwen G2接口，同时机制G1R仍待单独执行；没有训练、机制族或导航收益PASS。

新增：[数据生产管线 V1](data_pipeline/v1/PIPELINE_ZH.md)已确定来源、双分支与分批质量门槛；CPU 来源清单已实际生成（10,819 个任务/3,603 条参考路线/61 房屋），不是新 RGB 或训练标签。普通示范回放与机制族修复可分别推进；尚未启动新的仿真或训练。

最新：[G1F单族实测裁决](reviews/Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2/REPORT_ZH.md)已收口。四类事件证据通过，256配置未构造出合格历史族；实际只覆盖2个起点，水槽路线同时触发餐椅事件。没有冻结族、正式certification或训练。下一步推荐有界修正候选覆盖，保持主线与标签不变；原G1F停止规则不继续突破。

最新运行结果：[主 agent G0R 恢复与实测验收](reviews/Q35N_G0R_DEPENDENCY_RECOVERY_V1/REPORT_ZH.md)已通过。官方 22 个 wheel、独立 Habitat 导入、17DRP5sb8fy 的 RGB/semantic 与一次转向均已实测；首次失败及路径重定位修复完整保留。GPU 无自身残留，未停止占位。下一节点直接为 G1F 一个真实历史—续接族的数据验收，尚未自动批准执行；Qwen/训练/导航收益仍未测试。下方安装未开始的表述为历史状态。

最新：P2R1 已获 [主 agent 静态接收](reviews/Q35N_P2R1_MAIN_AGENT_ACCEPTANCE_V1/REPORT_ZH.md)，P2 主体规划结束。用户已确认 MP3D 合法取得、有授权文件并批准预算，[G0R 环境与渲染验收](HANDOFF_G0R_RUNTIME_SETUP_ACCEPTANCE_V1.md)现可执行。GPU 先空闲，必要时仅借用经核实的后部占位卡并恢复；详见 [本次授权](authorizations/G0R_EXECUTION_AUTHORIZATION_V1.json)。主 agent 本轮只登记授权，没有启动安装/运行。以下 P2/P2R1 指令为历史记录，不覆盖当前 G0R 精确许可。

主 agent 已验收 P2：保留环境/数据首选，但原 G1 草案存在具体规格矛盾，尚未批准执行。当前只按 [P2R1 修订交接](HANDOFF_P2R1_SPEC_CORRECTIONS_V1.md) 关闭七项问题，详见 [主 agent 审核](reviews/Q35N_P2_MAIN_AGENT_REVIEW_V1/REPORT_ZH.md)。下方原 P2 说明为历史范围；当前节点以 STATUS 为准。

执行会话使用 [P2 完整交接 prompt](HANDOFF_P2_DATA_AND_IMPLEMENTATION_PLAN_V1.md)。只写节点产物并回交；主 agent 审核后才更新准入，不由执行会话自行启动下游。

`Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1`：只读核验项目内资产与官方接口，确定一个最小合法历史—续接族、可验证事件、恢复示范、输入隔离、三个代码模块接口及资源预算。
完成数据验收规格后再进入对应实现节点；当前不安装、下载、编写算法实现、启动仿真/训练或操作 GPU。
不例行重开方向搜索；直接先例、核心改变或实质科学漏洞才触发有界重审。

全部本线新内容只写入本目录。旧结果保留，不继承旧 PASS。

## 历史记录与仍适用的约束

- [V2 候选重审](PAPER_DIRECTION_REVIEW_V2.md)已由 V3 完成裁决。
- [V1 贡献边界](CONTRIBUTION_BOUNDARY_V1.md)与[旧审查](reviews/Q35N_G0_CONTRIBUTION_REVIEW_V1.md)保留；预算反转已降为历史组件，不作为主方法。
- [旧 G0B 交接](HANDOFF_G0B_DATA_INTERFACE_AUDIT_V1.md)继续暂停，不执行其中 V1 专属任务。
- [初始故事](01_RESEARCH_STORY.md)、[初始方法预审](02_METHOD_PREFLIGHT.md)、[数据契约](03_DATA_CONTRACT.md)、[初始验证计划](04_VALIDATION_PLAN.md)、[隔离约定](05_ISOLATION_AND_HANDOFF.md)、[来源登记](REFERENCES.md)保留可复用约束；候选及下一步以 V3、STATUS 为准。

scientific-critical-thinking 技能用于区分研究投入决定与经验结论，方法引用已列于 P1 报告。
根 Git 忽略规则仍忽略本目录；文件已落盘，不代表已提交或备份。
